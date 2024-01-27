from diffusers.models.attention import Attention
import torch
from torch import FloatTensor, BoolTensor
from typing import Optional, NamedTuple
from einops import rearrange
from dataclasses import dataclass, field
import math

from .attn_processor import AttnProcessor

try:
    import natten
except ImportError:
    natten = None

class Dimension(NamedTuple):
    height: int
    width: int

@dataclass
class SelfToMeSelfAttnProcessor(AttnProcessor):
    r"""
    Processor for implementing local neighbourhood attention via NATTEN
    Based on:
    https://github.com/huggingface/diffusers/blob/3105c710ba16fa2cf54d8deb158099a4146da511/src/diffusers/models/attention_processor.py
    Once complete: this will make query tokens attend only to key tokens within a certain distance (local neighbourhood).
    """
    kernel_size: int
    # tell me on construction what size to expect, so I can unflatten the sequence into height and width again
    expect_size: Dimension
    global_subsample: int
    has_fused_scale_factor: bool = False
    has_fused_qkv: bool = False
    # our key size (the kernel area) is smaller than the key size used in training (entire canvas),
    # so the attention softmax is dividing by fewer elements and thus elements become too large / have too much variance
    # in other words, the softmax outputs too sharp a distribution of probabilities (closer to a one-hot vector).
    # we can multiply by a quotient smaller than 1, to compensate. this reduces logit variance,
    # makes attn probabilities more diffuse, more entropic, to try to match the training distribution more closely.
    # the only reason this defaults to False is because it's obscure. it's principled and seems to help, so you should turn it on.
    scale_attn_entropy: bool = False
    train_key_len: int = field(init=False)
    kernel_area: int = field(init=False)

    def __post_init__(self):
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        # Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis
        # https://arxiv.org/abs/2306.08645
        # instead of scaling logits by:
        #   self.scale
        # we scale by:
        #   self.scale * log(inference_key_len, train_key_len)**.5
        # if your train/inference key lengths can be as short as 1, then clamp key lengths to avoid an entropy scale of 0 or infinite
        #                log(max(inference_key_len, 2), max(train_key_len, 2))**.5
        self.train_key_len = self.expect_size.height * self.expect_size.width
        self.kernel_area = self.kernel_size**2
        assert self.global_subsample > 1

    def __call__(
        self,
        attn: Attention,
        hidden_states: FloatTensor,
        encoder_hidden_states: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        temb: Optional[FloatTensor] = None,
    ):
        assert hidden_states.ndim == 3, f"Expected a disappointing 3D tensor that I would have the fun job of unflattening. Instead received {hidden_states.ndim}-dimensional tensor."
        assert hidden_states.size(-2) == self.expect_size.height * self.expect_size.width, "Sequence dimension is not equal to the product of expected height and width, so we cannot unflatten sequence into 2D sequence."
        
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if attention_mask is not None:
            raise ValueError("No mask customization for neighbourhood attention; the mask is already complicated enough as it is")
        if encoder_hidden_states is not None:
            raise ValueError("NATTEN supports self-cross-attention (https://github.com/SHI-Labs/NATTEN/issues/82), but stable-diffusion doesn't use it so I haven't implemented support here. It's more of a DeepFloyd IF or Imagen thing.")

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)
            hidden_states = rearrange(hidden_states, '... c (h w) -> ... (h w) c')

        if self.has_fused_qkv:
            assert hasattr(attn, 'qkv'), "Did not find property qkv on attn. Expected you to fuse its q_proj, k_proj, v_proj weights and biases beforehand, and multiply attn.scale into the q weights and bias."
            qkv = attn.qkv(hidden_states)
            # assumes MHA (as opposed to GQA)
            # assumes that the scale factor has already been fused into the weights
            q, k, v = rearrange(qkv, "n (h w) (t nh e) -> t n nh h w e", t=3, nh=attn.heads, h=self.expect_size.height, w=self.expect_size.width)
        else:
            q = attn.to_q(hidden_states)
            k = attn.to_k(hidden_states)
            v = attn.to_v(hidden_states)
            q, k, v = [rearrange(p, "n (h w) (nh e) -> n nh h w e", nh=attn.heads, h=self.expect_size.height, w=self.expect_size.width) for p in (q, k, v)]

        k_sub, v_sub = [p[:,:,::self.global_subsample,::self.global_subsample,:].flatten(start_dim=-3, end_dim=-2).contiguous() for p in (k, v)]

        if self.scale_attn_entropy or not self.has_fused_scale_factor:
            scale = 1. if self.has_fused_scale_factor else attn.scale
            if self.scale_attn_entropy:
                subsamp_area: int = k_sub.size(-3) * k_sub.size(-2)
                inference_key_len: int = self.kernel_area + subsamp_area
                entropy_scale: float = math.log(inference_key_len, self.train_key_len)**.5
                scale *= entropy_scale
            q *= scale

        qk = natten.functional.natten2dqk(q, k, self.kernel_size, 1, additional_keys=k_sub)
        a = torch.softmax(qk, dim=-1)
        hidden_states = natten.functional.natten2dav(a, v, self.kernel_size, 1, additional_values=v_sub)
        hidden_states = rearrange(hidden_states, "n nh h w e -> n h w (nh e)")

        out_proj, dropout = attn.to_out
        hidden_states = out_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        if attn.group_norm is not None:
            hidden_states = rearrange(hidden_states, '... (h w) c -> ... c (h w)')

        hidden_states = rearrange(hidden_states, '... h w c -> ... (h w) c')

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states