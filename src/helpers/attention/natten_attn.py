from diffusers.models.attention import Attention
import torch
from torch import FloatTensor, BoolTensor
from typing import Optional, NamedTuple
from einops import rearrange
from dataclasses import dataclass

try:
    import natten
except ImportError:
    natten = None

class Dimension(NamedTuple):
    height: int
    width: int

@dataclass
class NattenAttnProcessor:
    r"""
    Processor for implementing local neighbourhood attention via NATTEN
    Based on:
    https://github.com/huggingface/diffusers/blob/3105c710ba16fa2cf54d8deb158099a4146da511/src/diffusers/models/attention_processor.py
    Once complete: this will make query tokens attend only to key tokens within a certain distance (local neighbourhood).
    """
    kernel_size: int
    # tell me on construction what size to expect, so I can unflatten the sequence into height and width again
    expect_size: int

    def __post_init__(self):
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")

    def __call__(
        self,
        attn: Attention,
        hidden_states: FloatTensor,
        encoder_hidden_states: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        temb: Optional[FloatTensor] = None,
    ):
        assert hasattr(attn, 'qkv'), "Did not find property qkv on attn. Expected you to fuse its q_proj, k_proj, v_proj weights and biases beforehand, and multiply attn.scale into the q weights and bias."
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

        qkv = attn.qkv(hidden_states)
        # assumes MHA (as opposed to GQA)
        q, k, v = rearrange(qkv, "n (h w) (t nh e) -> t n nh h w e", t=3, nh=attn.heads, h=self.expect_size.height, w=self.expect_size.width)

        qk = natten.functional.natten2dqk(q, k, self.kernel_size, 1)
        a = torch.softmax(qk, dim=-1)
        hidden_states = natten.functional.natten2dav(a, v, self.kernel_size, 1)
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