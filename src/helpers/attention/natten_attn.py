from diffusers.models.attention import Attention
import torch
from torch import FloatTensor, BoolTensor
from typing import Optional
from einops import rearrange

try:
    import natten
except ImportError:
    natten = None

class NattenAttnProcessor:
    r"""
    Processor for implementing local neighbourhood attention via NATTEN
    Based on:
    https://github.com/huggingface/diffusers/blob/3105c710ba16fa2cf54d8deb158099a4146da511/src/diffusers/models/attention_processor.py
    Once complete: this will make query tokens attend only to key tokens within a certain distance (local neighbourhood).
    """
    kernel_size: int

    def __init__(self, kernel_size: int):
        if natten is None:
            raise ModuleNotFoundError("natten is required for neighborhood attention")
        self.kernel_size = kernel_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: FloatTensor,
        encoder_hidden_states: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        temb: Optional[FloatTensor] = None,
    ):
        assert hasattr(attn, 'qkv'), "Did not find property qkv on attn. Expected you to fuse its q_proj, k_proj, v_proj weights and biases beforehand, and multiply attn.scale into the q weights and bias."
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        # assumes MHA (as opposed to GQA)
        inner_dim: int = attn.qkv.out_features // 3

        if attention_mask is not None:
            raise ValueError("No mask customization for neighbourhood attention; the mask is already complicated enough as it is")
        if encoder_hidden_states is not None:
            raise ValueError("NATTEN supports self-cross-attention (https://github.com/SHI-Labs/NATTEN/issues/82), but stable-diffusion doesn't use it so I haven't implemented support here. It's more of a DeepFloyd IF or Imagen thing.")

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)
            hidden_states = rearrange(hidden_states, '... c h w -> ... h w c')

        qkv = attn.qkv(hidden_states)
        # assumes MHA (as opposed to GQA)
        q, k, v = rearrange(qkv, "n h w (t nh e) -> t n nh h w e", t=3, e=inner_dim)

        qk = natten.functional.natten2dqk(q, k, self.kernel_size, 1)
        a = torch.softmax(qk, dim=-1)
        hidden_states = natten.functional.natten2dav(a, v, self.kernel_size, 1)
        hidden_states = rearrange(hidden_states, "n nh h w e -> n h w (nh e)")

        linear_proj, dropout = attn.to_out
        hidden_states = linear_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        hidden_states = rearrange(hidden_states, '... h w c -> ... c h w')

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states