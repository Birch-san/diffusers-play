from diffusers.models.attention import Attention
from torch import FloatTensor, BoolTensor
from typing import Optional
from einops import rearrange

from .attn_processor import AttnProcessor

class NullAttnProcessor(AttnProcessor):
    r"""
    Processor for skipping SDP attn, for cases where we suspect it's learned to drop out the token-mixing capability
    Based on:
    https://github.com/huggingface/diffusers/blob/3105c710ba16fa2cf54d8deb158099a4146da511/src/diffusers/models/attention_processor.py
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: FloatTensor,
        encoder_hidden_states: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        temb: Optional[FloatTensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if attention_mask is not None:
            raise ValueError("attn key masking not implemented, because this experiment altogether disables mixing of values based on qk-similarity")
        if encoder_hidden_states is not None:
            raise ValueError("cross-attn not implemented, because this experiment altogether disables mixing of values based on qk-similarity")

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states)
            hidden_states = rearrange(hidden_states, '... channels tokens -> ... tokens channels')
        
        v = attn.to_v(hidden_states)
        v = rearrange(v, "batch tokens (heads head_dim) -> batch heads tokens head_dim", heads=attn.heads)
        hidden_states = rearrange(v, "batch heads tokens head_dim -> batch tokens (heads head_dim)")
        del v

        linear_proj, dropout = attn.to_out
        hidden_states = linear_proj(hidden_states)
        hidden_states = dropout(hidden_states)

        if attn.group_norm is not None:
            hidden_states = rearrange(hidden_states, '... tokens channels -> ... channels tokens')

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states