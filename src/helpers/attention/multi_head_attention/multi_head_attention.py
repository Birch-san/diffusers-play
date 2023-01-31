from torch import nn, Tensor
from typing import Optional
from ..attn_compatible import CrossAttnCompatible

class MultiheadAttention(nn.MultiheadAttention, CrossAttnCompatible):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
    ):
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        super().__init__(
            embed_dim=inner_dim,
            num_heads=heads,
            dropout=dropout,
            bias=bias,
            batch_first=True,
            kdim=cross_attention_dim,
            vdim=cross_attention_dim,
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        # TODO: pass in cross_attn_mask
        cross_attn_mask: Optional[Tensor] = None,
        **cross_attention_kwargs,
    ) -> Tensor:
        kv = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        out, _ = super().forward(
            query=hidden_states,
            key=kv,
            value=kv,
            need_weights=False,
        )
        return out
