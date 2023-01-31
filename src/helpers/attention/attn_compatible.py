from torch import Tensor
from typing import Protocol, Optional

class CrossAttnCompatible(Protocol):
  def forward(
    self,
    hidden_states: Tensor,
    encoder_hidden_states: Optional[Tensor] = None,
    attention_mask: Optional[Tensor] = None,
    cross_attn_mask: Optional[Tensor] = None,
    **cross_attention_kwargs,
  ) -> Tensor: ...