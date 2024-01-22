from diffusers.models.attention import Attention
from torch import FloatTensor, BoolTensor
from typing import Protocol, Optional

class AttnProcessor(Protocol):
  def __call__(
    self,
    attn: Attention,
    hidden_states: FloatTensor,
    encoder_hidden_states: Optional[FloatTensor] = None,
    attention_mask: Optional[BoolTensor] = None,
    temb: Optional[FloatTensor] = None,
  ) -> FloatTensor: ...

class SigmaAttnProcessor(Protocol):
  def __call__(
    self,
    attn: Attention,
    hidden_states: FloatTensor,
    sigma: float,
    encoder_hidden_states: Optional[FloatTensor] = None,
    attention_mask: Optional[BoolTensor] = None,
    temb: Optional[FloatTensor] = None,
  ) -> FloatTensor: ...