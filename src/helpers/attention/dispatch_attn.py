from diffusers.models.attention import Attention
from dataclasses import dataclass
from torch import FloatTensor, BoolTensor
from typing import Protocol, Optional

from .attn_processor import SigmaAttnProcessor, AttnProcessor

class PickAttnDelegate(Protocol):
  @staticmethod
  def __call__(sigma: float) -> AttnProcessor: ...

@dataclass
class DispatchAttnProcessor(SigmaAttnProcessor):
  pick_delegate: PickAttnDelegate

  def __call__(
    self,
    attn: Attention,
    hidden_states: FloatTensor,
    sigma: float,
    encoder_hidden_states: Optional[FloatTensor] = None,
    attention_mask: Optional[BoolTensor] = None,
    temb: Optional[FloatTensor] = None,
  ) -> FloatTensor:
    delegate: AttnProcessor = self.pick_delegate(sigma)
    out: FloatTensor = delegate(
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      temb=temb,
    )
    return out
