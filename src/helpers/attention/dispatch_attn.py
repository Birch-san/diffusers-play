from diffusers.models.attention import Attention
from dataclasses import dataclass
from torch import FloatTensor, BoolTensor
from typing import Protocol, Optional

from .attn_processor import AttnProcessor

class PickAttnDelegate(Protocol):
  def __call__(self, sigma: float) -> AttnProcessor: ...

@dataclass
class DispatchAttnProcessor(AttnProcessor):
  pick_delegate: PickAttnDelegate

  def __call__(
    self,
    attn: Attention,
    hidden_states: FloatTensor,
    encoder_hidden_states: Optional[FloatTensor] = None,
    attention_mask: Optional[BoolTensor] = None,
    temb: Optional[FloatTensor] = None,
    **kwargs,
  ) -> FloatTensor:
    assert 'sigma' in kwargs
    sigma: float = kwargs['sigma']
    assert type(sigma) == float
    delegate: AttnProcessor = self.pick_delegate(sigma)
    out: FloatTensor = delegate(
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      temb=temb,
      **kwargs,
    )
    return out
