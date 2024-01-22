from dataclasses import dataclass
from diffusers.models.attention import Attention
from torch import FloatTensor, BoolTensor
from typing import Optional
from .attn_processor import SigmaAttnProcessor, AttnProcessor

@dataclass
class SigmaSwallower(SigmaAttnProcessor):
  delegate: AttnProcessor
  def __call__(
    self,
    attn: Attention,
    hidden_states: FloatTensor,
    sigma: float,
    encoder_hidden_states: Optional[FloatTensor] = None,
    attention_mask: Optional[BoolTensor] = None,
    temb: Optional[FloatTensor] = None,
  ) -> FloatTensor:
    return self.delegate(
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      temb=temb,
    )
