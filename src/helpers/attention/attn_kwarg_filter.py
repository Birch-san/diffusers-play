from dataclasses import dataclass, field
from diffusers.models.attention import Attention
from torch import FloatTensor, BoolTensor
from typing import Optional, Set, Dict, Any
import inspect

from .attn_processor import KwargsAttnProcessor

expected_params: Set[str] = {'attn', 'hidden_states', 'encoder_hidden_states', 'attention_mask', 'temb'}

# we may pass cross_attention_kwargs to UNet2DConditionModel#forward that not all AttnProcessor implementations utilize.
# for example when using a variety of AttnProcessors (e.g. local attention at high resolutions, or dispatch attention
# to change attention strategy as a function of timestep), we may pass in a sigma cross_attention_kwarg.
# we may also pass in the TransformerBlock2D's un-normed hidden states to support ToMe attention processors.
# we don't want to have to modify diffusers' built-in AttnProcessor2_0 signature just to tell it to ignore the extra kwargs.
# we don't want to have to make subclasses of every AttnProcessor to make them ignore extra kwargs. better to wrap them.
# so we employ this filter, which only passes kwargs down if the wrapped AttnProcessor is expecting them.
@dataclass
class AttnKwargFilter(KwargsAttnProcessor):
  delegate: KwargsAttnProcessor
  expected_extra_params: Set[str] = field(init=False)
  def __post_init__(self):
    delegate_params: Set[str] = set(inspect.signature(self.delegate).parameters.keys())
    self.expected_extra_params = delegate_params - expected_params

  def __call__(
    self,
    attn: Attention,
    hidden_states: FloatTensor,
    encoder_hidden_states: Optional[FloatTensor] = None,
    attention_mask: Optional[BoolTensor] = None,
    temb: Optional[FloatTensor] = None,
    **kwargs,
  ) -> FloatTensor:
    supported_kwargs: Dict[str, Any] = {k:v for k,v in kwargs.items() if k in self.expected_extra_params}
    if supported_kwargs:
      pass
    return self.delegate(
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      temb=temb,
      **supported_kwargs
    )
