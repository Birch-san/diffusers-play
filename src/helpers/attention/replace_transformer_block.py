from torch import FloatTensor, LongTensor
from torch.nn import Module
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from typing import Any, Protocol, Optional, Dict

class TransformerBlockCompatible(Protocol):
  def __call__(
    self,
    hidden_states: FloatTensor,
    attention_mask: Optional[FloatTensor] = None,
    encoder_hidden_states: Optional[FloatTensor] = None,
    encoder_attention_mask: Optional[FloatTensor] = None,
    timestep: Optional[LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    class_labels: Optional[LongTensor] = None,
  ) -> Module: ...

class GetTransformerBlock(Protocol):
  @staticmethod
  def __call__(block: BasicTransformerBlock) -> TransformerBlockCompatible: ...

def _replace_transformer_block(
  module: Module,
  get_replacement: GetTransformerBlock
) -> None:
  for name, m in module.named_children():
    _replace_transformer_block(m, get_replacement)
    if isinstance(m, BasicTransformerBlock):
      replacement: TransformerBlockCompatible = get_replacement(m)
      setattr(module, name, replacement)
      

def replace_transformer_block(
  unet: UNet2DConditionModel,
  get_replacement: GetTransformerBlock,
) -> None:
  _replace_transformer_block(unet, get_replacement)
    