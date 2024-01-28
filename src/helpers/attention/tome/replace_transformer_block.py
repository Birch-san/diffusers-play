from diffusers.models import UNet2DConditionModel
from ..replace_transformer_block import replace_transformer_block, TransformerBlockCompatible, GetTransformerBlock
from ..transformer_block_unnorm import TransformerBlockUnNorm

def set_transformer_block_with_tome_support(unet: UNet2DConditionModel) -> None:
  get_replacement: GetTransformerBlock = lambda block: TransformerBlockUnNorm(block)
  replace_transformer_block(unet, get_replacement=get_replacement)