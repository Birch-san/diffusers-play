from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, CrossAttnUpBlock2D, UNetMidBlock2DCrossAttn, Transformer2DModel
from diffusers.models.attention import Attention, BasicTransformerBlock
from typing import Protocol

class AttnAcceptor(Protocol):
  @staticmethod
  def __call__(attn: Attention) -> None: ...

def visit_t2d(t2d: Transformer2DModel, attn_acceptor: AttnAcceptor) -> None:
  for tblock in t2d.transformer_blocks:
    assert isinstance(tblock, BasicTransformerBlock)
    attn_acceptor(tblock.attn1)

def visit_attns(unet: UNet2DConditionModel, levels: int, attn_acceptor: AttnAcceptor) -> None:
  """
  counting from outermost level of UNet, how many levels to apply modifications to
  """
  for down_block, _ in zip(unet.down_blocks, range(levels)):
    if isinstance(down_block, CrossAttnDownBlock2D):
      for t2d in down_block.attentions:
        visit_t2d(t2d, attn_acceptor)
  
  if levels > len(unet.down_blocks):
    if isinstance(unet.mid_block, UNetMidBlock2DCrossAttn):
      for t2d in unet.mid_block.attentions:
        visit_t2d(t2d, attn_acceptor)

  for up_block, _ in zip(reversed(unet.up_blocks), range(levels)):
    if isinstance(up_block, CrossAttnUpBlock2D):
      for t2d in up_block.attentions:
        visit_t2d(t2d, attn_acceptor)