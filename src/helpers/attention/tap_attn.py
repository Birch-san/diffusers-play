from torch import nn
from functools import partial
from diffusers.models.attention import CrossAttention
from ..tap.tap import Tap
from ..tap.tap_module import TapModule

TapAttn = Tap[CrossAttention]

def _tap_attn(tap_attn: TapAttn, module: nn.Module) -> None:
  for m in module.children():
    if isinstance(m, CrossAttention):
      tap_attn(m)

def tap_attn_to_tap_module(tap_attn: TapAttn) -> TapModule:
  return partial(_tap_attn, tap_attn)