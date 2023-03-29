from torch import nn
from typing import Callable
from functools import partial
from diffusers.models.attention import Attention
from ..tap.tap_module import TapModule
from .attn_compatible import AttnCompatible

ReplaceAttn = Callable[[Attention], AttnCompatible]

def _replace_attn(replace_module: ReplaceAttn, module: nn.Module) -> None:
  for name, m in module.named_children():
    if isinstance(m, Attention):
      replacement: AttnCompatible = replace_module(m)
      setattr(module, name, replacement)

def replace_attn_to_tap_module(tap_attn: ReplaceAttn) -> TapModule:
  return partial(_replace_attn, tap_attn)