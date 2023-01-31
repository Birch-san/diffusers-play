from torch import nn
from diffusers.models.attention import CrossAttention
from .to_mha import to_mha
from .multi_head_attention import MultiheadAttention

def replace_attn_mha(module: nn.Module) -> None:
  for name, m in module.named_children():
    if isinstance(m, CrossAttention):
      mha: MultiheadAttention = to_mha(m)
      setattr(module, name, mha)