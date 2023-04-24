from diffusers.models.attention import Attention
from .tap_attn import TapAttn

def make_set_self_attn_aspect_ratio(self_attn_aspect_ratio = 1.) -> TapAttn:
  def set_self_attn_aspect_ratio(module: Attention) -> None:
    module.self_attn_aspect_ratio = self_attn_aspect_ratio
  return set_self_attn_aspect_ratio
