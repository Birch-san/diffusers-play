from diffusers.models.attention import Attention
from .tap_attn import TapAttn

def make_set_key_length_factor(
  self_attn_key_length_factor = 1.,
  cross_attn_key_length_factor = 1.,
) -> TapAttn:
  def set_key_length_factor(module: Attention) -> None:
    module.key_length_factor = self_attn_key_length_factor if module.is_self_attention else cross_attn_key_length_factor
  return set_key_length_factor
