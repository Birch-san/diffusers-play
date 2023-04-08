from diffusers.models.attention import Attention
from .tap_attn import TapAttn

def make_set_sigma(sigma: float) -> TapAttn:
  def set_sigma(module: Attention) -> None:
    module.sigma = sigma
  return set_sigma
