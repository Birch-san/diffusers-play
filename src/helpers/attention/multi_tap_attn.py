from .tap_attn import TapAttn
from typing import Iterable
from diffusers.models.attention import Attention

def make_multi_tap_attn(taps: Iterable[TapAttn]) -> TapAttn:
  def tap_attn(attn: Attention) -> None:
    for tap in taps:
      tap(attn)
  return tap_attn