from typing import Optional
from diffusers.models.attention import Attention
from .tap_attn import TapAttn

def make_set_chunked_attn(
  query_chunk_size = 1024,
  kv_chunk_size: Optional[int] = 4096,
  kv_chunk_size_min: Optional[int] = None,
  chunk_threshold_bytes: Optional[int] = None,
) -> TapAttn:
  def set_chunked_attn(module: Attention) -> None:
    module.set_subquadratic_attention(
      query_chunk_size=query_chunk_size,
      kv_chunk_size=kv_chunk_size,
      kv_chunk_size_min=kv_chunk_size_min,
      chunk_threshold_bytes=chunk_threshold_bytes,
    )
  return set_chunked_attn
