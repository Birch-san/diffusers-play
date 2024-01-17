from diffusers.models.attention import Attention
from .null_attn import NullAttnProcessor
from .natten_attn import NattenAttnProcessor, Dimension
from .qkv_fusion import fuse_qkv

def to_null_attn(attn: Attention, level: int) -> None:
  del attn.to_q, attn.to_k
  null_attn = NullAttnProcessor()
  attn.set_processor(null_attn)

def to_neighbourhood_attn(attn: Attention, level: int, sample_size: Dimension, kernel_size=7) -> None:
  fuse_qkv(attn)
  downsampled_size = sample_size
  # yes I know about raising 2 to the power of negative number, but I want to model a repeated rounding-down
  for _ in range(level):
    # haven't actually tested this for levels>1 so I've never run this code
    downsampled_size = Dimension(
      height=downsampled_size.height>>1,
      width=downsampled_size.width>>1,
    )
  natten = NattenAttnProcessor(kernel_size=kernel_size, expect_size=downsampled_size)
  attn.set_processor(natten)