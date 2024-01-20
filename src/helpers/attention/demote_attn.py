from diffusers.models.attention import Attention
from .null_attn import NullAttnProcessor
from .natten_attn import NattenAttnProcessor, Dimension
from .qkv_fusion import fuse_qkv as fuse_qkv_

def to_null_attn(attn: Attention, level: int, delete_qk=False) -> None:
  if delete_qk:
    del attn.to_q, attn.to_k
  null_attn = NullAttnProcessor()
  attn.set_processor(null_attn)

def to_neighbourhood_attn(
  attn: Attention,
  level: int,
  sample_size: Dimension,
  kernel_size=7,
  scale_attn_entropy=False,
  fuse_qkv=False,
  qkv_fusion_fuses_scale_factor=False,
) -> None:
  if fuse_qkv:
    fuse_qkv_(attn, fuse_scale_factor=qkv_fusion_fuses_scale_factor)
  downsampled_size = sample_size
  # yes I know about raising 2 to the power of negative number, but I want to model a repeated rounding-down
  for _ in range(level):
    # haven't actually tested this for levels>1 so I've never run this code
    downsampled_size = Dimension(
      height=downsampled_size.height>>1,
      width=downsampled_size.width>>1,
    )
  natten = NattenAttnProcessor(
    kernel_size=kernel_size,
    expect_size=downsampled_size,
    has_fused_scale_factor=fuse_qkv and qkv_fusion_fuses_scale_factor,
    has_fused_qkv=fuse_qkv,
    scale_attn_entropy=scale_attn_entropy,
  )
  attn.set_processor(natten)