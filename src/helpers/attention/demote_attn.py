from diffusers.models.attention import Attention
from .null_attn import NullAttnProcessor
from .natten_attn import NattenAttnProcessor
from .qkv_fusion import fuse_qkv

def to_null_attn(attn: Attention) -> None:
  del attn.to_q, attn.to_k
  null_attn = NullAttnProcessor()
  attn.set_processor(null_attn)

def to_neighbourhood_attn(attn: Attention, kernel_size=7) -> None:
  fuse_qkv(attn)
  natten = NattenAttnProcessor(kernel_size=kernel_size)
  attn.set_processor(natten)