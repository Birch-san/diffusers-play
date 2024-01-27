from diffusers.models.attention import Attention
from diffusers.models.attention_processor import AttnProcessor2_0
from typing import Protocol
import torch
from .null_attn import NullAttnProcessor
from .natten_attn import NattenAttnProcessor, Dimension
from .selfsubself_attn import SelfSubSelfAttnProcessor
from .selftomeself_attn import SelfToMeSelfAttnProcessor
from .dispatch_attn import DispatchAttnProcessor, PickAttnDelegate
from .sigma_swallower import SigmaSwallower
from .qkv_fusion import fuse_qkv as fuse_qkv_
from .attn_processor import AttnProcessor

class GetAttnProcessor(Protocol):
  @staticmethod
  def __call__(self, attn: Attention, level: int) -> AttnProcessor: ...

def set_attn_processor(attn: Attention, level: int, get_attn_processor: GetAttnProcessor) -> None:
  attn_processor: AttnProcessor = get_attn_processor(attn, level)
  attn.set_processor(attn_processor)

def make_null_attn(attn: Attention, level: int, delete_qk=False) -> NullAttnProcessor:
  if delete_qk:
    del attn.to_q, attn.to_k
  null_attn = NullAttnProcessor()
  return null_attn

def make_default_attn(attn: Attention, level: int) -> AttnProcessor2_0:
  return AttnProcessor2_0()

def make_neighbourhood_attn(
  attn: Attention,
  level: int,
  sample_size: Dimension,
  kernel_size=7,
  scale_attn_entropy=False,
  fuse_qkv=False,
  qkv_fusion_fuses_scale_factor=False,
) -> NattenAttnProcessor:
  if fuse_qkv:
    fuse_qkv_(attn, fuse_scale_factor=qkv_fusion_fuses_scale_factor)
  downsampled_size = sample_size
  # yes I know about raising 2 to the power of negative number, but I want to model a repeated rounding-down
  downsample = torch.nn.Conv2d(1,1, kernel_size=3, stride=2, padding=1)
  size_probe = torch.ones(1,sample_size.height,sample_size.width)
  for _ in range(level):
    # haven't actually tested this for levels>1 so I've never run this code
    size_probe = downsample(size_probe)
    height, width = size_probe.shape[1:]
    downsampled_size = Dimension(height=height, width=width)
  natten = NattenAttnProcessor(
    kernel_size=kernel_size,
    expect_size=downsampled_size,
    has_fused_scale_factor=fuse_qkv and qkv_fusion_fuses_scale_factor,
    has_fused_qkv=fuse_qkv,
    scale_attn_entropy=scale_attn_entropy,
  )
  return natten

def make_self_subself_attn(
  attn: Attention,
  level: int,
  sample_size: Dimension,
  kernel_size=7,
  global_subsample=2,
  scale_attn_entropy=False,
  fuse_qkv=False,
  qkv_fusion_fuses_scale_factor=False,
) -> SelfSubSelfAttnProcessor:
  if fuse_qkv:
    fuse_qkv_(attn, fuse_scale_factor=qkv_fusion_fuses_scale_factor)
  downsampled_size = sample_size
  # yes I know about raising 2 to the power of negative number, but I want to model a repeated rounding-down
  downsample = torch.nn.Conv2d(1,1, kernel_size=3, stride=2, padding=1)
  size_probe = torch.ones(1,sample_size.height,sample_size.width)
  for _ in range(level):
    size_probe = downsample(size_probe)
    height, width = size_probe.shape[1:]
    downsampled_size = Dimension(height=height, width=width)
  natten = SelfSubSelfAttnProcessor(
    kernel_size=kernel_size,
    expect_size=downsampled_size,
    global_subsample=global_subsample,
    has_fused_scale_factor=fuse_qkv and qkv_fusion_fuses_scale_factor,
    has_fused_qkv=fuse_qkv,
    scale_attn_entropy=scale_attn_entropy,
  )
  return natten

def make_self_tomeself_attn(
  attn: Attention,
  level: int,
  sample_size: Dimension,
  kernel_size=7,
  global_subsample=2,
  scale_attn_entropy=False,
  fuse_qkv=False,
  qkv_fusion_fuses_scale_factor=False,
) -> SelfToMeSelfAttnProcessor:
  if fuse_qkv:
    fuse_qkv_(attn, fuse_scale_factor=qkv_fusion_fuses_scale_factor)
  downsampled_size = sample_size
  # yes I know about raising 2 to the power of negative number, but I want to model a repeated rounding-down
  downsample = torch.nn.Conv2d(1,1, kernel_size=3, stride=2, padding=1)
  size_probe = torch.ones(1,sample_size.height,sample_size.width)
  for _ in range(level):
    size_probe = downsample(size_probe)
    height, width = size_probe.shape[1:]
    downsampled_size = Dimension(height=height, width=width)
  natten = SelfToMeSelfAttnProcessor(
    kernel_size=kernel_size,
    expect_size=downsampled_size,
    global_subsample=global_subsample,
    has_fused_scale_factor=fuse_qkv and qkv_fusion_fuses_scale_factor,
    has_fused_qkv=fuse_qkv,
    scale_attn_entropy=scale_attn_entropy,
  )
  return natten

def make_sigma_swallower(attn: Attention, level: int, get_attn_processor: GetAttnProcessor) -> SigmaSwallower:
  attn_processor: AttnProcessor = get_attn_processor(attn, level)
  return SigmaSwallower(attn_processor)

class MakePickAttnDelegate(Protocol):
  @staticmethod
  def __call__(attn: Attention, level: int) -> PickAttnDelegate: ...

def make_delegation_by_sigma_cutoff(
  attn: Attention,
  level: int,
  get_high_sigma_attn_processor: GetAttnProcessor,
  get_low_sigma_attn_processor: GetAttnProcessor,
  low_sigma: float,
) -> PickAttnDelegate:
  high_sigma_attn_processor: AttnProcessor = get_high_sigma_attn_processor(attn, level)
  low_sigma_attn_processor: AttnProcessor = get_low_sigma_attn_processor(attn, level)
  def delegate_by_sigma_cutoff(sigma: float) -> AttnProcessor:
    if sigma > low_sigma:
      return high_sigma_attn_processor
    return low_sigma_attn_processor
  return delegate_by_sigma_cutoff

def make_dispatch_attn(attn: Attention, level: int, make_pick_attn_delegate: MakePickAttnDelegate) -> DispatchAttnProcessor:
  pick_attn_delegate: PickAttnDelegate = make_pick_attn_delegate(attn, level)
  return DispatchAttnProcessor(pick_attn_delegate)