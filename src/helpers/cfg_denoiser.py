from .diffusers_denoiser import DiffusersSDDenoiser
from torch import Tensor, BoolTensor, cat
from typing import Optional, Protocol, NamedTuple
from abc import ABC, abstractmethod

class Denoiser(Protocol):
  def __call__(self, x: Tensor, sigma: Tensor) -> Tensor: ...

class CFGConds(NamedTuple):
  uncond: Tensor
  cond: Tensor

class AbstractCFGDenoiser(ABC, Denoiser):
  denoiser: DiffusersSDDenoiser
  cond_scale: float
  def __init__(
      self,
      denoiser: DiffusersSDDenoiser,
      cond_scale = 1.0,
    ):
    self.denoiser = denoiser
    self.cond_scale = cond_scale
  
  @abstractmethod
  def get_cfg_conds(self, x: Tensor, sigma: Tensor) -> CFGConds: ...

  def __call__(self, x: Tensor, sigma: Tensor) -> Tensor:
    uncond, cond = self.get_cfg_conds(x, sigma)
    return uncond + (cond - uncond) * self.cond_scale

class SerialCFGDenoiser(AbstractCFGDenoiser):
  uncond: Tensor
  cond: Tensor
  uc_mask: BoolTensor
  c_mask: BoolTensor
  def __init__(
    self,
    denoiser: DiffusersSDDenoiser,
    uncond: Tensor,
    cond: Tensor,
    cond_scale: float,
    attention_mask: Optional[BoolTensor] = None,
  ):
    self.uncond = uncond
    self.cond = cond
    uc_mask, c_mask = attention_mask.chunk(attention_mask.size(0))
    self.uc_mask = uc_mask
    self.c_mask = c_mask
    super().__init__(
      denoiser=denoiser,
      cond_scale=cond_scale,
    )
  def get_cfg_conds(self, x: Tensor, sigma: Tensor) -> CFGConds:
    uncond = self.denoiser(input=x, sigma=sigma, encoder_hidden_states=self.uncond, attention_mask=self.uc_mask)
    cond = self.denoiser(input=x, sigma=sigma, encoder_hidden_states=self.cond, attention_mask=self.c_mask)
    return CFGConds(uncond, cond)

class ParallelCFGDenoiser(AbstractCFGDenoiser):
  cond_in: Tensor
  attention_mask: Optional[BoolTensor]
  def __init__(
    self,
    denoiser: DiffusersSDDenoiser,
    uncond: Tensor,
    cond: Tensor,
    cond_scale: float,
    attention_mask: Optional[BoolTensor] = None,
  ):
    self.cond_in = cat([uncond, cond])
    self.attention_mask = attention_mask
    super().__init__(
      denoiser=denoiser,
      cond_scale=cond_scale,
    )
  def get_cfg_conds(self, x: Tensor, sigma: Tensor) -> CFGConds:
    x_in = x.expand(self.cond_in.size(dim=0), -1, -1, -1)
    del x
    uncond, cond = self.denoiser(input=x_in, sigma=sigma, encoder_hidden_states=self.cond_in, attention_mask=self.attention_mask).chunk(self.cond_in.size(dim=0))
    return CFGConds(uncond, cond)

class NoCFGDenoiser(Denoiser):
  denoiser: DiffusersSDDenoiser
  cond: Tensor
  attention_mask: Optional[BoolTensor]
  def __init__(
    self,
    denoiser: DiffusersSDDenoiser,
    cond: Tensor,
    attention_mask: Optional[BoolTensor] = None,
  ):
    self.denoiser = denoiser
    self.cond = cond
    self.attention_mask = attention_mask
  def __call__(self, x: Tensor, sigma: Tensor) -> Tensor:
    return self.denoiser(
      input=x,
      sigma=sigma,
      encoder_hidden_states=self.cond,
      attention_mask=self.attention_mask,
    )

class DenoiserFactory():
  denoiser: DiffusersSDDenoiser
  # this is a workaround which caters for some wacky experiments
  # - CLIP-guided diffusion goes bang on backpropagation when batch size exceeds 1
  # - ANE-optimized UNets produce incorrect results when running with batch>1 on MPS
  # - GPU-optimized Unets targeting crash when running with batch>1 on CoreML targeting ANE
  one_at_a_time: bool
  def __init__(self, denoiser: DiffusersSDDenoiser, one_at_a_time=False):
    self.denoiser = denoiser
    self.one_at_a_time = one_at_a_time
  
  def __call__(
    self,
    cond: Tensor,
    uncond: Optional[Tensor] = None,
    cond_scale: float = 1.0,
    attention_mask: Optional[BoolTensor] = None,
  ) -> Denoiser:
    if uncond is None or cond_scale is None:
      return NoCFGDenoiser(
        denoiser=self.denoiser,
        cond=cond,
        attention_mask=attention_mask,
      )
    if self.one_at_a_time:
      return SerialCFGDenoiser(
        denoiser=self.denoiser,
        uncond=uncond,
        cond=cond,
        cond_scale=cond_scale,
        attention_mask=attention_mask,
      )
    return ParallelCFGDenoiser(
      denoiser=self.denoiser,
      uncond=uncond,
      cond=cond,
      cond_scale=cond_scale,
      attention_mask=attention_mask,
    )