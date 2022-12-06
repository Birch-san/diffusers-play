from .diffusers_denoiser import DiffusersSDDenoiser
from torch import Tensor, BoolTensor, cat
from typing import Optional

class CFGDenoiser():
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
    x: Tensor,
    sigma: Tensor,
    uncond: Tensor,
    cond: Tensor, 
    cond_scale: float,
    cond_mask: Optional[BoolTensor] = None,
    uncond_mask: Optional[BoolTensor] = None,
  ) -> Tensor:
    if uncond is None or cond_scale == 1.0:
      # why are you using CFGDenoiser
      return self.denoiser(input=x, sigma=sigma, encoder_hidden_states=cond, cross_attn_mask=cond_mask)
    if self.one_at_a_time:
      # if batching doesn't work: don't batch
      uncond = self.denoiser(input=x, sigma=sigma, encoder_hidden_states=uncond, cross_attn_mask=uncond_mask)
      cond = self.denoiser(input=x, sigma=sigma, encoder_hidden_states=cond, cross_attn_mask=cond_mask)
      del x, sigma
    else:
      cond_mask_in: Optional[Tensor] = cat([uncond_mask, cond_mask]) if cond_mask is not None and uncond_mask is not None else None
      cond_in = cat([uncond, cond])
      del uncond, cond
      x_in = x.expand(cond_in.size(dim=0), -1, -1, -1)
      del x
      uncond, cond = self.denoiser(input=x_in, sigma=sigma, encoder_hidden_states=cond_in, cross_attn_mask=cond_mask_in).chunk(cond_in.size(dim=0))
      del x_in, cond_in
    return uncond + (cond - uncond) * cond_scale