from .diffusers_denoiser import DiffusersSDDenoiser

from torch import Tensor, cat

from typing import List, Dict, Optional, Protocol
from unittest import TestCase

class GetModelWeight(Protocol):
  def __call__(sigma: float) -> float: ...

def static_model_weight(weight: float) -> GetModelWeight:
  def get_model_weight(sigma: float) -> float:
    return weight
  return get_model_weight

class MultiUnetCFGDenoiser():
  denoisers: Dict[str, DiffusersSDDenoiser]
  def __init__(self, denoisers: Dict[str, DiffusersSDDenoiser]):
    self.denoisers = denoisers
  
  def __call__(
    self,
    x: Tensor,
    sigma: Tensor,
    unconds: Dict[str, Tensor],
    conds: Dict[str, Tensor],
    model_weights: Dict[str, GetModelWeight],
    cond_scale: float
  ) -> Tensor:
    assert cond_scale > 1.0, "non-CFG fastpath not implemented. doable but everybody loves CFG"
    assert len(conds) > 0
    # strictly speaking it's possible to re-use the same uncond between *some* models, but let's tackle general case
    TestCase().assertCountEqual(unconds.keys(), self.denoisers.keys(), "need to supply an uncond tensor per model ID")
    TestCase().assertCountEqual(conds.keys(), self.denoisers.keys(), "need to supply a cond tensor per model ID")
    TestCase().assertCountEqual(model_weights.keys(), self.denoisers.keys(), "need to supply a model weight per model ID")
    batch_sizes: List[int] = [cond.size(dim=0) for cond in conds.values()]
    batch_size: int = batch_sizes[0]
    assert all(batch_size_ == batch_size for batch_size_ in batch_sizes)

    x_in: Tensor = x.expand(batch_size + 1, -1, -1, -1)
    sigma_fl: float = sigma.item()
    
    # yes I know about reduce()
    # but multi-line lambda expressions would make that easier
    accumulated: Optional[Tensor] = None
    for id, denoiser in self.denoisers.items():
      get_model_weight: GetModelWeight = model_weights[id]
      model_weight: float = get_model_weight(sigma_fl)
      if model_weight == 0.:
        continue
      uncond: Tensor = unconds[id]
      cond: Tensor = conds[id]
      cond_in: Tensor = cat([uncond, cond])
      denoised: Tensor = denoiser(input=x_in, sigma=sigma, encoder_hidden_states=cond_in)
      uncond, cond = denoised.chunk(cond_in.size(dim=0))
      contribution: Tensor = (uncond + (cond - uncond) * cond_scale) * model_weight
      accumulated = contribution if accumulated is None else accumulated + contribution
      # it's possible to accumulate in a more functional / parallel way
      # (map each denoiser to an output, stack the outputs, sum them)
      # but it would have a higher peak memory usage (have to retain all outputs until you're ready to sum)
      # in a similar vein: perhaps we could run each denoiser simultaneously, but my GPU seems saturated by just one anyway
    
    assert accumulated is not None
    return accumulated