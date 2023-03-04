from torch import FloatTensor, cat
from .latent_spec import LatentSpec
from typing import Iterable, List, Protocol, Iterable
from ..iteration.rle import run_length, RLEGeneric

class MakeLatents(Protocol):
  @staticmethod
  def __call__(spec: LatentSpec, start_sigma: float) -> FloatTensor: ...

class BatchLatentMaker:
  make_latents_delegate: MakeLatents
  def __init__(
    self,
    make_latents: MakeLatents,
  ) -> None:
    self.make_latents_delegate = make_latents
  
  def make_latents(
    self,
    specs: Iterable[LatentSpec],
    start_sigma: float,
  ) -> FloatTensor:
    rle_specs: Iterable[RLEGeneric[LatentSpec]] = run_length.encode(specs)
    latent_batches: List[FloatTensor] = [
      self.make_latents_delegate(rle_spec.element, start_sigma).expand(rle_spec.count, -1, -1, -1) for rle_spec in rle_specs
    ]
    return cat(latent_batches, dim=0)