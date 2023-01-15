from torch import FloatTensor
from .latent_spec import LatentSpec
from typing import Iterable, Optional, Protocol, Iterable

class MakeLatentsStrategy(Protocol):
  @staticmethod
  def __call__(spec: LatentSpec) -> Optional[FloatTensor]: ...

class LatentMaker:
  strategies: Iterable[MakeLatentsStrategy]
  def __init__(
    self,
    strategies: Iterable[MakeLatentsStrategy],
  ) -> None:
    self.strategies = strategies

  def make_latents(
    self,
    spec: LatentSpec,
  ) -> FloatTensor:
    for strategy in self.strategies:
      latents: Optional[FloatTensor] = strategy(spec)
      if latents is not None:
        return latents
    else:
      raise ValueError(f'No LatentMaker strategy implemented for {spec}')

