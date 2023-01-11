from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Protocol, Generator, List, Iterable, NamedTuple, Optional, Generic, TypeVar, Any
from dataclasses import dataclass
import torch
from torch import FloatTensor, Generator as TorchGenerator, randn
from ..get_seed import get_seed
from ..device import DeviceType
from itertools import islice
from ..iteration.chunk import chunk
from ..iteration.rle import run_length, RLEGeneric

T = TypeVar('T')

class AbstractSeedGenerator(Protocol):
  def generate() -> Generator[int]: ...

class RandomSeedSequence(AbstractSeedGenerator):
  def generate() -> Generator[int]:
    while True:
      yield get_seed()

class FixedSeedSequence(AbstractSeedGenerator):
  seeds: Iterable[int]
  def __init__(
    self,
    seeds: Iterable[int],
  ) -> None:
    super().__init__()
    self.seeds = seeds

  def generate(self) -> Generator[int]:
    return self.seeds.__iter__()
    # return (seed for seed in self.seeds)

class AbstractLatentsGenerator:
  generator: TorchGenerator
  device: DeviceType
  def __init__(
    self,
    batch_size: int,
    device: DeviceType = torch.device('cpu')
  ):
    self.batch_size = batch_size
    self.generator = TorchGenerator(device='cpu')
    self.device = device

  @abstractmethod
  def generate(self) -> Generator[FloatTensor]: ...

class SeedsTaken(NamedTuple):
  seed: int
  taken: int
  next_spec: Optional[AbstractSeedSpec]

@dataclass
class AbstractSeedSpec(ABC):
  # seed: int
  @abstractmethod
  def take(self, want: int) -> SeedsTaken: ...

class SequenceSeedSpec(AbstractSeedSpec):
  seeds: Iterable[int]
  def __init__(
    self,
    seeds: Iterable[int],
  ) -> None:
    super().__init__()
    self.seeds = seeds

  def take(self, want: int) -> SeedsTaken:

    assert want > 0 and self.remaining > 0
    next_spec: Optional[AbstractSeedSpec] = FiniteSeedSpec(
      seed=self.seed,
      taken=want,
      remaining=self.remaining-want,
    ) if want < self.remaining else None
    return SeedsTaken(seeds=self.seed, taken=want, next_spec=self)

class FiniteSeedSpec(AbstractSeedSpec):
  remaining: int
  def __init__(
    self,
    seed: int,
    remaining=1,
  ) -> None:
    super().__init__(seed)
    self.remaining = remaining

  def take(self, want: int) -> SeedsTaken:
    assert want > 0 and self.remaining > 0
    next_spec: Optional[AbstractSeedSpec] = FiniteSeedSpec(
      seed=self.seed,
      taken=want,
      remaining=self.remaining-want,
    ) if want < self.remaining else None
    return SeedsTaken(seeds=self.seed, taken=want, next_spec=next_spec)

class InfiniteSeedSpec(AbstractSeedSpec):
  def take(self, want: int) -> SeedsTaken:
    return SeedsTaken(seeds=self.seed, taken=want, next_spec=self)

AbstractSeedSpec.register(FiniteSeedSpec)
AbstractSeedSpec.register(InfiniteSeedSpec)

class MakeLatents(Protocol, Generic[T]):
  @staticmethod
  def __call__(spec: T, repeat: int = 1) -> FloatTensor: ...

@dataclass
class MakeLatentsFromSeedSpec:
  seed: int

class MakeLatentsFromSeed(MakeLatents[MakeLatentsFromSeedSpec]):
  @staticmethod
  def make(spec: MakeLatentsFromSeedSpec) -> FloatTensor:
    pass

class LatentsShape(NamedTuple):
  channels: int
  height: int
  width: int

def latents_from_seed_factory(
  shape: LatentsShape,
  dtype: torch.dtype = torch.float32,
  device: DeviceType = torch.device('cpu')
) -> MakeLatents[int]:
  generator = TorchGenerator(device='cpu')
  def make_latents(seed: int, repeat: int = 1) -> FloatTensor:
    generator.manual_seed(seed)
    latents: FloatTensor = randn((1, *shape), generator=generator, device='cpu', dtype=dtype).to(device)
    return latents.expand(repeat, -1, -1, -1)
  return make_latents

class LatentBatcher(Generic[T]):
  batch_size: int
  specs: Iterable[T]
  make_latents: MakeLatents[T]
  def __init__(
    self,
    batch_size: int,
    specs: Iterable[T],
    make_latents: MakeLatents[T],
  ) -> None:
    self.batch_size = batch_size
    self.specs = specs
    self.make_latents = make_latents

  def generate(self) -> Generator[FloatTensor, None, None]:
    for chnk in chunk(self.specs, self.batch_size):
      rle_specs: List[RLEGeneric[T]] = list(run_length.encode(chnk))
      latents: List[FloatTensor] = [
        self.make_latents(rle_spec.element, rle_spec.count) for rle_spec in rle_specs
      ]
      yield torch.cat(latents, dim=0)

SampleSpec = TypeVar('SampleSpec')

class MakeLatentBatches(Protocol, Generic[SampleSpec]):
  @staticmethod
  def __call__(batch_size: int, specs: Iterable[SampleSpec]) -> Iterable[FloatTensor]: ...

class BatchSpecX(NamedTuple):
  latents: FloatTensor

class SampleSpecBatcher(Generic[SampleSpec]):
  make_latent_batches: MakeLatentBatches[SampleSpec]
  def __init__(
    self,
    batch_size: int,
    make_latent_batches: MakeLatentBatches[SampleSpec],
  ) -> None:
    self.batch_size = batch_size
    self.make_latent_batches = make_latent_batches
  
  def generate(
    self,
    specs: Iterable[SampleSpec],
  ) -> Generator[BatchSpecX, None, None]:
    latent_batches: Iterable[FloatTensor] = self.make_latent_batches(self.batch_size, specs)
    for latents in latent_batches:
      yield BatchSpecX(latents=latents)


class LatentSampleShape(NamedTuple):
  channels: int
  # dimensions of *latents*, not pixels
  height: int
  width: int

# class AbstractBatchSpecFactory(ABC):
#   def 

@dataclass
class BatchSpec:
  latents: FloatTensor
  seeds: List[int]

class AbstractBatchSpecFactory(ABC):
  batch_size: int
  generator: TorchGenerator
  device: DeviceType
  def __init__(
    self,
    batch_size: int,
    device: DeviceType = torch.device('cpu')
  ):
    self.batch_size = batch_size
    self.generator = TorchGenerator(device='cpu')

  @abstractmethod
  def generate() -> Generator[BatchSpec]: ...

class BasicBatchSpecFactory(AbstractBatchSpecFactory):
  def generate() -> Generator[BatchSpec]:
    pass

class BatchSpecFactory:
  def __init__(self) -> None:
    pass