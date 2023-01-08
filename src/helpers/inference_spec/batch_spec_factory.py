from abc import ABC, abstractmethod
from typing import Protocol, Generator, List, NamedTuple
from dataclasses import dataclass
from torch import FloatTensor, Generator as TorchGenerator
from ..get_seed import get_seed

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
  def __init__(
    self,
    batch_size: int,
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