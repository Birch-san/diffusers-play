from enum import Enum, auto
from typing import Protocol
from torch import FloatTensor

class InterpProto(Protocol):
  def __call__(start: FloatTensor, end: FloatTensor, t: float|FloatTensor) -> FloatTensor: ...

class InterpStrategy(Enum):
  Slerp = auto()
  Lerp = auto()