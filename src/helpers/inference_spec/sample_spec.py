from typing import Iterable, TypeAlias
from dataclasses import dataclass
from abc import ABC

@dataclass
class WeightedPrompt():
  text: str
  weight: float

MultiPrompt: TypeAlias = Iterable[WeightedPrompt]

@dataclass
class SampleSpec(ABC):
  seed: int
  multiprompt: MultiPrompt

@dataclass
class BetweenSampleSpec(SampleSpec):
  target_multiprompt: MultiPrompt
  interp_quotient: float
SampleSpec.register(BetweenSampleSpec)