from abc import ABC
from dataclasses import dataclass
from typing import Protocol
from torch import FloatTensor

@dataclass
class LatentSpec(ABC): pass

@dataclass
class SeedSpec(LatentSpec):
  seed: int

@dataclass
class Img2ImgSpec(SeedSpec, ABC):
  start_sigma: float

@dataclass
class FeedbackSpec(Img2ImgSpec): pass

class GetLatents(Protocol):
  @staticmethod
  def __call__() -> FloatTensor: ...

@dataclass
class ImgEncodeSpec(Img2ImgSpec):
  get_latents: GetLatents

LatentSpec.register(SeedSpec)
LatentSpec.register(Img2ImgSpec)
LatentSpec.register(FeedbackSpec)
LatentSpec.register(ImgEncodeSpec)

Img2ImgSpec.register(FeedbackSpec)
Img2ImgSpec.register(ImgEncodeSpec)