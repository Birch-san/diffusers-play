from helpers.inference_spec.batch_spec_factory import LatentsGenerator
from typing import List
from torch import FloatTensor
from helpers.get_seed import get_seed
import torch

def make_latents(seed: int, repeat: int = 1) -> FloatTensor:
  return torch.full((repeat, 1), seed)

generator = LatentsGenerator(
  batch_size=3,
  specs=[
    1,
    1,
    1,
    2,
    3,
    4,
    *[get_seed() for _ in range(2)]
  ],
  make_latents=make_latents,
).generate()
print(list(generator))