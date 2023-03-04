from helpers.inference_spec.latent_spec import LatentSpec
import torch
from torch import Generator as TorchGenerator, FloatTensor, randn
from ..device import DeviceType
from .latents_shape import LatentsShape
from .latent_spec import SeedSpec
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class SeedLatentMaker:
  shape: LatentsShape
  dtype: torch.dtype = torch.float32,
  device: DeviceType = torch.device('cpu')
  generator: TorchGenerator = field(init=False)

  def __post_init__(self):
    self.generator = TorchGenerator(device='cpu')
  
  def _make_latents(self, seed: int, start_sigma: float) -> FloatTensor:
    self.generator.manual_seed(seed)
    latents: FloatTensor = randn((1, *self.shape), generator=self.generator, device='cpu', dtype=self.dtype).to(self.device)
    latents *= start_sigma
    return latents
  
  def make_latents(self, spec: LatentSpec, start_sigma: float) -> Optional[FloatTensor]:
    match spec:
      case SeedSpec(seed):
        return self._make_latents(seed, start_sigma)
      case _:
        return None