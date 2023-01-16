from helpers.inference_spec.latent_spec import LatentSpec
import torch
from torch import Generator as TorchGenerator, FloatTensor, randn
from ..device import DeviceType
from .latents_shape import LatentsShape
from .latent_spec import SeedSpec
from typing import Optional

class SeedLatentMaker:
  shape: LatentsShape
  dtype: torch.dtype
  device: DeviceType
  generator: TorchGenerator
  def __init__(
    self,
    shape: LatentsShape,
    dtype: torch.dtype = torch.float32,
    device: DeviceType = torch.device('cpu')
  ) -> None:
    self.shape = shape
    self.dtype = dtype
    self.device = device
    self.generator = TorchGenerator(device='cpu')
  
  def _make_latents(self, seed: int) -> FloatTensor:
    self.generator.manual_seed(seed)
    latents: FloatTensor = randn((1, *self.shape), generator=self.generator, device='cpu', dtype=self.dtype).to(self.device)
    return latents
  
  def make_latents(self, spec: LatentSpec) -> Optional[FloatTensor]:
    match spec:
      case SeedSpec(seed):
        return self._make_latents(seed)
      case _:
        return None