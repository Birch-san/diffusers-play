from helpers.inference_spec.latent_spec import LatentSpec
from torch import Generator as FloatTensor
from .latent_spec import ImgEncodeSpec, GetLatents
from .latent_maker_seed_strategy import SeedLatentMaker
from typing import Optional
from dataclasses import dataclass

@dataclass
class ImgEncodeLatentMaker:
  seed_latent_maker: SeedLatentMaker
  
  def _make_latents(self, nominal: FloatTensor, get_latents: GetLatents) -> FloatTensor:
    latent_bias: FloatTensor = get_latents()
    return nominal + latent_bias
  
  def make_latents(self, spec: LatentSpec, start_sigma: float) -> Optional[FloatTensor]:
    match spec:
      case ImgEncodeSpec(_, _, get_latents):
        nominal: Optional[FloatTensor] = self.seed_latent_maker.make_latents(spec, start_sigma)
        assert nominal is not None
        return self._make_latents(nominal, get_latents)
      case _:
        return None