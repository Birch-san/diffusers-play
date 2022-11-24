from functools import partial
from torch import Tensor, no_grad
from typing import List, Callable
from typing_extensions import TypeAlias
from PIL import Image
from diffusers.models import AutoencoderKL

LatentsToPils: TypeAlias = Callable[[Tensor], List[Image.Image]]

@no_grad()
def latents_to_pils(vae: AutoencoderKL, latents: Tensor) -> List[Image.Image]:
  latents = 1 / 0.18215 * latents

  images: Tensor = vae.decode(latents).sample

  images = (images / 2 + 0.5).clamp(0, 1)

  # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
  images = images.cpu().permute(0, 2, 3, 1).float().numpy()
  images = (images * 255).round().astype("uint8")

  pil_images: List[Image.Image] = [Image.fromarray(image) for image in images]
  return pil_images

def make_latents_to_pils(vae: AutoencoderKL) -> LatentsToPils:
  return partial(latents_to_pils, vae)