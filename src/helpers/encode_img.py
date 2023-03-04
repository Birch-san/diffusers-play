from torch import FloatTensor
from typing import Optional, Protocol
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
from diffusers.models import AutoencoderKL
import torch
from functools import partial

class EncodeImg(Protocol):
  def __call__(img: FloatTensor, generator: Optional[torch.Generator]=None): ...

def _encode_img(
  vae: AutoencoderKL,
  img: FloatTensor,
  generator: torch.Generator=None,
) -> FloatTensor:
  init_image: FloatTensor = img.to(dtype=vae.dtype, device=vae.device)
  out: AutoencoderKLOutput = vae.encode(init_image) # move to latent space
  init_latent: FloatTensor = out.latent_dist.sample(generator=generator)
  init_latent = init_latent * (vae.config.scaling_factor if 'scaling_factor' in vae.config else 0.18215)
  return init_latent

def make_encode_img(vae: AutoencoderKL) -> EncodeImg:
  return partial(_encode_img, vae)