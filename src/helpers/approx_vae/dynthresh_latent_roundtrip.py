from torch import FloatTensor, no_grad
# from torchvision.transforms.functional import resize, InterpolationMode
from typing import Optional, Protocol
from .decoder import Decoder
from .encoder import Encoder
from functools import partial
from diffusers.models.autoencoder_kl import AutoencoderKLOutput
from diffusers.models import AutoencoderKL
import torch

class LatentsToRGB(Protocol):
  def __call__(latents: FloatTensor) -> FloatTensor: ...

class RGBToLatents(Protocol):
  def __call__(rgb: FloatTensor) -> FloatTensor: ...

@no_grad()
def approx_latents_to_rgb(decoder: Decoder, latents: FloatTensor) -> FloatTensor:
  """
  latents: [b,c,h,w]
  decoder: [b,h,w,c] -> [b,h,w,c]
  returns: [b,c,h,w]
  outputs RGB nominally in the range ±1
  but at high CFG will exceed this
  """
  channels_last: FloatTensor = latents.permute(0, 2, 3, 1)
  decoded: FloatTensor = decoder.forward(channels_last)
  channels_first: FloatTensor = decoded.permute(0, 3, 1, 2)
  return channels_first

def make_approx_latents_to_rgb(decoder: Decoder) -> LatentsToRGB:
  return partial(approx_latents_to_rgb, decoder)

@no_grad()
def approx_rgb_to_latents(encoder: Encoder, rgb: FloatTensor) -> FloatTensor:
  """
      rgb: [b,c,h,w]
  encoder: [b,h,w,c] -> [b,h,w,c]
  returns: [b,c,h,w]
  expects to output RGB nominally in the range ±1,
  except if your latents were created at high CFG
  """
  channels_last: FloatTensor = rgb.permute(0, 2, 3, 1)
  encoded: FloatTensor = encoder.forward(channels_last)
  channels_first: FloatTensor = encoded.permute(0, 3, 1, 2)
  return channels_first

def make_approx_rgb_to_latents(encoder: Encoder) -> RGBToLatents:
  return partial(approx_rgb_to_latents, encoder)

@no_grad()
def real_latents_to_rgb(vae: AutoencoderKL, latents: FloatTensor) -> FloatTensor:
  """
  latents: [b,c,h,w]
  returns: [b,c,h,w], 8x bigger
  """
  dtype, device = latents.dtype, latents.device
  latents: FloatTensor = latents / (vae.config.scaling_factor if 'scaling_factor' in vae.config else 0.18215)
  decoded: FloatTensor = vae.decode(latents.to(dtype=vae.dtype, device=vae.device)).sample.to(dtype=dtype, device=device)
  # originally I thought it'd be bad to return 8x bigger RGB, but encoder is expecting the same thing so it works out.
  # here's how to downsample though (in case you want to pair with an approx decoder that doesn't scale the image for you)
  # _, _, height, width = decoded.shape
  # vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
  # scaled_height = height//vae_scale_factor
  # scaled_width = width//vae_scale_factor
  # resized = resize(decoded, [scaled_height, scaled_width], InterpolationMode.BICUBIC, antialias=True)
  return decoded

def make_real_latents_to_rgb(decoder: Decoder) -> LatentsToRGB:
  return partial(real_latents_to_rgb, decoder)

@no_grad()
def real_rgb_to_latents(vae: AutoencoderKL, generator: Optional[torch.Generator], rgb: FloatTensor) -> FloatTensor:
  """
      rgb: [b,c,h,w]
  returns: [b,c,h,w], 8x smaller
  """
  dtype, device = rgb.dtype, rgb.device
  out: AutoencoderKLOutput = vae.encode(rgb.to(dtype=vae.dtype, device=vae.device))
  encoded: FloatTensor = out.latent_dist.sample(generator=generator).to(dtype=dtype, device=device)
  encoded: FloatTensor = encoded * (vae.config.scaling_factor if 'scaling_factor' in vae.config else 0.18215)
  return encoded

def make_real_rgb_to_latents(encoder: Encoder, generator: Optional[torch.Generator]=None) -> RGBToLatents:
  return partial(real_rgb_to_latents, encoder, generator)
  