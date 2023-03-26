from torch import FloatTensor, no_grad
from typing import Protocol
from .approx_decoder import Decoder
from .approx_encoder import Encoder
import torch
from functools import partial

class LatentsToRGB(Protocol):
  def __call__(latents: FloatTensor) -> FloatTensor: ...

class RGBToLatents(Protocol):
  def __call__(rgb: FloatTensor) -> FloatTensor: ...

int8_iinfo = torch.iinfo(torch.int8)
int8_range = int8_iinfo.max-int8_iinfo.min
int8_half_range = int8_range / 2

@no_grad()
def approx_latents_to_rgb(decoder: Decoder, latents: FloatTensor) -> FloatTensor:
  # batch, channels, height, width = latents.shape
  # flat_channels_last: FloatTensor = latents.flatten(-2).transpose(-2,-1)
  # decoded: FloatTensor = decoder.forward(flat_channels_last)
  # unflat: FloatTensor = decoded.unflatten(-2, (height, -1))
  decoded: FloatTensor = decoder.forward(latents)
  return decoded

#   centered = unflat - int8_half_range
#   normed = centered / int8_half_range
#   images: FloatTensor = unflat.round().clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()
#   pil_images: List[Image.Image] = [Image.fromarray(image) for image in images]
  # return unflat

def make_approx_latents_to_rgb(decoder: Decoder) -> LatentsToRGB:
  return partial(approx_latents_to_rgb, decoder)

@no_grad()
def approx_rgb_to_latents(encoder: Encoder, rgb: FloatTensor) -> FloatTensor:
  # batch, height, width, channels = rgb.shape
  # flat: FloatTensor = rgb.flatten(start_dim=-3, end_dim=-2)
  # encoded: FloatTensor = encoder.forward(flat)
  # channels_first: FloatTensor = encoded.transpose(-1, -2)
  # unflat: FloatTensor = channels_first.unflatten(-1, (height, -1))
  # return unflat
  encoded: FloatTensor = encoder.forward(rgb)
  return encoded

def make_approx_rgb_to_latents(encoder: Encoder) -> RGBToLatents:
  return partial(approx_rgb_to_latents, encoder)
  