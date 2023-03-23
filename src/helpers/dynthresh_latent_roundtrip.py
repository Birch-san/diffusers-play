from torch import FloatTensor, no_grad
from typing import Protocol
from .approx_decoder import Decoder
import torch
from functools import partial

class LatentsToRGB(Protocol):
  def __call__(latents: FloatTensor) -> FloatTensor: ...

class RGBToLatents(Protocol):
  def __call__(rgb: FloatTensor) -> FloatTensor: ...

@no_grad()
def approx_latents_to_rgb(decoder: Decoder, latents: FloatTensor) -> FloatTensor:
  _, _, height, _ = latents.shape
  flat_channels_last: FloatTensor = latents.flatten(-2).transpose(-2,-1)
  decoded: FloatTensor = decoder.forward(flat_channels_last)
  unflat: FloatTensor = decoded.unflatten(-2, (height, -1))
#   images: FloatTensor = unflat.round().clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()
#   pil_images: List[Image.Image] = [Image.fromarray(image) for image in images]
  return unflat

def make_approx_latents_to_rgb(decoder: Decoder) -> LatentsToRGB:
  return partial(approx_latents_to_rgb, decoder)

@no_grad()
def approx_rgb_to_latents(decoder: Decoder, rgb: FloatTensor) -> FloatTensor:
  _, _, height, _ = rgb.shape
  flat_channels_last: FloatTensor = rgb.flatten(-2).transpose(-2,-1)
  # TODO: run inverse transform
  decoded: FloatTensor = decoder.forward(flat_channels_last)
  unflat: FloatTensor = decoded.unflatten(-2, (height, -1))
  return unflat

def make_approx_rgb_to_latents(decoder: Decoder) -> RGBToLatents:
  return partial(approx_rgb_to_latents, decoder)
  