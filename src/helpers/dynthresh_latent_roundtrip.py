from torch import FloatTensor, no_grad
from typing import Protocol
from .approx_decoder import Decoder
from .approx_encoder import Encoder
from functools import partial

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
  