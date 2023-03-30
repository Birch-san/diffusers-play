from functools import partial
from torch import FloatTensor, no_grad
import torch
from PIL import Image
from typing import List
from .decoder import Decoder
from .int_info import int8_half_range
from ..latents_to_pils import LatentsToPils

@no_grad()
def approx_latents_to_pils(decoder: Decoder, latents: FloatTensor) -> FloatTensor:
  channels_last: FloatTensor = latents.permute(0, 2, 3, 1)
  decoded: FloatTensor = decoder.forward(channels_last)
  decoded = decoded * int8_half_range
  decoded = decoded + int8_half_range
  images: FloatTensor = decoded.round().clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()
  pil_images: List[Image.Image] = [Image.fromarray(image) for image in images]
  return pil_images

def make_approx_latents_to_pils(decoder: Decoder) -> LatentsToPils:
  return partial(approx_latents_to_pils, decoder)