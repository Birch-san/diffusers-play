import torch
from torch import FloatTensor, IntTensor, Tensor
from helpers.approx_vae.int_info import int8_half_range, int8_iinfo

def normalize_latents(latents: FloatTensor) -> FloatTensor:
  flat: FloatTensor = latents.flatten(-2).unsqueeze(-1)
  mean: FloatTensor = flat.mean(dim=-2, keepdim=True)
  del flat
  centered: FloatTensor = latents-mean
  norm: FloatTensor = centered / centered.flatten(-2).abs().max(dim=-1, keepdim=True).values.unsqueeze(-1)
  return norm

def norm_latents_to_rgb(normalized: FloatTensor) -> IntTensor:
  rgb: FloatTensor = (normalized + 1)*int8_half_range
  clamped: IntTensor = rgb.clamp(min=int8_iinfo.min, max=int8_iinfo.max).to(dtype=torch.uint8)
  return clamped

def collage_2by2(sample: Tensor, keepdim=False) -> Tensor:
  # collaged: IntTensor = cat([
  #   cat([sample[0], sample[1]], dim=-1),
  #   cat([sample[2], sample[3]], dim=-1)
  # ], dim=0)
  collaged: IntTensor = sample.unflatten(-3, (2,-1)).transpose(-3,-2).flatten(-2).flatten(start_dim=-3, end_dim=-2)
  if keepdim:
    return collaged.unsqueeze(0)
  return collaged