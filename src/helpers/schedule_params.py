import torch
from torch import Tensor, cumprod, linspace, argmin
from typing import Optional
from .device import DeviceType

def get_betas(
  num_train_timesteps: int = 1000,
  beta_start: float = 0.00085,
  beta_end: float = 0.012,
  device: Optional[DeviceType] = None,
  dtype: torch.dtype = torch.float32,
) -> Tensor:
  return linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=dtype, device=device) ** 2

def get_alphas(betas: Tensor) -> Tensor:
  return 1.0 - betas

def get_alphas_cumprod(alphas: Tensor) -> Tensor:
  return cumprod(alphas, dim=0)

def get_sigmas(alphas_cumprod: Tensor) -> Tensor:
  return ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

def get_log_sigmas(sigmas: Tensor) -> Tensor:
  return sigmas.log()

def quantize_to(proposed: Tensor, quanta: Tensor) -> Tensor:
  return quanta[argmin((proposed.unsqueeze(1).expand(-1, quanta.size(0)) - quanta).abs(), dim=1)]

def log_sigmas_to_t(proposed_log_sigmas: Tensor, log_sigma_quanta: Tensor) -> Tensor:
  """Quantized sigmas only, please"""
  return (proposed_log_sigmas-log_sigma_quanta.unsqueeze(1)).abs().argmin(dim=0)