from torch import Tensor, FloatTensor, BoolTensor
from diffusers.models import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from k_diffusion.external import DiscreteEpsDDPMDenoiser, DiscreteVDDPMDenoiser
from typing import Union, Optional
import torch

class DiffusersSDDenoiser(DiscreteEpsDDPMDenoiser):
  inner_model: UNet2DConditionModel
  sampling_dtype: torch.dtype
  def __init__(self, unet: UNet2DConditionModel, alphas_cumprod: Tensor, dtype: torch.dtype = None):
    self.sampling_dtype = unet.dtype if dtype is None else dtype
    super().__init__(unet, alphas_cumprod, quantize=True)

  def get_eps(
    self,
    sample: FloatTensor,
    timestep: Union[Tensor, float, int],
    encoder_hidden_states: Tensor,
    return_dict: bool = True,
    cross_attention_mask: Optional[BoolTensor] = None,
  ) -> Tensor:
    out: UNet2DConditionOutput = self.inner_model(
      sample.to(self.inner_model.dtype),
      timestep.to(self.inner_model.dtype),
      encoder_hidden_states=encoder_hidden_states.to(self.inner_model.dtype),
      encoder_attention_mask=cross_attention_mask,
      return_dict=return_dict,
    )
    return out.sample.to(self.sampling_dtype)

  def sigma_to_t(self, sigma: Tensor, quantize=None) -> Tensor:
    return super().sigma_to_t(sigma, quantize=quantize).to(dtype=self.inner_model.dtype)

class DiffusersSD2Denoiser(DiscreteVDDPMDenoiser):
  inner_model: UNet2DConditionModel
  sampling_dtype: torch.dtype
  def __init__(self, unet: UNet2DConditionModel, alphas_cumprod: Tensor, dtype: torch.dtype = None):
    self.sampling_dtype = unet.dtype if dtype is None else dtype
    super().__init__(unet, alphas_cumprod, quantize=True)

  def get_v(
    self,
    sample: FloatTensor,
    timestep: Union[Tensor, float, int],
    encoder_hidden_states: Tensor,
    return_dict: bool = True,
    cross_attention_mask: Optional[BoolTensor] = None,
    ) -> Tensor:
    out: UNet2DConditionOutput = self.inner_model(
      sample.to(self.inner_model.dtype),
      timestep.to(self.inner_model.dtype),
      encoder_hidden_states=encoder_hidden_states.to(self.inner_model.dtype),
      encoder_attention_mask=cross_attention_mask,
      return_dict=return_dict,
    )
    return out.sample.to(self.sampling_dtype)

  def sigma_to_t(self, sigma: Tensor, quantize=None) -> Tensor:
    return super().sigma_to_t(sigma, quantize=quantize).to(dtype=self.inner_model.dtype)
