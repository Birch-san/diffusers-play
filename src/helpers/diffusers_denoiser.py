from torch import Tensor, FloatTensor, BoolTensor
import torch.autograd.forward_ad as fwAD
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
    u_sample = fwAD.unpack_dual(sample)
    if u_sample.tangent is None:
      sample = sample.to(self.inner_model.dtype)
    else:
      sample = fwAD.make_dual(u_sample.primal.to(self.inner_model.dtype), u_sample.tangent.to(self.inner_model.dtype))
    del u_sample
    
    u_timestep = fwAD.unpack_dual(timestep)
    if u_timestep.tangent is None:
      timestep = timestep.to(self.inner_model.dtype)
    else:
      timestep = fwAD.make_dual(u_timestep.primal.to(self.inner_model.dtype), u_timestep.tangent.to(self.inner_model.dtype))
    del u_timestep
    
    u_encoder_hidden_states = fwAD.unpack_dual(encoder_hidden_states)
    if u_encoder_hidden_states.tangent is None:
      encoder_hidden_states = encoder_hidden_states.to(self.inner_model.dtype)
    else:
      encoder_hidden_states = fwAD.make_dual(u_encoder_hidden_states.primal.to(self.inner_model.dtype), u_encoder_hidden_states.tangent.to(self.inner_model.dtype))
    del u_encoder_hidden_states

    out: UNet2DConditionOutput = self.inner_model(
      sample,
      timestep,
      encoder_hidden_states=encoder_hidden_states,
      encoder_attention_mask=cross_attention_mask,
      return_dict=return_dict,
    )
    u_sample = fwAD.unpack_dual(out.sample)
    if u_sample.tangent is None:
      sample = out.sample.to(self.sampling_dtype)
    else:
      sample = fwAD.make_dual(u_sample.primal.to(self.sampling_dtype), u_sample.tangent.to(self.sampling_dtype))
    return sample

  def sigma_to_t(self, sigma: Tensor, quantize=None) -> Tensor:
    proposed = super().sigma_to_t(sigma, quantize=quantize)
    u_proposed = fwAD.unpack_dual(proposed)
    cast_primal = u_proposed.primal.to(dtype=self.inner_model.dtype)
    if u_proposed.tangent is None:
      # either we never had a tangent to begin with, or we lost it
      u_sigma = fwAD.unpack_dual(sigma)
      if u_sigma.tangent is None:
        # we didn't have a tangent to begin with; return the proposed primal
        return cast_primal
      # we had a tangent to begin with but lost it. conventionally our tangent is a negative sigma. let's try negative timestep. questionable, but nothing is valid.
      return fwAD.make_dual(cast_primal, -cast_primal)
    # use the proposed primal and tangent, but cast them
    return fwAD.make_dual(cast_primal, u_proposed.tangent.to(dtype=self.inner_model.dtype))

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
    u_sample = fwAD.unpack_dual(sample)
    if u_sample.tangent is None:
      sample = sample.to(self.inner_model.dtype)
    else:
      sample = fwAD.make_dual(u_sample.primal.to(self.inner_model.dtype), u_sample.tangent.to(self.inner_model.dtype))
    del u_sample
    
    u_timestep = fwAD.unpack_dual(timestep)
    if u_timestep.tangent is None:
      timestep = timestep.to(self.inner_model.dtype)
    else:
      timestep = fwAD.make_dual(u_timestep.primal.to(self.inner_model.dtype), u_timestep.tangent.to(self.inner_model.dtype))
    del u_timestep
    
    u_encoder_hidden_states = fwAD.unpack_dual(encoder_hidden_states)
    if u_encoder_hidden_states.tangent is None:
      encoder_hidden_states = encoder_hidden_states.to(self.inner_model.dtype)
    else:
      encoder_hidden_states = fwAD.make_dual(u_encoder_hidden_states.primal.to(self.inner_model.dtype), u_encoder_hidden_states.tangent.to(self.inner_model.dtype))
    del u_encoder_hidden_states

    out: UNet2DConditionOutput = self.inner_model(
      sample,
      timestep,
      encoder_hidden_states=encoder_hidden_states,
      encoder_attention_mask=cross_attention_mask,
      return_dict=return_dict,
    )
    u_sample = fwAD.unpack_dual(out.sample)
    if u_sample.tangent is None:
      sample = out.sample.to(self.sampling_dtype)
    else:
      sample = fwAD.make_dual(u_sample.primal.to(self.sampling_dtype), u_sample.tangent.to(self.sampling_dtype))
    return sample

  def sigma_to_t(self, sigma: Tensor, quantize=None) -> Tensor:
    proposed = super().sigma_to_t(sigma, quantize=quantize)
    u_proposed = fwAD.unpack_dual(proposed)
    cast_primal = u_proposed.primal.to(dtype=self.inner_model.dtype)
    if u_proposed.tangent is None:
      # either we never had a tangent to begin with, or we lost it
      u_sigma = fwAD.unpack_dual(sigma)
      if u_sigma.tangent is None:
        # we didn't have a tangent to begin with; return the proposed primal
        return cast_primal
      # we had a tangent to begin with but lost it. conventionally our tangent is a negative sigma. let's try negative timestep. questionable, but nothing is valid.
      return fwAD.make_dual(cast_primal, -cast_primal)
    # use the proposed primal and tangent, but cast them
    return fwAD.make_dual(cast_primal, u_proposed.tangent.to(dtype=self.inner_model.dtype))
    
    
