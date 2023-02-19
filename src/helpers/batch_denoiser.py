from abc import ABC
from dataclasses import dataclass, field
from torch import LongTensor, BoolTensor, FloatTensor
import torch
from typing import Protocol, Optional
from .diffusers_denoiser import DiffusersSDDenoiser

class Denoiser(Protocol):
  cond_summation_ixs: LongTensor
  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor: ...

# https://stackoverflow.com/a/59987363
class PostInitMixin(object):
  def __post_init__(self):
    # just intercept the __post_init__ calls so they
    # aren't relayed to `object`
    pass

@dataclass
class AbstractBatchDenoiser(PostInitMixin, ABC, Denoiser):
  denoiser: DiffusersSDDenoiser
  cross_attention_conds: FloatTensor
  cross_attention_mask: Optional[BoolTensor]
  conds_per_prompt: LongTensor
  cond_weights: FloatTensor
  cond_count: int = field(init=False)
  batch_size: int = field(init=False)
  
  def __post_init__(self):
    super().__post_init__()
    self.cond_count = self.cross_attention_conds.size(0)
    self.batch_size = self.conds_per_prompt.size(0)

@dataclass
class BatchNoCFGDenoiser(AbstractBatchDenoiser):
  """
  If you're submitting multi-cond to this:
  ensure prompt weights add up to 1.0.
  """
  def __post_init__(self):
    super().__post_init__()
    self.cond_summation_ixs = torch.arange(self.batch_size, device=self.conds_per_prompt.device).repeat_interleave(self.conds_per_prompt, dim=0).reshape(-1, 1, 1, 1)
    self.cond_weights = self.cond_weights.reshape(-1, 1, 1, 1)

  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor:
    noised_latents_in: FloatTensor = noised_latents.repeat_interleave(self.conds_per_prompt, dim=0, output_size=self.cond_count)
    del noised_latents
    sigma_in: FloatTensor = sigma.repeat_interleave(self.conds_per_prompt, dim=0, output_size=self.cond_count)
    del sigma
    denoised_latents: FloatTensor = self.denoiser.forward(
      input=noised_latents_in,
      sigma=sigma_in,
      encoder_hidden_states=self.cross_attention_conds,
      attention_mask=self.cross_attention_mask,
    )
    del noised_latents_in, sigma_in
    scaled_conds: FloatTensor = denoised_latents * self.cond_weights
    del denoised_latents
    summed: FloatTensor = torch.zeros(self.batch_size, *scaled_conds.shape[1:], dtype=scaled_conds.dtype, device=scaled_conds.device).scatter_add_(0, self.cond_summation_ixs.expand(scaled_conds.shape), scaled_conds)
    return summed

@dataclass
class BatchCFGDenoiser(AbstractBatchDenoiser):
  uncond_ixs: LongTensor
  cfg_scales: FloatTensor
  cond_ixs: LongTensor = field(init=False)
  conds_ex_uncond_per_prompt: LongTensor = field(init=False)
  cond_ex_uncond_count: LongTensor = field(init=False)
  cfg_scaled_cond_weights: FloatTensor = field(init=False)

  def __post_init__(self):
    super().__post_init__()
    all_cond_ixs: LongTensor = torch.arange(self.cond_count, device=self.cross_attention_conds.device)
    is_cond: BoolTensor = torch.isin(all_cond_ixs, self.uncond_ixs, invert=True, assume_unique=True)
    self.cond_ixs = all_cond_ixs.masked_select(is_cond)
    self.conds_ex_uncond_per_prompt = self.conds_per_prompt-1
    self.cond_ex_uncond_count = self.cond_count-self.conds_per_prompt.size(0)
    self.cfg_scaled_cond_weights = (self.cond_weights * self.cfg_scales.repeat_interleave(self.conds_ex_uncond_per_prompt, dim=0)).reshape(-1, 1, 1, 1)
    self.cond_summation_ixs = torch.arange(self.batch_size, device=self.conds_per_prompt.device).repeat_interleave(self.conds_ex_uncond_per_prompt, dim=0).reshape(-1, 1, 1, 1)

  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor:
    noised_latents_in: FloatTensor = noised_latents.repeat_interleave(self.conds_per_prompt, dim=0, output_size=self.cond_count)
    del noised_latents
    sigma_in: FloatTensor = sigma.repeat_interleave(self.conds_per_prompt, dim=0, output_size=self.cond_count)
    del sigma
    denoised_latents: FloatTensor = self.denoiser.forward(
      input=noised_latents_in,
      sigma=sigma_in,
      encoder_hidden_states=self.cross_attention_conds,
      attention_mask=self.cross_attention_mask,
    )
    del noised_latents_in, sigma_in
    unconds: FloatTensor = denoised_latents.index_select(0, self.uncond_ixs)
    conds: FloatTensor = denoised_latents.index_select(0, self.cond_ixs)
    del denoised_latents
    unconds_r: FloatTensor = unconds.repeat_interleave(self.conds_ex_uncond_per_prompt, dim=0, output_size=self.cond_ex_uncond_count)
    diffs: FloatTensor = conds-unconds_r
    del conds, unconds_r
    scaled_diffs: FloatTensor = diffs * self.cfg_scaled_cond_weights
    del diffs
    cfg_denoised: FloatTensor = unconds.scatter_add_(0, self.cond_summation_ixs.expand(scaled_diffs.shape), scaled_diffs)
    return cfg_denoised


@dataclass
class BatchDenoiserFactory():
  denoiser: DiffusersSDDenoiser
  def __call__(
    self,
    cross_attention_conds: FloatTensor,
    cross_attention_mask: Optional[BoolTensor],
    conds_per_prompt: LongTensor,
    cond_weights: FloatTensor,
    uncond_ixs: Optional[LongTensor],
    cfg_scales: Optional[FloatTensor],
  ) -> Denoiser:
    assert (cfg_scales is None) == (uncond_ixs is None)
    if uncond_ixs is None:
      return BatchNoCFGDenoiser(
        denoiser=self.denoiser,
        cross_attention_conds=cross_attention_conds,
        cross_attention_mask=cross_attention_mask,
        conds_per_prompt=conds_per_prompt,
        cond_weights=cond_weights,
      )
    return BatchCFGDenoiser(
      denoiser=self.denoiser,
      cross_attention_conds=cross_attention_conds,
      cross_attention_mask=cross_attention_mask,
      conds_per_prompt=conds_per_prompt,
      cond_weights=cond_weights,
      uncond_ixs=uncond_ixs,
      cfg_scales=cfg_scales,
    )