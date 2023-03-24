from abc import ABC
from dataclasses import dataclass, field
from torch import LongTensor, BoolTensor, FloatTensor, where
import torch
from typing import Protocol, Optional
from .diffusers_denoiser import DiffusersSDDenoiser
from .post_init import PostInitMixin
from .dynthresh_latent_roundtrip import LatentsToRGB, RGBToLatents

class Denoiser(Protocol):
  cond_summation_ixs: LongTensor
  def __call__(
    self,
    noised_latents: FloatTensor,
    sigma: FloatTensor,
  ) -> FloatTensor: ...

@dataclass
class AbstractBatchDenoiser(PostInitMixin, ABC, Denoiser):
  denoiser: DiffusersSDDenoiser
  cross_attention_conds: FloatTensor
  cross_attention_bias: Optional[FloatTensor]
  conds_per_prompt: LongTensor
  cond_weights: FloatTensor
  center_denoise_outputs: Optional[BoolTensor]
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
      cross_attention_bias=self.cross_attention_bias,
    )
    del noised_latents_in, sigma_in
    if self.center_denoise_outputs is not None:
      denoised_latents = where(
        self.center_denoise_outputs,
        denoised_latents-denoised_latents.mean(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1).expand(denoised_latents.shape),
        denoised_latents,
      )
    scaled_conds: FloatTensor = denoised_latents * self.cond_weights
    del denoised_latents
    summed: FloatTensor = torch.zeros(self.batch_size, *scaled_conds.shape[1:], dtype=scaled_conds.dtype, device=scaled_conds.device).scatter_add_(0, self.cond_summation_ixs.expand(scaled_conds.shape), scaled_conds)
    return summed

@dataclass
class BatchCFGDenoiser(AbstractBatchDenoiser):
  uncond_ixs: LongTensor
  cfg_scales: FloatTensor = field(repr=False)
  mimic_scales: Optional[FloatTensor] = field(repr=False)
  dynthresh_percentile: Optional[float]
  dynthresh_latent_decoder: Optional[LatentsToRGB]
  dynthresh_latent_encoder: Optional[RGBToLatents]
  pixel_space_dynthresh: bool
  cond_ixs: LongTensor = field(init=False)
  conds_ex_uncond_per_prompt: LongTensor = field(init=False)
  cond_ex_uncond_count: LongTensor = field(init=False)
  cfg_scaled_cond_weights: FloatTensor = field(init=False)
  mimic_scaled_cond_weights: Optional[FloatTensor] = field(init=False)

  def __post_init__(self):
    super().__post_init__()
    all_cond_ixs: LongTensor = torch.arange(self.cond_count, device=self.cross_attention_conds.device)
    is_cond: BoolTensor = torch.isin(all_cond_ixs, self.uncond_ixs, invert=True, assume_unique=True)
    self.cond_ixs = all_cond_ixs.masked_select(is_cond)
    self.conds_ex_uncond_per_prompt = self.conds_per_prompt-1
    self.cond_ex_uncond_count = self.cond_count-self.conds_per_prompt.size(0)
    self.cfg_scaled_cond_weights = (self.cond_weights * self.cfg_scales.repeat_interleave(self.conds_ex_uncond_per_prompt, dim=0)).reshape(-1, 1, 1, 1)
    if self.mimic_scales is not None:
      # we use .minimum() because range mimicking is intended to give you *smaller* values, to prevent deep-frying
      self.mimic_scaled_cond_weights = (self.cond_weights * torch.minimum(self.cfg_scales, self.mimic_scales).repeat_interleave(self.conds_ex_uncond_per_prompt, dim=0)).reshape(-1, 1, 1, 1)
      del self.mimic_scales
    else:
      self.mimic_scaled_cond_weights = None
    del self.cond_weights, self.cfg_scales
    self.cond_summation_ixs = torch.arange(self.batch_size, device=self.conds_per_prompt.device).repeat_interleave(self.conds_ex_uncond_per_prompt, dim=0).reshape(-1, 1, 1, 1)
  
  def _compute_for_scale(
    self,
    unconds: FloatTensor,
    diffs: FloatTensor,
    scaled_cond_weights: FloatTensor,
  ) -> FloatTensor:
    """
    Mutates `unconds`
    """
    scaled_diffs: FloatTensor = diffs * scaled_cond_weights
    cfg_denoised: FloatTensor = unconds.scatter_add_(0, self.cond_summation_ixs.expand(scaled_diffs.shape), scaled_diffs)
    return cfg_denoised

  def _mimic_scale(
    self,
    target: FloatTensor,
    actual: FloatTensor,
  ) -> FloatTensor:
    dimensions = target.shape[-2:]
    target_means: FloatTensor = target.mean(dim=(-2,-1))
    target_centered: FloatTensor = target.flatten(-2)-target_means.unsqueeze(-1)
    del target_means
    target_max: FloatTensor = target_centered.abs().max(dim=-1).values
    del target_centered

    actual_means: FloatTensor = actual.mean(dim=(-2,-1))
    actual_centered: FloatTensor = actual.flatten(-2)-actual_means.unsqueeze(-1)

    if self.dynthresh_percentile is None:
      actual_peak: FloatTensor = actual_centered.abs().max(dim=-1).values
      actual_clamped: FloatTensor = actual_centered
    else:
      actual_q: FloatTensor = torch.quantile(actual_centered.abs(), self.dynthresh_percentile, dim=-1)
      actual_peak: FloatTensor = torch.maximum(actual_q, target_max)
      del actual_q
      actual_peak_broadcast: FloatTensor = actual_peak.unsqueeze(-1)
      actual_clamped: FloatTensor = actual_centered.clamp(
        min=-actual_peak_broadcast,
        max=actual_peak_broadcast,
      )
      del actual_peak_broadcast
    del actual_centered
    ratio: FloatTensor = target_max/actual_peak
    del target_max, actual_peak

    rescaled: FloatTensor = actual_clamped * ratio.unsqueeze(-1)
    del actual_clamped, ratio
    decentered: FloatTensor = rescaled + actual_means.unsqueeze(-1)
    del rescaled, actual_means
    unflattened: FloatTensor = decentered.unflatten(-1, dimensions)

    return unflattened

  def _pixel_space_dynthresh(
    self,
    denoised_latents: FloatTensor
  ) -> FloatTensor:
    rgb: FloatTensor = self.dynthresh_latent_decoder(denoised_latents)
    int8_iinfo = torch.iinfo(torch.int8)
    int8_range = int8_iinfo.max-int8_iinfo.min
    int8_half_range = int8_range / 2
    centered = rgb - int8_half_range
    normed = centered / int8_half_range

    s: FloatTensor = torch.quantile(
      normed.flatten(start_dim=1).abs(),
      self.dynthresh_percentile,
      dim = -1
    )
    s.clamp_(min = 1.)
    s = s.reshape(*s.shape, 1, 1, 1)
    normed.clamp(-s, s) #/ s
    decentered = normed + 1.
    scaled = decentered * int8_half_range

    latents: FloatTensor = self.dynthresh_latent_encoder(scaled)
    return latents

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
      cross_attention_bias=self.cross_attention_bias,
    )
    if self.center_denoise_outputs is not None:
      denoised_latents = where(
        self.center_denoise_outputs,
        denoised_latents-denoised_latents.mean(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1).expand(denoised_latents.shape),
        denoised_latents,
      )
    del noised_latents_in, sigma_in
    unconds: FloatTensor = denoised_latents.index_select(0, self.uncond_ixs)
    conds: FloatTensor = denoised_latents.index_select(0, self.cond_ixs)
    del denoised_latents
    unconds_r: FloatTensor = unconds.repeat_interleave(self.conds_ex_uncond_per_prompt, dim=0, output_size=self.cond_ex_uncond_count)
    diffs: FloatTensor = conds-unconds_r
    del conds, unconds_r
    unconds_backup: Optional[FloatTensor] = None if self.mimic_scaled_cond_weights is None else unconds.detach().clone()
    cfg_denoised: FloatTensor = self._compute_for_scale(unconds, diffs, self.cfg_scaled_cond_weights)
    if self.pixel_space_dynthresh and self.dynthresh_percentile is not None:
      cfg_denoised = self._pixel_space_dynthresh(cfg_denoised)
    if self.mimic_scaled_cond_weights is None:
      return cfg_denoised
    target: FloatTensor = self._compute_for_scale(unconds_backup, diffs, self.mimic_scaled_cond_weights)
    return self._mimic_scale(target=target, actual=cfg_denoised)




@dataclass
class BatchDenoiserFactory():
  denoiser: DiffusersSDDenoiser
  def __call__(
    self,
    cross_attention_conds: FloatTensor,
    cross_attention_bias: Optional[FloatTensor],
    conds_per_prompt: LongTensor,
    cond_weights: FloatTensor,
    uncond_ixs: Optional[LongTensor],
    cfg_scales: Optional[FloatTensor],
    mimic_scales: Optional[FloatTensor],
    dynthresh_percentile: Optional[float],
    center_denoise_outputs: Optional[BoolTensor],
    dynthresh_latent_decoder: Optional[LatentsToRGB],
    dynthresh_latent_encoder: Optional[RGBToLatents],
    pixel_space_dynthresh: bool = True,
  ) -> Denoiser:
    assert (cfg_scales is None) == (uncond_ixs is None)
    if uncond_ixs is None:
      return BatchNoCFGDenoiser(
        denoiser=self.denoiser,
        cross_attention_conds=cross_attention_conds,
        cross_attention_bias=cross_attention_bias,
        conds_per_prompt=conds_per_prompt,
        cond_weights=cond_weights,
        center_denoise_outputs=center_denoise_outputs,
      )
    return BatchCFGDenoiser(
      denoiser=self.denoiser,
      cross_attention_conds=cross_attention_conds,
      cross_attention_bias=cross_attention_bias,
      conds_per_prompt=conds_per_prompt,
      cond_weights=cond_weights,
      center_denoise_outputs=center_denoise_outputs,
      uncond_ixs=uncond_ixs,
      cfg_scales=cfg_scales,
      mimic_scales=mimic_scales,
      dynthresh_percentile=dynthresh_percentile,
      dynthresh_latent_decoder=dynthresh_latent_decoder,
      dynthresh_latent_encoder=dynthresh_latent_encoder,
      pixel_space_dynthresh=pixel_space_dynthresh,
    )