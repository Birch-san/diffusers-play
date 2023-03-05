from abc import ABC
from dataclasses import dataclass, field
from torch import LongTensor, BoolTensor, FloatTensor, ones_like
import torch
from typing import Protocol, Optional
from .diffusers_denoiser import DiffusersSDDenoiser
from .post_init import PostInitMixin

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
  cfg_scales: FloatTensor = field(repr=False)
  mimic_scales: Optional[FloatTensor] = field(repr=False)
  dynthresh_percentile: Optional[float]
  channel_limits: Optional[FloatTensor]
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
    # del self.cfg_scales
    if self.channel_limits is None:
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
  
  def _limit_channels(
    self,
    conds_scattered: FloatTensor,
    unconds: FloatTensor,
    diffs: FloatTensor,
  ) -> FloatTensor:
    dimensions = unconds.shape[-2:]

    conds_means: FloatTensor = conds_scattered.mean(dim=(-2,-1))
    conds_centered: FloatTensor = conds_scattered.flatten(-2)-conds_means.unsqueeze(-1)
    conds_max: torch.return_types.max = conds_centered.abs().max(dim=-1)
    conds_absmax: FloatTensor = conds_max.values
    conds_agnmax: FloatTensor = conds_centered.gather(-1, conds_max.indices.unsqueeze(-1))

    nocfg: FloatTensor = self._compute_for_scale(unconds.detach().clone(), diffs, ones_like(self.cfg_scaled_cond_weights))
    nocfg_means: FloatTensor = nocfg.mean(dim=(-2,-1))
    nocfg_centered: FloatTensor = nocfg.flatten(-2)-nocfg_means.unsqueeze(-1)
    nocfg_max: torch.return_types.max = nocfg_centered.abs().max(dim=-1)
    nocfg_absmax: FloatTensor = nocfg_max.values
    nocfg_agnmax: FloatTensor = nocfg_centered.gather(-1, nocfg_max.indices.unsqueeze(-1))

    uncond_means: FloatTensor = unconds.mean(dim=(-2,-1))
    uncond_centered: FloatTensor = unconds.flatten(-2)-uncond_means.unsqueeze(-1)
    uncond_max: torch.return_types.max = uncond_centered.abs().max(dim=-1)
    uncond_absmax: FloatTensor = uncond_max.values
    uncond_agnmax: FloatTensor = uncond_centered.gather(-1, uncond_max.indices.unsqueeze(-1))

    diff_means: FloatTensor = diffs.mean(dim=(-2,-1))
    diff_centered: FloatTensor = diffs.flatten(-2)-diff_means.unsqueeze(-1)
    diff_max: torch.return_types.max = diff_centered.abs().max(dim=-1)
    diff_absmax: FloatTensor = diff_max.values
    diff_agnmax: FloatTensor = diff_centered.gather(-1, diff_max.indices.unsqueeze(-1))
    
    actual: FloatTensor = self._compute_for_scale(unconds.detach().clone(), diffs, self.cfg_scaled_cond_weights)
    means: FloatTensor = actual.mean(dim=(-2,-1))
    centered: FloatTensor = actual.flatten(-2)-means.unsqueeze(-1)
    max: torch.return_types.max = centered.abs().max(dim=-1)
    actual_absmax: FloatTensor = max.values
    actual_agnmax: FloatTensor = centered.gather(-1, max.indices.unsqueeze(-1))

    # (self._compute_for_scale(unconds.detach().clone(), diffs, self.cfg_scaled_cond_weights*2).mean(dim=(-2,-1))+uncond_means)/2

    # rescaled: FloatTensor = clamped * ratio.unsqueeze(-1)
    # del clamped, ratio
    # decentered: FloatTensor = rescaled + means.unsqueeze(-1)
    # del rescaled, means
    # unflattened: FloatTensor = decentered.unflatten(-1, dimensions)

    cfg_scales_broadcast: FloatTensor = self.cfg_scales.reshape(-1, 1, 1)
    adjusted: FloatTensor = uncond_centered*(1-cfg_scales_broadcast) + conds_centered*cfg_scales_broadcast
    unflattened: FloatTensor = adjusted.unflatten(-1, dimensions)

    return unflattened

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
    if self.channel_limits is not None:
      scaled_conds: FloatTensor = conds * self.cond_weights.reshape(-1, 1, 1, 1)
      conds_scattered: FloatTensor = torch.zeros(self.batch_size, *scaled_conds.shape[1:], dtype=scaled_conds.dtype, device=scaled_conds.device).scatter_add_(0, self.cond_summation_ixs.expand(scaled_conds.shape), scaled_conds)
      return self._limit_channels(conds_scattered, unconds, diffs)
    del conds, unconds_r
    unconds_backup: Optional[FloatTensor] = None if self.mimic_scaled_cond_weights is None else unconds.detach().clone()
    cfg_denoised: FloatTensor = self._compute_for_scale(unconds, diffs, self.cfg_scaled_cond_weights)
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
    cross_attention_mask: Optional[BoolTensor],
    conds_per_prompt: LongTensor,
    cond_weights: FloatTensor,
    uncond_ixs: Optional[LongTensor],
    cfg_scales: Optional[FloatTensor],
    mimic_scales: Optional[FloatTensor],
    dynthresh_percentile: Optional[float],
    channel_limits: Optional[FloatTensor],
  ) -> Denoiser:
    assert (cfg_scales is None) == (uncond_ixs is None)
    if dynthresh_percentile is not None:
      assert mimic_scales is not None, "use of dynthresh requires specifying a mimic_scale too"
    if channel_limits is not None:
      # assert mimic_scales is None and dynthresh_percentile is None, "channel_limits cannot be used with dynthresh"
      assert dynthresh_percentile is None, "channel_limits cannot be used with dynthresh"
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
      mimic_scales=mimic_scales,
      dynthresh_percentile=dynthresh_percentile,
      channel_limits=channel_limits,
    )