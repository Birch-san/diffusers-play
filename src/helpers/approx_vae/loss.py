from dataclasses import dataclass
from torch import sub, FloatTensor
from torch.nn import MSELoss, L1Loss
from typing import NamedTuple

l2_loss = MSELoss()
l1_loss = L1Loss()

@dataclass
class LossBreakdown:
  l2: FloatTensor
  l1: FloatTensor
  range: FloatTensor
  l2_scaled: FloatTensor
  l1_scaled: FloatTensor
  range_scaled: FloatTensor

class LossComponents(NamedTuple):
  total_loss: FloatTensor
  breakdown: LossBreakdown

def describe_loss(loss_components: LossComponents) -> str:
  loss, b = loss_components
  unscaled_components = f'l2: {b.l2.abs().max().item():.2f}, l1: {b.l1.abs().max().item():.2f}, r: {b.range.abs().max().item():.2f}'
  scaled_components = f'l2: {b.l2_scaled.abs().max().item():.2f}, l1: {b.l1_scaled.abs().max().item():.2f}, r: {b.range_scaled.abs().max().item():.2f}'
  return f'loss: {loss.item():.2f}, [u] {unscaled_components} [s] {scaled_components}'

def loss_fn(input: FloatTensor, target: FloatTensor) -> LossComponents:
  # return l2_loss(input, target) + 0.05 * l1_loss(input, target) + 0.05 * (input.abs().max() - 1).clamp(min=0)**2
  # return l2_loss(input, target) #+ 0.025 * (input.abs().max() - 1).clamp(min=0)**2
  # return 0.9 * l2_loss(input, target) + 0.1 * l1_loss(input, target) #+ 0.025 * (input.abs().max() - 1).clamp(min=0)**2
  l2 = l2_loss(input, target)
  l2_scaled = 1. * l2
  l1 = l1_loss(input, target)
  l1_scaled = 0. * l1
  range = sub(input.abs().max(dim=0).values, 1).clamp(min=0).mean()
  range_scaled = 0. * range
  breakdown = LossBreakdown(
    l2=l2,
    l1=l1,
    range=range,
    l2_scaled=l2_scaled,
    l1_scaled=l1_scaled,
    range_scaled=range_scaled,
  )
  total_loss: FloatTensor = l2_scaled + l1_scaled + range_scaled
  return LossComponents(
    total_loss=total_loss,
    breakdown=breakdown,
  )