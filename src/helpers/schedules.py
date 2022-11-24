import torch
from dataclasses import dataclass
from enum import Enum, auto
from torch import Tensor
from .device import DeviceType
from k_diffusion.external import DiscreteSchedule

@dataclass
class KarrasScheduleParams:
  steps: int
  sigma_max: Tensor
  sigma_min: Tensor
  rho: float

class KarrasScheduleTemplate(Enum):
  # aggressively short sigma schedule; for cheap iteration
  Prototyping = auto()
  # cheap but reasonably good results; for exploring seeds to pick one to subsequently master in more detail
  Searching = auto()
  # higher quality, but still not too expensive
  Mastering = auto()

def get_template_schedule(template: KarrasScheduleTemplate, denoiser: DiscreteSchedule) -> KarrasScheduleParams:
  device=denoiser.sigmas.device
  dtype=denoiser.sigmas.dtype
  match(template):
    case KarrasScheduleTemplate.Prototyping:
      return KarrasScheduleParams(
        steps=5,
        sigma_max=torch.tensor(7.0796, device=device, dtype=dtype),
        sigma_min=torch.tensor(0.0936, device=device, dtype=dtype),
        rho=9.
      )
    case KarrasScheduleTemplate.Searching:
      return KarrasScheduleParams(
        steps=8,
        sigma_max=denoiser.sigma_max,
        sigma_min=torch.tensor(0.0936, device=device, dtype=dtype),
        rho=7.
      )
    case KarrasScheduleTemplate.Mastering:
      return KarrasScheduleParams(
        steps=15,
        sigma_max=denoiser.sigma_max,
        sigma_min=denoiser.sigma_min,
        rho=7.
      )
    case _:
      raise f"never heard of a {template} KarrasScheduleTemplate"