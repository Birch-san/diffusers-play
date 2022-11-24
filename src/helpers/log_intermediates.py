from functools import partial
import os
from typing import Callable, TypedDict, List
from typing_extensions import TypeAlias
from torch import Tensor
from PIL import Image
from .latents_to_pils import latents_to_pils

class KSamplerCallbackPayload(TypedDict):
  x: Tensor
  i: int
  sigma: Tensor
  sigma_hat: Tensor
  denoised: Tensor

KSamplerCallback: TypeAlias = Callable[[KSamplerCallbackPayload], None]

def log_intermediate(intermediates_path: str, payload: KSamplerCallbackPayload) -> None:
  sample_pils: List[Image.Image] = latents_to_pils(payload['denoised'])
  for img in sample_pils:
    img.save(os.path.join(intermediates_path, f"inter.{payload['i']}.png"))

LogIntermediates: TypeAlias = Callable[[KSamplerCallbackPayload], None]
def make_log_intermediates(intermediates_path: str) -> LogIntermediates:
  return partial(log_intermediate, intermediates_path)