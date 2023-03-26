from os import path
from torch import Tensor, load
from torch.nn import Module, Linear, Conv2d
from typing import OrderedDict
from .approx_encoder_ckpt import EncoderCkpt, approx_encoder_ckpt_filenames
import torch

class Encoder(Module):
  # lin: Linear
  proj: Conv2d
  def __init__(self) -> None:
    super().__init__()
    # self.lin = Linear(3, 4, True)
    self.proj = Conv2d(3, 4, kernel_size=3, padding=1)
  
  def forward(self, input: Tensor) -> Tensor:
    # output: Tensor = self.lin(input)
    output: Tensor = self.proj(input)
    return output

def get_approx_encoder(encoder_ckpt: EncoderCkpt, device: torch.device = torch.device('cpu')) -> Encoder:
  approx_encoder_ckpt: str = path.join(path.dirname(__file__), approx_encoder_ckpt_filenames[encoder_ckpt])
  approx_state: OrderedDict[str, Tensor] = load(approx_encoder_ckpt, map_location=device, weights_only=True)
  approx_encoder = Encoder()
  approx_encoder.load_state_dict(approx_state)
  return approx_encoder.eval().to(device)