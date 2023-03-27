from os import path
from torch import Tensor, load
from torch.nn import Module, Linear, ModuleList, SiLU
from typing import OrderedDict
from .approx_encoder_ckpt import EncoderCkpt, approx_encoder_ckpt_filenames
import torch

class Encoder(Module):
  in_proj: Linear
  hidden_layers: ModuleList
  out_proj: Linear
  def __init__(self, hidden_layer_count=1, inner_dim=12) -> None:
    super().__init__()
    self.in_proj = Linear(3, inner_dim)
    make_nonlin = SiLU
    self.nonlin = make_nonlin()
    self.hidden_layers = ModuleList([
      layer for layer in (Linear(inner_dim, inner_dim), make_nonlin()) for _ in range(hidden_layer_count)
    ])
    self.out_proj = Linear(inner_dim, 4)
  
  def forward(self, sample: Tensor) -> Tensor:
    sample: Tensor = self.in_proj(sample)
    sample: Tensor = self.nonlin(sample)
    for layer in self.hidden_layers:
      sample: Tensor = layer.forward(sample)
    sample: Tensor = self.out_proj(sample)
    return sample

def get_approx_encoder(encoder_ckpt: EncoderCkpt, device: torch.device = torch.device('cpu')) -> Encoder:
  approx_encoder_ckpt: str = path.join(path.dirname(__file__), approx_encoder_ckpt_filenames[encoder_ckpt])
  approx_state: OrderedDict[str, Tensor] = load(approx_encoder_ckpt, map_location=device, weights_only=True)
  approx_encoder = Encoder()
  approx_encoder.load_state_dict(approx_state)
  return approx_encoder.eval().to(device)