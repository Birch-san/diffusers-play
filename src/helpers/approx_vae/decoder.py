from torch.nn import Module, ModuleList, Linear, SiLU
from torch import Tensor, load
import torch
from os.path import join, dirname
from typing import OrderedDict
from .decoder_ckpt import DecoderCkpt, approx_decoder_ckpt_filenames

class Decoder(Module):
  in_proj: Linear
  hidden_layers: ModuleList
  out_proj: Linear
  def __init__(self, hidden_layer_count=1, inner_dim=12) -> None:
    super().__init__()
    self.in_proj = Linear(4, inner_dim)
    make_nonlin = SiLU
    self.nonlin = make_nonlin()
    self.hidden_layers = ModuleList([
      layer for layer in (Linear(inner_dim, inner_dim), make_nonlin()) for _ in range(hidden_layer_count)
    ])
    self.out_proj = Linear(inner_dim, 3)
  
  def forward(self, sample: Tensor) -> Tensor:
    sample: Tensor = self.in_proj(sample)
    sample: Tensor = self.nonlin(sample)
    for layer in self.hidden_layers:
      sample: Tensor = layer.forward(sample)
    sample: Tensor = self.out_proj(sample)
    return sample

def get_approx_decoder(
  decoder_ckpt: DecoderCkpt,
  device: torch.device = torch.device('cpu'),
) -> Decoder:  
  approx_decoder_ckpt: str = join(dirname(__file__), approx_decoder_ckpt_filenames[decoder_ckpt])
  approx_state: OrderedDict[str, Tensor] = load(approx_decoder_ckpt, map_location=device, weights_only=True)
  approx_decoder = Decoder()
  approx_decoder.load_state_dict(approx_state)
  return approx_decoder.eval().to(device)