from torch import Tensor, load
import torch
from os.path import join, dirname
from typing import OrderedDict
from .encoder import Encoder
from .encoder_ckpt import EncoderCkpt, approx_encoder_ckpt_filenames

repo_root: str = join(dirname(__file__), '../../..')
ckpts_dir: str = join(repo_root, 'approx_vae')

def get_approx_encoder(
  encoder_ckpt: EncoderCkpt,
  device: torch.device = torch.device('cpu'),
) -> Encoder:
  approx_encoder_ckpt: str = join(ckpts_dir, approx_encoder_ckpt_filenames[encoder_ckpt])
  approx_state: OrderedDict[str, Tensor] = load(approx_encoder_ckpt, map_location=device, weights_only=True)
  approx_encoder = Encoder()
  approx_encoder.load_state_dict(approx_state)
  return approx_encoder.eval().to(device)