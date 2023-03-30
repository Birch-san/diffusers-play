from torch import Tensor, load
import torch
from os.path import join, dirname
from typing import OrderedDict
from .decoder import Decoder
from .decoder_ckpt import DecoderCkpt, approx_decoder_ckpt_filenames

repo_root: str = join(dirname(__file__), '../../..')
ckpts_dir: str = join(repo_root, 'approx_vae')

def get_approx_decoder(
  decoder_ckpt: DecoderCkpt,
  device: torch.device = torch.device('cpu'),
) -> Decoder:
  approx_decoder_ckpt: str = join(ckpts_dir, approx_decoder_ckpt_filenames[decoder_ckpt])
  approx_state: OrderedDict[str, Tensor] = load(approx_decoder_ckpt, map_location=device, weights_only=True)
  approx_decoder = Decoder()
  approx_decoder.load_state_dict(approx_state)
  return approx_decoder.eval().to(device)