from dataclasses import dataclass
from torch import load, FloatTensor
from typing import Literal
import torch

@dataclass
class Fixtures:
  hidden_states: FloatTensor
  encoder_hidden_states: FloatTensor
  to_q: FloatTensor
  to_k: FloatTensor
  expected_q_proj: FloatTensor
  expected_k_proj: FloatTensor
  expected_scores: FloatTensor
  expected_probs: FloatTensor
  to_v: FloatTensor
  expected_v_proj: FloatTensor
  expected_bmm_output: FloatTensor
  to_out_weight: FloatTensor
  to_out_bias: FloatTensor
  expected_returned_hidden_states: FloatTensor
  heads: int
  scale: float

def get_fixtures(
  self_attn: bool,
  half: bool,
  sd2: bool,
  device: torch.device,
) -> Fixtures:
  f_dtype = torch.float16 if half else torch.float32

  dtype_ext: Literal['fp16', 'fp32'] = 'fp16' if f_dtype is torch.float16 else 'fp32'

  heads: Literal[5, 8] = 5 if sd2 else 8
  dim_head: Literal[64, 40] = 64 if sd2 else 40
  scale: float = dim_head**-.5
  # scale = 0.125 if sd2 else 0

  dir: Literal['wd1_5', 'wd1_3'] = 'wd1_5' if sd2 else 'wd1_3'
  selfq: Literal['self', 'cross'] = 'self' if self_attn else 'cross'

  hidden_states: FloatTensor = load(f'out_attn/{dir}/{selfq}_hidden_states.{dtype_ext}.pt', map_location=device, weights_only=True)
  encoder_hidden_states: FloatTensor = hidden_states if self_attn else load(f'out_attn/{dir}/encoder_hidden_states.{dtype_ext}.pt', map_location=device, weights_only=True)

  to_q: FloatTensor = load(f'out_attn/{dir}/{selfq}_to_q.{dtype_ext}.pt', map_location=device, weights_only=True)
  to_k: FloatTensor = load(f'out_attn/{dir}/{selfq}_to_k.{dtype_ext}.pt', map_location=device, weights_only=True)
  expected_q_proj: FloatTensor = load(f'out_attn/{dir}/{selfq}_q_proj.{dtype_ext}.pt', map_location=device, weights_only=True)
  expected_k_proj: FloatTensor = load(f'out_attn/{dir}/{selfq}_k_proj.{dtype_ext}.pt', map_location=device, weights_only=True)

  expected_scores: FloatTensor = load(f'out_attn/{dir}/{selfq}_attention_scores.{dtype_ext}.pt', map_location=device, weights_only=True)
  expected_probs: FloatTensor = load(f'out_attn/{dir}/{selfq}_attention_probs.{dtype_ext}.pt', map_location=device, weights_only=True)

  to_v: FloatTensor = load(f'out_attn/{dir}/{selfq}_to_v.{dtype_ext}.pt', map_location=device, weights_only=True)
  expected_v_proj: FloatTensor = load(f'out_attn/{dir}/{selfq}_v_proj.{dtype_ext}.pt', map_location=device, weights_only=True)
  expected_bmm_output: FloatTensor = load(f'out_attn/{dir}/{selfq}_bmm_output.{dtype_ext}.pt', map_location=device, weights_only=True)

  to_out_weight: FloatTensor = load(f'out_attn/{dir}/{selfq}_to_out.{dtype_ext}.pt', map_location=device, weights_only=True)
  to_out_bias: FloatTensor = load(f'out_attn/{dir}/{selfq}_to_out_bias.{dtype_ext}.pt', map_location=device, weights_only=True)
  expected_returned_hidden_states: FloatTensor = load(f'out_attn/{dir}/{selfq}_returned_hidden_states.{dtype_ext}.pt', map_location=device, weights_only=True)

  return Fixtures(
    hidden_states=hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    to_q=to_q,
    to_k=to_k,
    expected_q_proj=expected_q_proj,
    expected_k_proj=expected_k_proj,
    expected_scores=expected_scores,
    expected_probs=expected_probs,
    to_v=to_v,
    expected_v_proj=expected_v_proj,
    expected_bmm_output=expected_bmm_output,
    to_out_weight=to_out_weight,
    to_out_bias=to_out_bias,
    expected_returned_hidden_states=expected_returned_hidden_states,
    heads=heads,
    scale=scale,
  )