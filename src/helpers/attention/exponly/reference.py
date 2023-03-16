import torch
from torch import FloatTensor
from torch import FloatTensor, baddbmm, zeros, bmm
from torch.nn.functional import linear
from helpers.attention.exponly.fixtures import Fixtures

def reference_attn(
  fixtures: Fixtures,
  device: torch.device
) -> FloatTensor:
  hidden_states: FloatTensor = fixtures.hidden_states
  encoder_hidden_states: FloatTensor = fixtures.encoder_hidden_states
  to_q: FloatTensor = fixtures.to_q
  to_k: FloatTensor = fixtures.to_k
  expected_q_proj: FloatTensor = fixtures.expected_q_proj
  expected_k_proj: FloatTensor = fixtures.expected_k_proj
  expected_scores: FloatTensor = fixtures.expected_scores
  expected_probs: FloatTensor = fixtures.expected_probs
  to_v: FloatTensor = fixtures.to_v
  expected_v_proj: FloatTensor = fixtures.expected_v_proj
  expected_bmm_output: FloatTensor = fixtures.expected_bmm_output
  to_out_weight: FloatTensor = fixtures.to_out_weight
  to_out_bias: FloatTensor = fixtures.to_out_bias
  expected_returned_hidden_states: FloatTensor = fixtures.expected_returned_hidden_states
  heads: int = fixtures.heads
  scale: float = fixtures.scale

  # reference impl
  q_proj: FloatTensor = linear(hidden_states, to_q)         # hidden_states @ to_q.T
  assert q_proj.allclose(expected_q_proj)                   #   rtol=1e-4 if computed via @ operator 
  k_proj: FloatTensor = linear(encoder_hidden_states, to_k) # encoder_hidden_states @ to_k.T
  assert k_proj.allclose(expected_k_proj)                   #   rtol=1e-4 if computed via @ operator
  v_proj: FloatTensor = linear(encoder_hidden_states, to_v) # encoder_hidden_states @ to_v.T
  assert v_proj.allclose(expected_v_proj)                   #   rtol=1e-3 if computed via @ operator

  q_proj = q_proj.unflatten(-1, (heads, -1)).transpose(1,2).flatten(end_dim=1)
  # k_proj_t = k_proj.transpose(1,2).unflatten(1, (heads, -1)).flatten(end_dim=1)
  k_proj = k_proj.unflatten(-1, (heads, -1)).transpose(1,2).flatten(end_dim=1)
  v_proj = v_proj.unflatten(-1, (heads, -1)).transpose(1,2).flatten(end_dim=1)

  attn_bias: FloatTensor = zeros(
    1, 1, 1, dtype=q_proj.dtype, device=device,
  ).expand(*q_proj.shape[0:2], k_proj.shape[1])

  attn_scores: FloatTensor = baddbmm(
    attn_bias,
    q_proj,
    k_proj.transpose(-1, -2),
    beta=0,
    alpha=scale,
  )
  assert not attn_scores.isnan().any().item()
  assert attn_scores.allclose(expected_scores)

  attn_probs: FloatTensor = attn_scores.softmax(dim=-1)
  assert attn_probs.allclose(expected_probs)

  hidden_states: FloatTensor = bmm(attn_probs, v_proj)
  assert hidden_states.allclose(expected_bmm_output) # rtol=1e-3 if v_proj computed via @ operator

  hidden_states = hidden_states.unflatten(0, (-1, heads)).transpose(1,2).flatten(start_dim=2)

  out_proj: FloatTensor = linear(hidden_states, to_out_weight, to_out_bias) # hidden_states @ to_out.T
  assert out_proj.allclose(expected_returned_hidden_states)