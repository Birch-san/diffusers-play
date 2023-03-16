import torch
from torch import load, FloatTensor, baddbmm, zeros, bmm
from torch.nn.functional import linear
# from src.helpers.attention.exponly.matmul import 

self = False
half = False
sd2 = True
device = torch.device('mps')
f_dtype = torch.float16 if half else torch.float32

dtype_ext = 'fp16' if f_dtype is torch.float16 else 'fp32'

heads = 5 if sd2 else 8
dim_head = 64 if sd2 else 40
# scale = 0.125 if sd2 else 0
scale = dim_head**-.5

dir = 'wd1_5' if sd2 else 'wd1_3'
selfq = 'self' if self else 'cross'

hidden_states: FloatTensor = load(f'out_attn/{dir}/{selfq}_hidden_states.{dtype_ext}.pt', map_location=device, weights_only=True)
encoder_hidden_states: FloatTensor = hidden_states if self else load(f'out_attn/{dir}/encoder_hidden_states.{dtype_ext}.pt', map_location=device, weights_only=True)

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