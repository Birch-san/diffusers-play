import torch
from torch import tensor, matmul

q_powers = tensor([[1,  2],
                   [3, -1],
                   [-2, 3]], dtype=torch.int8)
k_powers = tensor([[1, -1, 2, 3],
                   [2, 1, -1, 2]], dtype=torch.int8)
q_sign = tensor([[ 1,  1],
                 [-1,  1],
                 [-1, -1]], dtype=torch.int8)
k_sign = tensor([[ 1, -1,  1, 1],
                 [-1, -1, -1, 1]], dtype=torch.int8)
q = q_powers.exp2()*q_sign
k = k_powers.exp2()*k_sign
q_mag = q.abs().log2().to(torch.int8)
q_sgn = q.sign().to(torch.int8)
k_mag = k.abs().log2().to(torch.int8)
k_sgn = k.sign().to(torch.int8)
assert q_sgn.allclose(q_sign)
assert k_sgn.allclose(k_sign)
assert q_mag.allclose(q_powers)
assert k_mag.allclose(k_powers)

# here we go!
q @ k
(((q_mag.unsqueeze(-2).expand(-1, k_mag.size(-1), -1)) + k_mag.T).exp2() * k_sgn.T * q_sgn.unsqueeze(-2).expand(-1, k_sgn.size(-1), -1)).sum(-1)

# ((q.unsqueeze(-2).expand(-1, k.size(-1), -1)) * k.T).sum(-1)

# q_pos = q.clamp(min=0)
# q_neg = q.clamp(max=0)
# k_pos = k.clamp(min=0)
# k_neg = k.clamp(max=0)

# q_pos @ k_pos + q_neg @ k_neg - q_pos @ k_neg - q_neg @ k_pos

# (((q_pos.unsqueeze(-2).expand(-1, k_pos.size(-1), -1)) + k_pos.T)
# + ((q_neg.unsqueeze(-2).expand(-1, k_neg.size(-1), -1)) + k_neg.T)
# - ((q_pos.unsqueeze(-2).expand(-1, k_neg.size(-1), -1)) + k_neg.T)
# - ((q_neg.unsqueeze(-2).expand(-1, k_pos.size(-1), -1)) + k_pos.T)).sum(-1)