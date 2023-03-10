import torch
from torch import tensor, matmul, Tensor
import math

# def softmax(x: Tensor, dim=1) -> Tensor:
#   means = torch.mean(x, dim, keepdim=True)[0]
#   x_exp = torch.exp(x-means)
#   x_exp_sum = torch.sum(x_exp, dim, keepdim=True)

#   return x_exp/x_exp_sum
def softmax(x: Tensor, dim=1) -> Tensor:
  maxes = torch.max(x, dim, keepdim=True).values
  x_exp = torch.exp(x-maxes)
  x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
  return x_exp/x_exp_sum

scale = 64**-0.5

# q = tensor([[0.25, 0.5],[2,4]])
# k = tensor([[1.  , 2. ],[4,8]])
q = tensor([[2, 8],[2,4]], dtype=torch.float32)
k = tensor([[1.  , 2. ],[4,8]], dtype=torch.float32)
kt = k.transpose(-1, -2)

s = matmul(q, kt)
# aka:
# (q.unsqueeze(-1).repeat(1,1,k.size(-1)) * kt).sum(-2)
se = s.frexp().exponent # seems to align with u, except u is twice

ss = s * scale
sss = ss.softmax(dim=-1)

# qf = q.frexp().exponent
# kf = k.frexp().exponent
# qe = q.log2()
# ke = k.log2()
# qe = q.log()
# ke = k.log()
qe = q.exp()
ke = k.exp()
# qe = q
# ke = k
qu = qe.sum(-1).unsqueeze(-1)
ku = ke.sum(-1)
u = qu+ku
uu = u# * scale
uuu = uu.softmax(dim=-1)
# qs = (q.unsqueeze(-2).expand(-1, k.size(-1), -1)+kt).exp().sum(-2)
# qs = (q.exp().unsqueeze(-1).repeat(1,1,k.size(-1)) + kt.exp()).sum(-2)
# qs = (q.unsqueeze(-1).repeat(1,1,k.size(-1)) + kt).exp().sum(-2)
# qs = (q.log().unsqueeze(-1).repeat(1,1,k.size(-1)) + kt.log()).exp().sum(-2)
# (q.log().unsqueeze(-1).repeat(1,1,k.size(-1)) + kt.log()).exp().sum(-2)

qm = (q.clamp(min=0.0001).frexp().exponent.to(torch.int8)-1).clamp(max=15)
ktm = (kt.clamp(min=0.0001).frexp().exponent.to(torch.int8)-1).clamp(max=15)
# qm = q.clamp(min=0.0001).log2().to(torch.int8).clamp(max=15)
# ktm = kt.clamp(min=0.0001).log2().to(torch.int8).clamp(max=15)
# (qm.unsqueeze(-1).repeat(1,1,k.size(-1)) + ktm).exp().sum(-2)

# approx s
b_u = torch.bitwise_left_shift(torch.full((1,), 1, dtype=torch.int16), qm.unsqueeze(-1).repeat(1,1,k.size(-1)) + ktm).sum(-2, dtype=torch.int16)
# approx ss
b = torch.bitwise_left_shift(torch.full((1,), 1, dtype=torch.int16), qm.unsqueeze(-1).repeat(1,1,k.size(-1)) + ktm + int(math.log2(scale))).sum(-2, dtype=torch.int16)

# b = torch.bitwise_left_shift(torch.full((1,), 1, dtype=torch.int16), qm.unsqueeze(-1).repeat(1,1,k.size(-1)) + ktm).sum(-2, dtype=torch.int16)
# bb = b * scale
# bbb = bb/bb.sum()
# bbb = b.softmax(dim=-1)
b_exps = torch.bitwise_left_shift(torch.full((1,), 1, dtype=torch.int16), b)
# bbb = b_exps / b_exps.sum()
bbb = b_exps.float().softmax(dim=-1)


pass
# https://pytorch.org/docs/stable/generated/torch.logsumexp.html?highlight=sum#torch.logsumexp

# matmul is:
#   ((q.unsqueeze(-2).expand(-1, k.size(-1), -1)) * k.T).sum(-1)
# the intermediate,
#   q.unsqueeze(-2).expand(-1, k.size(-1), -1)
# has shape:
#   q_tokens, channels = q.shape
#   channels, k_tokens = k.shape
#   [q_tokens, k_tokens, channels]
# then once you sum(-1) them:
#   [q_tokens, k_tokens]