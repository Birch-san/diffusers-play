from torch import nn

# scaled dot-product attention is formulated as:
#   softmax((q_proj @ k_proj.T) * scale) @ V
# the purpose of the softmax() is to produce a probability distribution,
#   which highlights the largest values and suppresses values significantly below the maximum.
# softmax output is dominated by its inputs' *exponents*.
#
# I hypothesize that we could get a decent approximation of softmax probabilities,
#   by looking *only* at the *exponents* of Q_proj and K_proj.
# advantages of exponent-only arithmetic:
#   - multiplication can be done via integer addition
#     - int addition requires less silicon than fp multiplication
#   - we can use our entire bit width on exponent
#     - int8 supports the same range of exponents as IEEE754 float32
#   - if we use base-2 exponent, we can get cheap exponentiation via bit shifts
#     - 1 << x = 2^x for positive x
#     - x << y = x*2^y for positive x and y
#     - x >> y = x*2^-y for positive x and y
#
#=  0: moving to int exponent domain
# first let's move from float domain into int exponent domain.
#   q_mag = q.log2().to(torch.int8)
# okay, we have problems already: you can't compute the log of a negative number.
#   and logs of numbers close to 0 are very large negative values.
#   they won't win the softmax anyway, but we need wide accumulator to add exponents that large.
#
#=  0.1: magnitudes of negative numbers
# 
# let's separately compute magnitudes of positive and negative exponents,
#   and use an epsilon to avoid returning massively-negative exponents.
#     q_mag_pos = q.clamp(min=torch.finfo(q.dtype).tiny).log2().to(torch.int8)
#     q_mag_neg = q.clamp(max=-torch.finfo(q.dtype).tiny).abs().log2().to(torch.int8)
# note: we may be able to compute these more cheaply by reading the IEEE754 exponent portion:
#     q_mag = (q.frexp().exponent-1).to(torch.int8)
#   we can split those into positive/negative like so:
#     q_mag_pos = (q.clamp(min=0).frexp().exponent-1).to(torch.int8)
#     q_mag_neg = (q.clamp(max=0).frexp().exponent-1).to(torch.int8)
# 
#=  1: projection
# now we need to do projection.
#   projection is a linear layer.
#   which (when bias=False) is just a matmul
#   which is lots of dot products.
# so we need to figure out how to do a dot product.
#
#=  2: dot products
# generally, x @ y is:
#   [x_0 * y_0 + ... + x_n * y_n]
# in floating-point, dot product decomposes to:
#   m = mantissa (aka significand). note: sign lives here
#   e = exponent
#   we assume that exponents are base-2 (though IEEE754 allows for base-10 also).
#     [xm_0*2^xe_0 * ym_0*2^ye_0 + ... + xm_n*2^xe_n * ym_n*2^ye_n]
#    =[xm_0*ym_0*2^(xe_0+ye_0)   + ... + xm_n*ym_n*2^(xe_n+ye_n)  ]
# if we're only interested in exponents, that's:
#     [2^(xe_0+ye_0)             + ... + 2^(xe_n+ye_n)            ]
# we lost our way of expressing negatives though, so we need a new solution for that.
# recall section [0.1] where we decomposed our tensor into views over its positive and negative values.
# 
# x @ y = x_pos @ y_pos + x_neg @ y_neg - x_pos @ y_neg - x_neg @ y_pos
# err this is only true when _neg and _pos tensors are abs().
# we can't really do that. it prevents the tensors' attentuating each other.
# does it get better if we do pairwise addition instead?
# x @ y = x_pos $ y_pos + x_neg $ y_neg - x_pos $ y_neg - x_neg $ y_pos
# 
#
#== 2.1 adding powers-of-2
# 
# how do we add powers-of-2?
# I see two interesting ways to approach this.
#   - take each power-sum, exponentiate via bit-shift, sum results
#     - discussed in Appendix A
#   - keep it in exponent-space
#     - track the negative exponent, don't reify it into a floating-point number until we're forced to
#
#=appendix
#
#== A. sum-of-bit-shifted-power-sums
#
# we can exponentiate via bit-shifts:
#   [1 << (xe_0+ye_0)          + ... + 1 << (xe_n+ye_n)         ]
# we cannot left-shift by negative places.
# we'll have to clamp(min=0) the power-sums.
#   so the smallest 1 << (xe_k+ye_k) would be 1 << 0 = 1
#     we could subtract 1 to ensure that amplified fractional results don't accumulate
#   maybe softmax can tolerate this raised noise floor?
#   dot product's magnitude is dominated by the *largest* (xe_k+ye_k) elements,
#   (or from smaller (xe_k+ye_k) sums in sufficient quantities)
#   and softmax diminishes anything that's sufficiently small
# how big of an accumulator do we need to hold a (xe_k+ye_k)?
#   for int8, each exponent can be as big as 127
#   we can use an uint8 register to hold the summation,
#   and store 0 if the bigger of the elements is negative.
#   so the largest sum we can produce is 254.
#   
# how big of register do we need to hold 1 << 254?
#   very big. you'd need four 64-bit unsigned ints. those don't exist, so five 64-bit signed ints.
#     torch.iinfo(torch.int64).max**5 >= (1 << 254)
# how big of an accumulator do we need to hold sum n of those (1 << 254) results together?
#   in the attention q,k matmul,
#   n = channels-per-head = 40 (for stable-diffusion 1.5)
# so the worst-case is storing the result 40 * (1 << 254)
#   or 1 << (254 + ceil(log2(40)))
#    = 1 << (254 + 6)
#    = 1 << 260
# this still fits into five 64-bit signed ints.
#
#== B. background
# this is just to introduce the terminology I'll be using
# to explain where our inputs came from
# typical tensor shapes based on stable-diffusion 1.5

# we begin with latents
# [b, c, h,  w]  batch, latent_channel, height, width
# [1, 4, 64, 64]

# convolution selects 320 learned features from those 4 latent channels
# [b, c,   h,  w] batch, feature, height, width
# [1, 320, 64, 64]

# we flatten into a sequence of vision tokens.
# [b, c,   t] batch, feature, vision_token
# [1, 320, 4096]
# we transpose to make it channels-last:
# [1, 4096, 320]
# this is our query.
# our context (from which k,v projections will be taken) is:
#   for self-attention: same as query
#   for cross-attention: a CLIP embedding [1, 77, 768]

# now we project Q, K to Q_proj, K_proj
# our projections will be 320-channeled (i.e. 40 channels for each of 8 attention heads)
# so our query is projected,
# onto [batch, tokens, proj_channel]:
# [1, 4096, 320] ->
#  [1, 4096, 320]
# key is computed:
#   for self-attention: same as query (but via different weights)
#   for cross-attention: 
# [1, 77, 768] ->
#  [1, 77, 320]

# we then unflatten the 320 channels into 40 per head.
# query unflattened:
# [1, 4096, 320] ->
#  [1, 4096, 8, 40]
# and transpose to channels-per-token-per-head (rather than channels-per-head-per-token):
# [1, 4096, 8, 40] ->
#  [1, 8, 4096, 40]
# then flatten the batch dim: (heads is just another dimension other which we batch our results):
# [1, 8, 4096, 40] ->
#  [8, 4096, 40]
# key is computed:
#   for self-attention: same as above
#   for cross-attention: same as above, different sequence length
#  [8, 77, 40]

#==can we do projection differently?

# ordinarily we would project like so:
# to_q = nn.Linear(query_dim, inner_dim, bias=False)
# to_k = nn.Linear(context_dim, inner_dim, bias=False)
# q_proj = to_q(query)
# which is equivalent to:
# 


## separating positive and negative parts

from torch import tensor, matmul

q = tensor([[1,  2],
            [3, -1],
            [-2, 3]])
k = tensor([[1, -1, 2, 3],
            [2, 1, -1, 2]])

q_pos = q.clamp(min=0)
q_neg = q.clamp(max=0).abs()
k_pos = k.clamp(min=0)
k_neg = k.clamp(max=0).abs()

((q.unsqueeze(-2).expand(-1, k.size(-1), -1)) * k.T).sum(-1)

# q_pos @ k_pos + q_neg @ k_neg - q_pos @ k_neg - q_neg @ k_pos

# (((q_pos.unsqueeze(-2).expand(-1, k_pos.size(-1), -1)) + k_pos.T)
# + ((q_neg.unsqueeze(-2).expand(-1, k_neg.size(-1), -1)) + k_neg.T)
# - ((q_pos.unsqueeze(-2).expand(-1, k_neg.size(-1), -1)) + k_neg.T)
# - ((q_neg.unsqueeze(-2).expand(-1, k_pos.size(-1), -1)) + k_pos.T)).sum(-1)

##

import torch
from torch import tensor, matmul, arange, iinfo

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
((q_mag.unsqueeze(-2) + k_mag.T).exp2() * k_sgn.T * q_sgn.unsqueeze(-2)).sum(-1)

iinfo_int8 = iinfo(torch.int8)
int8min, int8max = iinfo_int8.min, iinfo_int8.max
q_tokens, channels_per_head = q_mag.shape
_, k_tokens = k_mag.shape

# r = arange(int8min, int8max).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(q_tokens, k_tokens, channels_per_head, -1)
r = arange(int8min, int8max+1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

# p = ((q_mag.unsqueeze(-2) + k_mag.T).exp2() * k_sgn.T * q_sgn.unsqueeze(-2)).unsqueeze(-1)
# (p == r).sum(-2)
# p = (q_mag.unsqueeze(-2) + k_mag.T) * k_sgn.T * q_sgn.unsqueeze(-2)

exp_products = q_mag.unsqueeze(-2) + k_mag.T
exp_prod_counters = exp_products.unsqueeze(-1) == r
signed_counters = exp_prod_counters * k_sgn.T.unsqueeze(0).unsqueeze(-1) * q_sgn.unsqueeze(-2).unsqueeze(-1)
# accumulator needs to be big enough to hold channels_per_head elements. for SD this is just 40
dp_counters = signed_counters.sum(-2, dtype=torch.int8)
for ix in range((1<<7)-1):
  un_carryable = torch.where(dp_counters.abs() <= 1, dp_counters, 0)
  carryable = torch.where(dp_counters.abs() > 1, dp_counters, 0)
  dp_counters = torch.cat([
    un_carryable[:,:,:1<<7],
    un_carryable[:,:,1<<7:]
  ], dim=-1) + torch.cat([
    carryable[:,:,1:(1<<7)-1],
    torch.zeros((*carryable.shape[:-1], 4), dtype=carryable.dtype),
    carryable[:,:,(1<<7)+1:(1<<8)-1]
  ], dim=-1) // 2
(dp_counters * arange(int8min, int8max+1).unsqueeze(0).unsqueeze(0)).max(-1)

# equivalent to q @ k
q @ k
(arange(int8min, int8max+1).unsqueeze(0).unsqueeze(0).exp2() * signed_counters).sum(-2).sum(-1)