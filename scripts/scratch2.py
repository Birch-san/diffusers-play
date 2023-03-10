from torch import tensor, matmul

q = tensor([[1,  2],
            [3, -1],
            [-2, 3]])
k = tensor([[1, -1, 2, 3],
            [2, 1, -1, 2]])

q_pos = q.clamp(min=0)
q_neg = q.clamp(max=0)
k_pos = k.clamp(min=0)
k_neg = k.clamp(max=0)

((q.unsqueeze(-2).expand(-1, k.size(-1), -1)) * k.T).sum(-1)


# q_pos @ k_pos + q_neg @ k_neg - q_pos @ k_neg - q_neg @ k_pos

(((q_pos.unsqueeze(-2).expand(-1, k_pos.size(-1), -1)) + k_pos.T)
+ ((q_neg.unsqueeze(-2).expand(-1, k_neg.size(-1), -1)) + k_neg.T)
- ((q_pos.unsqueeze(-2).expand(-1, k_neg.size(-1), -1)) + k_neg.T)
- ((q_neg.unsqueeze(-2).expand(-1, k_pos.size(-1), -1)) + k_pos.T)).sum(-1)