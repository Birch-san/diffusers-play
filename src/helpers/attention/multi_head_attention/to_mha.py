from diffusers.models.attention import CrossAttention
from .multi_head_attention import MultiheadAttention
from torch import cat

def to_mha(ca: CrossAttention) -> MultiheadAttention:
    bias = ca.to_k.bias is not None
    assert bias == False
    mha = MultiheadAttention(
        query_dim=ca.to_q.in_features,
        cross_attention_dim=ca.to_k.in_features,
        heads=ca.heads,
        dim_head=ca.to_q.out_features//ca.heads,
        dropout=ca.to_out[1].p,
        bias=bias,
    )
    # is self-attention?
    if ca.to_q.in_features == ca.to_k.in_features:
        mha.get_parameter('in_proj_weight').data = cat([ca.to_q.weight, ca.to_k.weight, ca.to_v.weight])
    else:
        mha.get_parameter('q_proj_weight').data = ca.to_q.weight
        mha.get_parameter('k_proj_weight').data = ca.to_k.weight
        mha.get_parameter('v_proj_weight').data = ca.to_v.weight
    mha.out_proj.weight = ca.to_out[0].weight
    mha.out_proj.bias = ca.to_out[0].bias
    return mha
