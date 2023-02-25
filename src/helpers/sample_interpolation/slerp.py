from torch import FloatTensor, LongTensor, Tensor, lerp, zeros_like, Size
from torch.linalg import norm

# adapted to PyTorch from:
# https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
# most of the extra complexity is to support:
# - many-dimensional vectors
# - v0 or v1 with last dim all zeroes, or v0 and v1 which are approximately colinear
#   - we fall back to lerp()
#   - we do so without conditions, flow control or Python-land loops
# - many-dimensional tensor for t
#   - you can ask for batches of slerp outputs by making t more-dimensional than the vectors
#   -   slerp(
#         v0:   torch.Size([2,3]),
#         v1:   torch.Size([2,3]),
#         t:  torch.Size([4,1,1]), 
#       )
#   - this makes it interface-compatible with lerp()
def slerp(v0: FloatTensor, v1: FloatTensor, t: float|FloatTensor, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        v0: Starting vector
        v1: Final vector
        t: Float value between 0.0 and 1.0
        DOT_THRESHOLD: Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1
    '''
    assert v0.shape == v1.shape, "shapes of v0 and v1 must match"

    # Normalize the vectors to get the directions and angles
    v0_norm: FloatTensor = norm(v0, dim=-1)
    v1_norm: FloatTensor = norm(v1, dim=-1)

    v0_normed: FloatTensor = v0 / v0_norm.unsqueeze(-1)
    v1_normed: FloatTensor = v1 / v1_norm.unsqueeze(-1)

    # Dot product with the normalized vectors
    dot: FloatTensor = (v0_normed * v1_normed).sum(-1)
    dot_mag: FloatTensor = dot.abs()

    # if dp is NaN, it's because the v0 or v1 row was filled with 0s
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    gotta_lerp: LongTensor = dot_mag.isnan() | (dot_mag > DOT_THRESHOLD)
    can_slerp: LongTensor = ~gotta_lerp

    v0_l: FloatTensor = v0.masked_select(gotta_lerp.unsqueeze(-1)).unflatten(dim=-1, sizes=(-1, *[1]*(v0.dim()-2), v0.size(-1)))
    v1_l: FloatTensor = v1.masked_select(gotta_lerp.unsqueeze(-1)).unflatten(dim=-1, sizes=(-1, *[1]*(v1.dim()-2), v1.size(-1)))

    lerped: FloatTensor = lerp(v0_l, v1_l, t)

    v0_s: FloatTensor = v0.masked_select(can_slerp.unsqueeze(-1)).unflatten(dim=-1, sizes=(-1, *[1]*(v0.dim()-2), v0.size(-1)))
    v1_s: FloatTensor = v1.masked_select(can_slerp.unsqueeze(-1)).unflatten(dim=-1, sizes=(-1, *[1]*(v1.dim()-2), v1.size(-1)))
    dot_s: FloatTensor = dot.masked_select(can_slerp).unflatten(dim=-1, sizes=(-1, *[1]*(v1.dim()-1)))

    # Calculate initial angle between v0 and v1
    theta_0: FloatTensor = dot_s.arccos()
    sin_theta_0: FloatTensor = theta_0.sin()
    # Angle at timestep t
    theta_t: FloatTensor = theta_0 * t
    sin_theta_t: FloatTensor = theta_t.sin()
    # Finish the slerp algorithm
    s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
    s1: FloatTensor = sin_theta_t / sin_theta_0
    slerped: FloatTensor = s0 * v0_s + s1 * v1_s

    t_batch_dim_count: int = max(0, t.dim()-v0.dim()) if isinstance(t, Tensor) else 0
    t_batch_dims: Size = t.shape[:t_batch_dim_count] if isinstance(t, Tensor) else Size()
    out: FloatTensor = zeros_like(v0.expand(*t_batch_dims, *[-1]*v0.dim()))
    out: FloatTensor = out.masked_scatter_(gotta_lerp.expand(*t_batch_dims, *gotta_lerp.shape).unsqueeze(-1), lerped)
    out: FloatTensor = out.masked_scatter_(can_slerp.expand(*t_batch_dims, *can_slerp.shape).unsqueeze(-1), slerped)
    
    return out