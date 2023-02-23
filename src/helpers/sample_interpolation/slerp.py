from torch import FloatTensor, LongTensor, Tensor, lerp, zeros_like, where
from torch.linalg import norm

# adapted to PyTorch from:
# https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
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

    v0_l: FloatTensor = v0.masked_select(gotta_lerp.unsqueeze(-1)).unflatten(dim=0, sizes=(-1, v0.size(-1)))
    v1_l: FloatTensor = v1.masked_select(gotta_lerp.unsqueeze(-1)).unflatten(dim=0, sizes=(-1, v1.size(-1)))

    lerped: FloatTensor = lerp(v0_l, v1_l, t)

    v0_s: FloatTensor = v0.masked_select(can_slerp.unsqueeze(-1)).unflatten(dim=-1, sizes=(-1, v0.size(-1)))
    v1_s: FloatTensor = v1.masked_select(can_slerp.unsqueeze(-1)).unflatten(dim=-1, sizes=(-1, v1.size(-1)))
    # TODO: can we do this with unflatten?
    dot_s: FloatTensor = dot.masked_select(can_slerp).unsqueeze(0)

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
    
    # TODO: we want [t_batch, token, feature]
    #       we have [t_batch, feature]
    # slerped is weird. it's:
    #   torch.Size([4, 3])
    # we want:
    #   torch.Size([4, 1, 3])
    # slerped.unsqueeze(-2)
    index: FloatTensor = can_slerp.expand(*t.shape, -1) if isinstance(t, Tensor) else can_slerp.unsqueeze(-1)
    out: FloatTensor = where(index, slerped, lerped)
    
    return out