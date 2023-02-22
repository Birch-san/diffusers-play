from torch import FloatTensor, lerp
from torch.linalg import norm

# adapted to PyTorch from:
# https://gist.github.com/dvschultz/3af50c40df002da3b751efab1daddf2c
def slerp(t: float, v0: FloatTensor, v1: FloatTensor, DOT_THRESHOLD=0.9995):
    '''
    Spherical linear interpolation
    Args:
        t: Float value between 0.0 and 1.0
        v0: Starting vector
        v1: Final vector
        DOT_THRESHOLD: Threshold for considering the two vectors as
                               colinear. Not recommended to alter this.
    Returns:
        Interpolation vector between v0 and v1
    '''
    # Normalize the vectors to get the directions and angles
    v0_norm: FloatTensor = norm(v0)
    if not v0_norm.is_nonzero():
        # normalizing v0 would make it infinite, making its dot product with v1 infinite,
        # making them ~colinear
        return lerp(v0, v1, t)
    v1_norm: FloatTensor = norm(v1)
    if not v1_norm.is_nonzero():
        return lerp(v0, v1, t)
    v0_normed: FloatTensor = v0 / v0_norm
    v1_normed: FloatTensor = v1 / v1_norm
    # Dot product with the normalized vectors
    dot: FloatTensor = v0_normed @ v1_normed
    # If absolute value of dot product is almost 1, vectors are ~colinear, so use lerp
    if dot.abs() > DOT_THRESHOLD:
        return lerp(v0, v1, t)
    # Calculate initial angle between v0 and v1
    theta_0: FloatTensor = dot.arccos()
    sin_theta_0: FloatTensor = theta_0.sin()
    # Angle at timestep t
    theta_t: FloatTensor = theta_0 * t
    sin_theta_t: FloatTensor = theta_t.sin()
    # Finish the slerp algorithm
    s0: FloatTensor = (theta_0 - theta_t).sin() / sin_theta_0
    s1: FloatTensor = sin_theta_t / sin_theta_0
    v2: FloatTensor = s0 * v0 + s1 * v1
    return v2