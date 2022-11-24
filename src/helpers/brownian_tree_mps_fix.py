from torch import randn_like

# monkey-patch _randn to use CPU random before k-diffusion uses it
from torchsde._brownian.brownian_interval import _randn
from torchsde._brownian import brownian_interval
brownian_interval._randn = lambda size, dtype, device, seed: (
  _randn(size, dtype, 'cpu' if device.type == 'mps' else device, seed).to(device)
)

from k_diffusion import sampling
sampling.default_noise_sampler = lambda x: (
  lambda sigma, sigma_next: randn_like(x, device='cpu' if x.device.type == 'mps' else x.device).to(x.device)
)

reassuring_message = "monkey-patched BrownianTree noise sampler's _randn to use CPU random, for compatibility on MPS backend"