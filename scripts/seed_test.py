from helpers.inference_spec.batch_spec_factory import LatentsGenerator, latents_from_seed_factory, MakeLatents, LatentsShape
from helpers.get_seed import get_seed
from helpers.device import DeviceLiteral, get_device_type
import torch

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

height = 768
width = 640**2//height

unet_channels = 4
latent_scale_factor = 8

latents_shape = LatentsShape(unet_channels, height // latent_scale_factor, width // latent_scale_factor)
make_latents: MakeLatents[int] = latents_from_seed_factory(latents_shape, dtype=torch.float32, device=device)

generator = LatentsGenerator(
  batch_size=3,
  specs=[
    *(1,)*3,
    2,
    3,
    4,
    *[get_seed() for _ in range(2)]
  ],
  make_latents=make_latents,
).generate()
latents = list(generator)
print(latents)