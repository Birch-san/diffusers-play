import torch
from torch.optim import Adam
from itertools import chain

from lora_diffusion import inject_trainable_lora, extract_lora_up_downs
from diffusers.models import UNet2DConditionModel

from helpers.device import DeviceLiteral, get_device_type

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

model_name = (
  # 'CompVis/stable-diffusion-v1-4'
  'hakurei/waifu-diffusion'
  # 'runwayml/stable-diffusion-v1-5'
  # 'stabilityai/stable-diffusion-2'
  # 'stabilityai/stable-diffusion-2-1'
  # 'stabilityai/stable-diffusion-2-base'
  # 'stabilityai/stable-diffusion-2-1-base'
)

unet: UNet2DConditionModel = UNet2DConditionModel.from_pretrained(
  model_name,
  subfolder='unet',
).to(device)
unet.requires_grad_(False)
unet_lora_params, train_names = inject_trainable_lora(unet)



optimizer = Adam(
  chain(*unet_lora_params, text_encoder.parameters()), lr=1e-4
)