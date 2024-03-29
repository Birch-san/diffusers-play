from dataclasses import dataclass
from .embed_text import ClipCheckpoint
from .approx_vae.decoder_ckpt import DecoderCkpt
from .approx_vae.encoder_ckpt import EncoderCkpt
import torch
from typing import Dict

# models where if you use 16-bit computation in the Unet: you will get NaN latents. prefer to run attention in 32-bit
upcast_attention_models = { 'stabilityai/stable-diffusion-2-1' }
laion_embed_models = { 'stabilityai/stable-diffusion-2', 'stabilityai/stable-diffusion-2-1', 'stabilityai/stable-diffusion-2-base', 'stabilityai/stable-diffusion-2-1-base' }
_768_models = { 'stabilityai/stable-diffusion-2', 'stabilityai/stable-diffusion-2-1' }
vparam_models = { 'stabilityai/stable-diffusion-2', 'stabilityai/stable-diffusion-2-1', 'waifu-diffusion/wd-1-5-beta', 'waifu-diffusion/wd-1-5-beta2', 'waifu-diffusion/wd-1-5-beta3' }
penultimate_clip_hidden_state_models = { 'stabilityai/stable-diffusion-2', 'stabilityai/stable-diffusion-2-1', 'stabilityai/stable-diffusion-2-base', 'stabilityai/stable-diffusion-2-1-base', 'hakurei/waifu-diffusion', 'waifu-diffusion/wd-1-5-beta', 'waifu-diffusion/wd-1-5-beta2', 'waifu-diffusion/wd-1-5-beta3' }

xattn_max_context_segments: Dict[str, int] = {
  # you can try using more than this, but it was only trained on up to 3
  'hakurei/waifu-diffusion': 3,
  # not actually known how many context segments they used
  'waifu-diffusion/wd-1-5-beta': 3,
  'waifu-diffusion/wd-1-5-beta2': 3,
  'waifu-diffusion/wd-1-5-beta3': 3
}

model_shortname: Dict[str, str] = {
  'CompVis/stable-diffusion-v1-3': 'sd1.3',
  'CompVis/stable-diffusion-v1-4': 'sd1.4',
  'runwayml/stable-diffusion-v1-5': 'sd1.5',
}

def get_clip_ckpt(model_name: str) -> ClipCheckpoint|str:
  match model_name:
    case 'hakurei/waifu-diffusion' | 'waifu-diffusion/wd-1-5-beta' | 'waifu-diffusion/wd-1-5-beta2':
      return model_name
    case 'waifu-diffusion/wd-1-5-beta3':
      return '/home/birch/ml-data/wd1-5-b3'
  if model_name in laion_embed_models:
    return ClipCheckpoint.LAION
  return ClipCheckpoint.OpenAI

def get_is_768(model_name: str) -> bool:
  return model_name in _768_models

def get_needs_vparam(model_name: str) -> bool:
  return model_name in vparam_models

def get_needs_penultimate_clip_hidden_state(model_name: str) -> bool:
  return model_name in penultimate_clip_hidden_state_models

def get_needs_upcast_attention(model_name: str, unet_dtype: torch.dtype) -> bool:
  return unet_dtype is torch.float16 and model_name in upcast_attention_models

def get_xattn_max_context_segments(model_name: str) -> int:
  return xattn_max_context_segments[model_name] if model_name in xattn_max_context_segments else 1

def get_approx_decoder_ckpt(model_name: str, wd_prefer_1_3: bool) -> DecoderCkpt:
  match model_name:
    case 'CompVis/stable-diffusion-v1-4':
      return DecoderCkpt.SD1_4
    case 'hakurei/waifu-diffusion':
      return DecoderCkpt.WD1_3 if wd_prefer_1_3 else DecoderCkpt.WD1_4
    case 'waifu-diffusion/wd-1-5-beta2' | 'waifu-diffusion/wd-1-5-beta3':
      return DecoderCkpt.WD1_5
    case 'runwayml/stable-diffusion-v1-5' | _:
      return DecoderCkpt.SD1_5

def get_approx_encoder_ckpt(model_name: str, wd_prefer_1_3: bool) -> EncoderCkpt:
  match model_name:
    case 'CompVis/stable-diffusion-v1-4':
      return EncoderCkpt.SD1_4
    case 'hakurei/waifu-diffusion':
      return EncoderCkpt.WD1_3 if wd_prefer_1_3 else EncoderCkpt.WD1_4
    case 'waifu-diffusion/wd-1-5-beta2' | 'waifu-diffusion/wd-1-5-beta3':
      return EncoderCkpt.WD1_5
    case 'runwayml/stable-diffusion-v1-5' | _:
      return EncoderCkpt.SD1_5

@dataclass
class ModelNeeds:
  clip_ckpt: ClipCheckpoint
  is_768: bool
  needs_vparam: bool
  needs_penultimate_clip_hidden_state: bool
  needs_upcast_attention: bool
  xattn_max_context_segments: int

def get_model_needs(model_name: str, unet_dtype: torch.dtype) -> ModelNeeds:
  return ModelNeeds(
    clip_ckpt=get_clip_ckpt(model_name),
    is_768=get_is_768(model_name),
    needs_vparam=get_needs_vparam(model_name),
    needs_penultimate_clip_hidden_state=get_needs_penultimate_clip_hidden_state(model_name),
    needs_upcast_attention=get_needs_upcast_attention(model_name, unet_dtype),
    xattn_max_context_segments=get_xattn_max_context_segments(model_name),
  )