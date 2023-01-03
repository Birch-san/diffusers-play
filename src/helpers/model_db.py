from dataclasses import dataclass
from helpers.embed_text import ClipCheckpoint
import torch
from typing import Dict

sd2_768_models = { 'stabilityai/stable-diffusion-2', 'stabilityai/stable-diffusion-2-1' }
sd2_base_models = { 'stabilityai/stable-diffusion-2-base', 'stabilityai/stable-diffusion-2-1-base' }
sd2_models = { *sd2_768_models, *sd2_base_models }
# models where if you use 16-bit computation in the Unet: you will get NaN latents. prefer to run attention in 32-bit
upcast_attention_models = { 'stabilityai/stable-diffusion-2-1' }

laion_embed_models = { *sd2_models }
_768_models = { *sd2_768_models }
vparam_models = { *sd2_768_models }
penultimate_clip_hidden_state_models = { *sd2_models }

xattn_max_context_segments: Dict[str, int] = {
  'hakurei/waifu-diffusion': 3
}

def get_clip_ckpt(model_name: str) -> ClipCheckpoint:
  if model_name == 'hakurei/waifu-diffusion':
    return ClipCheckpoint.Waifu
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