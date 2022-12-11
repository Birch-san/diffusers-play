import torch
from torch import Tensor, no_grad

from helpers.device import DeviceLiteral, get_device_type
from helpers.clip_identifiers import ClipCheckpoint, ClipImplementation
from helpers.embed_text import Embed, get_embedder
from helpers.tokenize_text import CountTokens, get_token_counter
from helpers.structured_diffusion import get_structured_embedder, StructuredEmbed, StructuredEmbedding
from typing import List

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)
torch_dtype=torch.float32
cfg_enabled=True

model_name = (
  # 'CompVis/stable-diffusion-v1-4'
  # 'hakurei/waifu-diffusion'
  'runwayml/stable-diffusion-v1-5'
  # 'stabilityai/stable-diffusion-2'
  # 'stabilityai/stable-diffusion-2-1'
  # 'stabilityai/stable-diffusion-2-base'
  # 'stabilityai/stable-diffusion-2-1-base'
)

sd2_768_models = { 'stabilityai/stable-diffusion-2', 'stabilityai/stable-diffusion-2-1' }
sd2_base_models = { 'stabilityai/stable-diffusion-2-base', 'stabilityai/stable-diffusion-2-1-base' }
sd2_models = { *sd2_768_models, *sd2_base_models }

laion_embed_models = { *sd2_models }
penultimate_clip_hidden_state_models = { *sd2_models }

needs_laion_embed = model_name in laion_embed_models
needs_penultimate_clip_hidden_state = model_name in penultimate_clip_hidden_state_models

clip_impl = ClipImplementation.HF
clip_ckpt = ClipCheckpoint.LAION if needs_laion_embed else ClipCheckpoint.OpenAI
clip_subtract_hidden_state_layers = 1 if needs_penultimate_clip_hidden_state else 0
embed: Embed = get_embedder(
  impl=clip_impl,
  ckpt=clip_ckpt,
  subtract_hidden_state_layers=clip_subtract_hidden_state_layers,
  device=device,
  torch_dtype=torch_dtype
)
count_tokens: CountTokens = get_token_counter(
  impl=clip_impl,
  ckpt=clip_ckpt,
)
sembed: StructuredEmbed = get_structured_embedder(
  embed=embed,
  count_tokens=count_tokens,
)

prompt = 'two blue sheep with a red car'
prompts: List[str] = [prompt]
with no_grad():
  structured_embedding: StructuredEmbedding = sembed(prompts, gimme_uncond=cfg_enabled)
  uc = structured_embedding.uncond
  c = structured_embedding.embeds