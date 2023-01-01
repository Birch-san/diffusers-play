import torch
from torch import Tensor, LongTensor, BoolTensor, no_grad
from enum import Enum, auto
from .log_level import log_level
from .device import DeviceType
from .embed_text_types import Prompts, Embed, EmbeddingAndMask
from .clip_embed_text import get_embedder as get_clip_embedder

class ClipImplementation(Enum):
  HF = auto()
  OpenCLIP = auto()
  # OpenAI CLIP and clip-anytorch not implemented

class ClipCheckpoint(Enum):
  OpenAI = auto()
  LAION = auto()
  Waifu = auto()

def get_embedder(
  impl: ClipImplementation,
  ckpt: ClipCheckpoint,
  subtract_hidden_state_layers = 0,
  device: DeviceType = 'cpu',
  torch_dtype: torch.dtype = torch.float32
) -> Embed:
  match(impl):
    case ClipImplementation.HF:
      from transformers import CLIPTextModel, PreTrainedTokenizer, CLIPTokenizer, logging
      match(ckpt):
        case ClipCheckpoint.OpenAI:
          model_name = 'openai/clip-vit-large-patch14'
          tokenizer_extra_args = {}
          encoder_extra_args = {}
        case ClipCheckpoint.LAION:
          # model_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
          model_name = 'stabilityai/stable-diffusion-2'
          tokenizer_extra_args = {'subfolder': 'tokenizer'}
          encoder_extra_args = {'subfolder': 'text_encoder'}
        case ClipCheckpoint.Waifu:
          model_name = 'hakurei/waifu-diffusion'
          tokenizer_extra_args = {'subfolder': 'tokenizer'}
          encoder_extra_args = {'subfolder': 'text_encoder'}
        case _:
          raise "never heard of '{ckpt}' ClipCheckpoint."
      tokenizer: PreTrainedTokenizer = CLIPTokenizer.from_pretrained(model_name, **tokenizer_extra_args)
      with log_level(logging.ERROR):
        text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(model_name, torch_dtype=torch_dtype, **encoder_extra_args).to(device).eval()
      
      embed: Embed = get_clip_embedder(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        subtract_hidden_state_layers=subtract_hidden_state_layers,
      )
      return embed
    case ClipImplementation.OpenCLIP:
      # this doesn't seem to get same result as HF implementation, but it does give a result that matches the prompt. maybe I goofed somewhere.
      # I adapted it from:
      # https://github.com/Stability-AI/stablediffusion/blob/a436738fc31da2def74dad426a02cdc9b6f009d0/ldm/modules/encoders/modules.py#L147
      import open_clip
      from open_clip import CLIP as OpenCLIP, tokenize
      from open_clip.tokenizer import _tokenizer
      match(ckpt):
        case ClipCheckpoint.OpenAI:
          model_name = 'ViT-L-14'
          pretrained = 'openai'
        case ClipCheckpoint.LAION:
          model_name = 'ViT-H-14'
          pretrained = 'laion2b_s32b_b79k'
        case ClipCheckpoint.Waifu:
          raise 'OpenCLIP support not yet implemented for waifu-diffusion checkpoint'
        case _:
          raise "never heard of '{ckpt}' ClipCheckpoint."
      encoder, _, _ = open_clip.create_model_and_transforms(model_name, device=device, pretrained=pretrained)
      encoder: OpenCLIP = encoder.eval()
      # TODO: source this in a dynamic way instead of hardcoding
      context_length = 77
      def make_attention_mask(prompts: Prompts) -> BoolTensor:
        keep_count = torch.tensor(
          [
            min(
              len(_tokenizer.encode(p)) + 2,
              context_length
            ) for p in prompts
          ],
          dtype=torch.long,
          device=device,
        )
        token_ix = torch.arange(0, context_length, dtype=torch.long, device=device)
        return token_ix.expand(2, -1) < keep_count.unsqueeze(0).transpose(0, 1)
      def text_transformer_forward(x: Tensor, attn_mask = None) -> Tensor:
        for r in encoder.transformer.resblocks[:len(encoder.transformer.resblocks) - subtract_hidden_state_layers]:
          x = r(x, attn_mask=attn_mask)
        return x
      def postprocess(x: Tensor) -> Tensor:
        x = x + encoder.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = text_transformer_forward(x, attn_mask=encoder.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = encoder.ln_final(x)
        return x
      def embed(prompts: Prompts) -> EmbeddingAndMask:
        mask: BoolTensor = make_attention_mask(prompts)
        tokens: LongTensor = tokenize(prompts).to(device)
        with no_grad():
          text_embeddings: Tensor = encoder.token_embedding(tokens)
          text_embeddings: Tensor = postprocess(text_embeddings)
        return text_embeddings, mask
      return embed
    case _:
      raise f"never heard of a '{impl}' ClipImplementation."


