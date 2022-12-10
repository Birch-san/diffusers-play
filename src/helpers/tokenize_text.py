import torch
from typing import List, Protocol
from torch import LongTensor
from .device import DeviceType
from .clip_identifiers import ClipImplementation, ClipCheckpoint
from .prompt_type import Prompts

class CountTokens(Protocol):
  def count_tokens(prompts: Prompts, device: DeviceType=torch.device('cpu')) -> LongTensor: ...

def get_hf_tokenizer(ckpt: ClipCheckpoint):
  from transformers import PreTrainedTokenizer, CLIPTokenizer
  match(ckpt):
    case ClipCheckpoint.OpenAI:
      model_name = 'openai/clip-vit-large-patch14'
      extra_args = {}
    case ClipCheckpoint.LAION:
      # model_name = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
      model_name = 'stabilityai/stable-diffusion-2'
      extra_args = {'subfolder': 'tokenizer'}
    case _:
      raise "never heard of '{ckpt}' ClipCheckpoint."
  tokenizer: PreTrainedTokenizer = CLIPTokenizer.from_pretrained(model_name, **extra_args)
  return tokenizer

def get_token_counter(
  impl: ClipImplementation,
  ckpt: ClipCheckpoint,
) -> CountTokens:
  """Counts tokens, does not truncate, does not pad"""
  match(impl):
    case ClipImplementation.HF:
      from transformers import PreTrainedTokenizer
      from transformers.tokenization_utils_base import BatchEncoding
      from transformers.utils.generic import PaddingStrategy
      tokenizer: PreTrainedTokenizer = get_hf_tokenizer(ckpt=ckpt)
      def tokenize(prompts: Prompts, device: DeviceType=torch.device('cpu')) -> LongTensor:
        tokens: BatchEncoding = tokenizer(prompts, truncation=False, padding=PaddingStrategy.DO_NOT_PAD, max_length=None, add_special_tokens=False, return_tensors="pt", return_length=True)
        token_counts: LongTensor = tokens.length.to(device).reshape(-1, 1)
        return token_counts
      return tokenize
    case ClipImplementation.OpenCLIP:
      from open_clip.tokenizer import _tokenizer
      def count_tokens(prompts: Prompts, device: DeviceType=torch.device('cpu')) -> LongTensor:
        if isinstance(prompts, str):
          prompts: List[str] = [prompts]
        tokens: List[List[int]] = _tokenizer.encode(prompts)
        token_counts: List[int] = [len(tokens_) for tokens_ in tokens]
        return torch.tensor(token_counts, dtype=torch.long, device=device).reshape(-1, 1)
      return count_tokens
    case _:
      raise f"never heard of a '{impl}' ClipImplementation."