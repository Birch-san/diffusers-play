from transformers import CLIPTokenizer, PreTrainedTokenizer
from typing import List, Tuple
# tokenizer: PreTrainedTokenizer = CLIPTokenizer.from_pretrained('waifu-diffusion/wd-1-5-beta', subfolder='tokenizer')
tokenizer: PreTrainedTokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
def to(prompt: str) -> List[Tuple[int, str]]:
  tokens: List[int] = tokenizer.convert_ids_to_tokens(tokenizer(prompt).input_ids)
  return list(enumerate(tokens))
t = to('realistic, real life, waifu, instagram, anime, watercolor (medium), traditional media'); t

# somewhere to put a breakpoint!
pass