from transformers import CLIPTokenizer, PreTrainedTokenizer
from typing import List, Tuple
# tokenizer: PreTrainedTokenizer = CLIPTokenizer.from_pretrained('waifu-diffusion/wd-1-5-beta', subfolder='tokenizer')
tokenizer: PreTrainedTokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
def to(prompt: str) -> List[Tuple[int, str]]:
  tokens: List[int] = tokenizer.convert_ids_to_tokens(tokenizer(prompt).input_ids)
  return list(enumerate(tokens))
t = to('realistic, real life, waifu, instagram, anime, watercolor (medium), traditional media'); t
t = to('flandre scarlet, carnelian, 1girl, blonde hair, blush, light smile, collared shirt, hair between eyes, hat bow, looking at viewer, medium hair, mob cap, upper body, puffy short sleeves, red bow, watercolor (medium), traditional media, red eyes, red vest, small breasts, upper body, white shirt, yellow ascot'); t

# somewhere to put a breakpoint!
pass