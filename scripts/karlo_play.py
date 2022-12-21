from os import makedirs, listdir, path
import torch
from diffusers import UnCLIPPipeline

pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha", torch_dtype=torch.float16)
pipe = pipe.to("mps")

# prompt = "manga illustration of Artoria Pendragon, Fate/Stay Night, light smile, copic marker, pixiv."
prompt = "a high-resolution photograph of a big red frog on a green leaf."
image = pipe(prompt).images[0]

sample_path='out'
for path_ in [sample_path]:
  makedirs(path_, exist_ok=True)

base_count = len(listdir(sample_path))
image.save(path.join(sample_path, f"{base_count:05}.png"))