from PIL import Image
from torch import FloatTensor
import numpy as np
import torch

def load_img(path) -> FloatTensor:
  image: Image.Image = Image.open(path).convert("RGB")
  w, h = image.size
  print(f"loaded input image of size ({w}, {h}) from {path}")
  img_arr: np.ndarray = np.array(image)
  del image
  img_tensor: FloatTensor = torch.from_numpy(img_arr).to(dtype=torch.float32)
  del img_arr
  img_tensor: FloatTensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
  img_tensor: FloatTensor = img_tensor / 127.5 - 1.0
  return img_tensor