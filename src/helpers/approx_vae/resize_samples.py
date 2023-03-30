from cv2 import Mat, cvtColor
import cv2
from os.path import join
from torch import load, save, from_numpy, stack, IntTensor
from .get_file_names import GetFileNames
from typing import Tuple, List
import torch

debug_resize=False

# there's so few of these we may as well keep them all resident in memory
def get_resized_samples(
  in_dir: str,
  out_dir: str,
  get_sample_filenames: GetFileNames,
  device: torch.device = torch.device('cpu')
) -> IntTensor:
  filename_stem = 'resized_samples'

  resized_path_png = join(out_dir, f'{filename_stem}.png')
  resized_path_pt = join(out_dir, f'{filename_stem}.pt')
  try:
    resized: IntTensor = load(resized_path_pt, map_location=device, weights_only=True)
    return resized
  except:
    # load/resize via cv2 instead of torchvision because cv2 has INTER_AREA downsample for best PSNR.
    # CUDA-accelerating this requires building opencv from source. instead we'll cache the result on-disk.
    sample_filenames: List[str] = get_sample_filenames()
    mats: List[Mat] = [cv2.imread(join(in_dir, sample_filename)) for sample_filename in sample_filenames]
    first, *_ = mats
    height, width, _ = first.shape
    vae_scale_factor = 8
    scaled_height = height//vae_scale_factor
    scaled_width = width//vae_scale_factor
    dsize: Tuple[int, int] = (scaled_width, scaled_height)
    resizeds: List[Mat] = [cv2.resize(mat, dsize, interpolation=cv2.INTER_AREA) for mat in mats]
    for img, filename in zip(resizeds, sample_filenames):
      cv2.imwrite(join(out_dir, filename), img)
    tensors: List[IntTensor] = [from_numpy(cvtColor(resized, cv2.COLOR_BGR2RGB)) for resized in resizeds]
    tensor: IntTensor = stack(tensors)
    save(tensor, resized_path_pt)
    if debug_resize:
      from torchvision.io import write_png
      write_png(tensor.permute(3, 0, 1, 2).flatten(1, end_dim=2), resized_path_png)
    return tensor.to(device)