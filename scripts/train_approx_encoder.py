from enum import Enum, auto
import fnmatch
import torch
from os import listdir, makedirs
from os.path import join, exists, dirname
from torch import IntTensor, FloatTensor, inference_mode, load, save
from torch.optim import AdamW
from torchvision.io import write_png
from helpers.device import get_device_type, DeviceLiteral
from helpers.approx_vae.encoder import Encoder
from helpers.approx_vae.dataset import get_data, Dataset
from helpers.approx_vae.get_file_names import GetFileNames
from helpers.approx_vae.get_latents import get_latents
from helpers.approx_vae.resize_samples import get_resized_samples
from helpers.approx_vae.int_info import int8_half_range
from helpers.approx_vae.loss import loss_fn, describe_loss, LossComponents
from helpers.approx_vae.visualize_latents import normalize_latents, norm_latents_to_rgb, collage_2by2
from helpers.latents_to_pils import LatentsToPils, LatentsToBCHW, make_latents_to_pils, make_latents_to_bchw
from diffusers.models import AutoencoderKL
from typing import List
from PIL import Image

device_type: DeviceLiteral = get_device_type()
device = torch.device(device_type)

model_shortname = 'sd1.5'
repo_root=join(dirname(__file__), '..')
assets_dir = f'out_learn_{model_shortname}'
samples_dir=join(assets_dir, 'samples')
processed_train_data_dir=join(assets_dir, 'processed_train_data')
processed_test_data_dir=join(assets_dir, 'processed_test_data')
processed_test_visualization_dir=join(processed_test_data_dir, 'latent_vis')
latents_dir=join(assets_dir, 'pt')
test_latents_dir=join(assets_dir, 'test_pt')
test_samples_dir=join(assets_dir, 'test_png')
predictions_root_dir=join(assets_dir, 'test_pred_encoder')
predictions_latents_dir=join(predictions_root_dir, 'pt')
predictions_samples_dir=join(predictions_root_dir, 'png')
science_dir=join(assets_dir, 'science')
science2_dir=join(assets_dir, 'science2')
weights_dir=join(repo_root, 'approx_vae')
for path_ in [processed_train_data_dir, processed_test_data_dir, processed_test_visualization_dir, predictions_root_dir, predictions_latents_dir, predictions_samples_dir, science_dir, science2_dir]:
  makedirs(path_, exist_ok=True)

weights_path = join(weights_dir, f'encoder_{model_shortname}.pt')

class Mode(Enum):
  Train = auto()
  Test = auto()
  Science = auto()
  Science2 = auto()
mode = Mode.Train
test_after_train=True
resume_training=False

model = Encoder()
if exists(weights_path) and resume_training or mode is not Mode.Train:
  model.load_state_dict(load(weights_path, weights_only=True))
model = model.to(device)

training_dtype = torch.float32

# epochs = 3000
# epochs = 1000
epochs = 400

optim = AdamW(model.parameters(), lr=5e-2)

def train(epoch: int, dataset: Dataset):
  model.train()
  out: FloatTensor = model(dataset.samples)
  loss_components: LossComponents = loss_fn(out, dataset.latents)
  loss, _ = loss_components
  optim.zero_grad()
  loss.backward()
  optim.step()
  if epoch % 100 == 0:
    loss_desc: str = describe_loss(loss_components)
    print(f'epoch {epoch:04d}, {loss_desc}')

@inference_mode(True)
def test():
  get_sample_filenames: GetFileNames = lambda: sorted(fnmatch.filter(listdir(test_samples_dir), f"*.png"), key=lambda fname: int(fname.split('.', 1)[0]))
  get_latent_filenames: GetFileNames = lambda: [sample_path.replace('png', 'pt') for sample_path in get_sample_filenames()]

  samples: IntTensor = get_resized_samples(
    in_dir=test_samples_dir,
    out_dir=processed_test_data_dir,
    get_sample_filenames=get_sample_filenames,
    device=device,
  )
  samples: FloatTensor = samples.to(training_dtype)
  samples = samples-int8_half_range
  samples = samples/int8_half_range

  true_latents: FloatTensor = get_latents(
    in_dir=test_latents_dir,
    out_dir=processed_test_data_dir,
    get_latent_filenames=get_latent_filenames,
    device=device
  )
  true_latents = true_latents.to(training_dtype)

  # save visualizations of true latents
  true_norm: FloatTensor = normalize_latents(true_latents)
  true_rgb: IntTensor = norm_latents_to_rgb(true_norm)
  true_collages: IntTensor = collage_2by2(true_rgb, keepdim=False).cpu()
  for filename, collage in zip(get_latent_filenames(), true_collages.split(1)):
    write_png(collage, join(processed_test_visualization_dir, filename.replace('.pt', '.png')))

  model.eval() # might be redundant due to inference mode but whatever

  predicted_latents: FloatTensor = model.forward(samples)
  # back to channels-first for comparison with true latents
  predicted_latents = predicted_latents.permute(0, 3, 1, 2)

  # save latent predictions, and visualizations thereof
  pred_norm: FloatTensor = normalize_latents(predicted_latents)
  pred_rgb: IntTensor = norm_latents_to_rgb(pred_norm)
  pred_collages: IntTensor = collage_2by2(pred_rgb, keepdim=False).cpu()
  for filename, prediction, collage in zip(get_latent_filenames(), predicted_latents.split(1), pred_collages.split(1)):
    save(prediction, join(predictions_latents_dir, filename))
    write_png(collage, join(predictions_samples_dir, filename.replace('.pt', '.png')))
  
  loss_components = loss_fn(predicted_latents, true_latents)
  loss_desc: str = describe_loss(loss_components)
  print(f'validation {loss_desc}')

match(mode):
  case Mode.Train:
    dataset: Dataset = get_data(
      latents_dir=latents_dir,
      processed_train_data_dir=processed_train_data_dir,
      samples_dir=samples_dir,
      dtype=training_dtype,
      device=device,
    )
    for epoch in range(epochs):
      train(epoch, dataset)
    del dataset
    save(model.state_dict(), weights_path)
    if test_after_train:
      test()
  case Mode.Test:
    test()
  case Mode.Science:
    latent_filename = '00242.1222668698.cfg07.50.sd1.5.pt'
    input_path = join(test_latents_dir, latent_filename)
    latents: FloatTensor = load(input_path, map_location=device, weights_only=True)
    norm: FloatTensor = normalize_latents(latents)
    rgb: IntTensor = norm_latents_to_rgb(norm)
    for ix, channel in enumerate(rgb.split(1)):
      write_png(channel.cpu(), join(science_dir, latent_filename.replace('.pt', f'.{ix}.png')))
    collage: IntTensor = collage_2by2(rgb, keepdim=True)
    write_png(collage.cpu(), join(science_dir, latent_filename.replace('.pt', '.png')))
  case Mode.Science2:
    # see what happens if we try to decode our approx latents using a real VAE
    from torchvision.io import read_image
    filenames = ['rooster.png', 'birb.png', '00234.3620773285.cfg07.50.sd1.5.from_pred_latents.png', '00234.3620773285.cfg07.50.sd1.5.from_true_latents.png']
    rooster: IntTensor = read_image('/home/birch/rooster.png').to(device=device, dtype=training_dtype)
    birb: IntTensor = read_image('/home/birch/birb.png').to(device=device, dtype=training_dtype)
    test_miko: IntTensor = read_image(join(processed_test_data_dir, '00234.3620773285.cfg07.50.sd1.5.png')).to(device=device, dtype=training_dtype)
    samples: IntTensor = torch.stack([rooster, birb, test_miko])
    samples: FloatTensor = samples.to(training_dtype)
    samples = samples-int8_half_range
    samples = samples/int8_half_range

    model.eval()

    # channels-last
    samples = samples.permute(0, 2, 3, 1)
    predicted_latents: FloatTensor = model.forward(samples)
    # back to channels-first for comparison with true latents
    predicted_latents = predicted_latents.permute(0, 3, 1, 2)

    true_girl: FloatTensor = load(join(test_latents_dir, '00234.3620773285.cfg07.50.sd1.5.pt'), map_location=device, weights_only=True)

    latents: FloatTensor = torch.cat([predicted_latents, true_girl.unsqueeze(0)], dim=0)

    vae: AutoencoderKL = AutoencoderKL.from_pretrained(
      'runwayml/stable-diffusion-v1-5',
      subfolder='vae',
      revision='fp16',
      torch_dtype=torch.float16,
    ).to(device).eval()
    latents_to_bchw: LatentsToBCHW = make_latents_to_bchw(vae)
    latents_to_pils: LatentsToPils = make_latents_to_pils(latents_to_bchw)
    pil_images: List[Image.Image] = latents_to_pils(latents)

    for filename, image in zip(filenames, pil_images):
      image.save(join(science2_dir, filename))
