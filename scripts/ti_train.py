import argparse
from dataclasses import dataclass
import itertools
import math
import os
import random
from pathlib import Path
from typing import Optional, Dict, NamedTuple, List, Callable
from argparse import Namespace
from random import sample, random

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor, Generator, randn
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from helpers.cumsum_mps_fix import reassuring_message
print(reassuring_message) # avoid "unused" import :P

import PIL
from PIL.Image import Image as Img
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami

from k_diffusion.sampling import get_sigmas_karras, sample_dpmpp_2m
from helpers.schedules import KarrasScheduleParams, KarrasScheduleTemplate, get_template_schedule
from helpers.schedule_params import get_alphas, get_alphas_cumprod, get_betas, get_sigmas, get_log_sigmas, log_sigmas_to_t
from helpers.model_db import get_model_needs, ModelNeeds
from helpers.embed_text_types import Embed
from helpers.clip_embed_text import get_embedder
from helpers.diffusers_denoiser import DiffusersSD2Denoiser, DiffusersSDDenoiser
from helpers.cfg_denoiser import Denoiser, DenoiserFactory
from helpers.latents_to_pils import LatentsToBCHW, make_latents_to_bchw
from helpers.get_seed import get_seed

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import hflip
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer, PreTrainedTokenizer


if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")


logger = get_logger(__name__)


class AddedTokens(NamedTuple):
    placeholder_token: str
    placeholder_token_ids: List[int]

def add_tokens_and_get_placeholder_token(
    args: Namespace,
    token_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    text_encoder: CLIPTextModel
) -> AddedTokens:
    assert args.num_vec_per_token >= len(token_ids)
    placeholder_tokens: List[str] = [f"{args.placeholder_token}_{i}" for i in range(args.num_vec_per_token)]

    for placeholder_token in placeholder_tokens:
        num_added_tokens = tokenizer.add_tokens(placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )
    placeholder_token = " ".join(placeholder_tokens)
    placeholder_token_ids: List[int] = tokenizer.encode(placeholder_token, add_special_tokens=False)
    print(f"The placeholder tokens are: {placeholder_token} while the ids are {placeholder_token_ids}")
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    if args.initialize_rest_random:
        # The idea is that the placeholder tokens form adjectives as in x x x white dog.
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            if len(placeholder_token_ids) - i < len(token_ids):
                token_embeds[placeholder_token_id] = token_embeds[token_ids[i % len(token_ids)]]
            else:
                token_embeds[placeholder_token_id] = torch.rand_like(token_embeds[placeholder_token_id])
    else:
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = token_embeds[token_ids[i % len(token_ids)]]
    return AddedTokens(placeholder_token, placeholder_token_ids)


def save_progress(text_encoder, placeholder_tokens, placeholder_token_ids, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_ids]
    learned_embeds_dict: Dict[str, Tensor] = {
        placeholder_token: learned_embed.detach().cpu() for placeholder_token, learned_embed in zip(placeholder_tokens.split(" "), learned_embeds)
    }
    torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--num_vec_per_token",
        type=int,
        default=1,
        help=(
            "The number of vectors used to represent the placeholder token. The higher the number, the better the"
            " result at the cost of editability. This can be fixed by prompt editing."
        ),
    )
    parser.add_argument(
        "--initialize_rest_random", action="store_true", help="Initialize rest of the placeholder tokens with random."
    )
    parser.add_argument(
        "--cache_images", action="store_true", help="Cache tensors of every image we load. You should only do this if your training set is small."
    )
    parser.add_argument(
        "--visualization_steps",
        type=int,
        default=100,
        help="Log image every X updates steps.",
    )
    parser.add_argument(
        "--visualization_train_samples",
        type=int,
        default=2,
        help="How many samples (using prompts from training set) to output when visualizing.",
    )
    parser.add_argument(
        "--visualization_test_samples",
        type=int,
        default=2,
        help="How many samples (using unseen prompts) to output when visualizing.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=True,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

@dataclass
class Variations:
    original: Tensor
    flipped: Tensor

class TextualInversionDataset(Dataset):
    cache: Dict[str, Variations]
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="lanczos",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        cache_enabled=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [
            os.path.join(self.data_root, file_path)
            for file_path in os.listdir(self.data_root) if file_path.endswith('.png') or file_path.endswith('.jpg')
        ]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        # we have so few images and so much VRAM that we should prefer to retain tensors rather than redo work
        self.cache = {}
        self.cache_enabled = cache_enabled

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path: str = self.image_paths[i % self.num_images]
        stem: str = Path(image_path).stem

        placeholder_string = self.placeholder_token
        # text = random.choice(self.templates).format(placeholder_string)
        def describe_placeholder() -> str:
            if random() < 0.3:
                return self.placeholder_token
            return placeholder_string

        def describe_subject(character: str) -> str:
            placeholder: str = describe_placeholder()
            if random() < 0.3:
                return f"photo of {placeholder}"
            return f"photo of {character} {placeholder}"

        def make_prompt(character: str, general_labels: List[str], sitting=True, on_floor=True) -> str:
            even_more_labels = [*general_labels, '1girl']
            if sitting:
                even_more_labels.append('sitting')
            if on_floor:
                even_more_labels.append('on floor')
            subject: str = describe_subject(character)
            # we can use this for dropout but I think dropout is undesirable
            # label_count = randrange(0, len(even_more_labels))
            label_count = len(even_more_labels)
            if label_count == 0:
                return subject
            labels = sample(even_more_labels, label_count)
            joined = ', '.join(labels)
            return f"{subject} with {joined}"

        match stem:
            case 'koishi':
                text = make_prompt('komeiji koishi', ['green hair', 'black footwear', 'medium hair', 'blue eyes', 'yellow jacket', 'green skirt' 'hat', 'black headwear', 'smile', 'touhou project'])
            case 'flandre':
                text = make_prompt('flandre scarlet', ['fang', 'red footwear', 'slit pupils', 'medium hair', 'blonde hair', 'red eyes', 'red dress', 'mob cap', 'smile', 'short sleeves', 'yellow ascot', 'touhou project'])
            case 'sanae':
                text = make_prompt('kochiya sanae', ['green hair', 'blue footwear', 'long hair', 'green eyes', 'white dress', 'blue skirt', 'frog hair ornament', 'snake hair ornament', 'smile', 'standing', 'touhou project'])
            case 'sanaestand':
                text = make_prompt('kochiya sanae', ['green hair', 'blue footwear', 'long hair', 'green eyes', 'white dress', 'blue skirt', 'frog hair ornament', 'snake hair ornament', 'smile', 'touhou project'], sitting=False)
            case 'tenshi':
                text = make_prompt('hinanawi tenshi', ['blue hair', 'brown footwear', 'slit pupils', 'very long hair', 'red eyes', 'white dress', 'blue skirt', 'hat', 'black headwear', 'smile', 'touhou project'])
            case 'youmu':
                text = make_prompt('konpaku youmu', ['silver hair', 'black footwear', 'medium hair', 'slit pupils', 'green eyes', 'green dress', 'sleeveless dress', 'white sleeves', 'black ribbon', 'hair ribbon', 'unhappy', 'touhou project'])
            case 'yuyuko':
                text = make_prompt('saigyouji yuyuko', ['pink hair', 'black footwear', 'medium hair', 'pink eyes', 'wide sleeves', 'long sleeves', 'blue dress', 'mob cap', 'touhou project'])
            case 'nagisa':
                text = make_prompt('furukawa nagisa', ['brown hair', 'brown footwear', 'medium hair', 'brown eyes', 'smile', 'school briefcase', 'blue skirt', 'yellow jacket', 'antenna hair', 'dango', 'clannad'])
            case 'teto':
                text = make_prompt('kasane teto', ['pink hair', 'red footwear', 'red eyes', 'medium hair', 'detached sleeves', 'twin drills', 'drill hair', 'grey dress', 'smile', 'vocaloid'])
            case 'korone':
                text = make_prompt('inugami korone', ['yellow jacket', 'blue footwear', 'long hair', 'white dress', 'brown hair', 'brown eyes', 'on chair', 'hairclip', 'uwu', 'hololive'], on_floor=False)
            case 'kudo':
                text = make_prompt('kudryavka noumi', ['fang', 'black footwear', 'very long hair', 'white hat', 'white cape', 'silver hair', 'grey skirt', 'blue eyes', 'smile', 'little busters!'])
            case 'patchouli':
                text = make_prompt('patchouli knowledge', ['mob cap', 'pink footwear', 'long hair', 'slit pupils', 'striped dress', 'pink dress', 'purple hair', 'ribbons in hair', 'unhappy', 'touhou project'])
            case 'marisa':
                text = make_prompt('kirisame marisa', ['witch hat', 'black footwear', 'long hair', 'black dress', 'yellow eyes', 'blonde hair', 'white ribbon', 'smile', 'touhou project', 'puffy short sleeves', 'white shirt', 'buttons', 'white apron', 'bare legs', 'bare arms', 'braid', 'side braid', 'single braid', 'black headwear'])
            case _:
                text = f"photo of {placeholder_string}"

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        if stem not in self.cache:
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)

            if self.center_crop:
                crop = min(img.shape[0], img.shape[1])
                h, w, = (
                    img.shape[0],
                    img.shape[1],
                )
                img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

            image: Img = Image.fromarray(img)
            image: Img = image.resize((self.size, self.size), resample=self.interpolation)

            flipped: Img = hflip(image)

            def pil_to_latents(image: Img) -> Tensor:
                image = np.array(image).astype(np.uint8)
                image = (image / 127.5 - 1.0).astype(np.float32)
                latents: Tensor = torch.from_numpy(image).permute(2, 0, 1)
                return latents

            image, flipped = (pil_to_latents(variation) for variation in (image, flipped))
            
            self.cache[stem] = Variations(
                original=image,
                flipped=flipped,
            )
        variations = self.cache[stem]
        flip = torch.rand(1) < self.flip_p
        image = variations.flipped if flip else variations.original

        example["pixel_values"] = image
        example["prompt"] = text
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer and add the placeholder token as a additional special token
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models and create wrapper for stable diffusion
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    if is_xformers_available():
        try:
            unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed"
                f" correctly and a GPU is available: {e}"
            )
    
    token_ids: List[int] = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # regardless of whether the number of token_ids is 1 or more, it'll set one and then keep repeating.
    placeholder_token, placeholder_token_ids = add_tokens_and_get_placeholder_token(
        args, token_ids, tokenizer, text_encoder
    )

    # Freeze vae and unet
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    # Freeze all parameters except for the token embeddings in text encoder
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    train_dataset = TextualInversionDataset(
        data_root=args.train_data_dir,
        tokenizer=tokenizer,
        size=args.resolution,
        placeholder_token=placeholder_token,
        repeats=args.repeats,
        learnable_property=args.learnable_property,
        center_crop=args.center_crop,
        set="train",
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )
    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # Keep vae and unet in eval model as we don't train these
    vae.eval()
    unet.eval()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(args.output_dir, path))
        global_step = int(path.split("-")[1])

        resume_global_step = global_step * args.gradient_accumulation_steps
        first_epoch = resume_global_step // num_update_steps_per_epoch
        resume_step = resume_global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

    sampling_dtype = torch.float32
    alphas_cumprod: Tensor = get_alphas_cumprod(get_alphas(get_betas(device=accelerator.device)))
    model_sigmas: Tensor = get_sigmas(alphas_cumprod).to(sampling_dtype)
    model_log_sigmas: Tensor = get_log_sigmas(model_sigmas)
    model_sigma_min: Tensor = model_sigmas[0]
    model_sigma_max: Tensor = model_sigmas[-1]

    schedule_params_to_sigmas: Callable[[KarrasScheduleParams], Tensor] = lambda schedule: get_sigmas_karras(
        n=schedule.steps,
        sigma_max=schedule.sigma_max,
        sigma_min=schedule.sigma_min,
        rho=schedule.rho,
        device=model_sigmas.device,
    )

    # rather than training denoising on *random* timesteps: let's train on *only* the timesteps I'm planning to use
    # train it on two schedules: one for quick drafts, another for mastering
    searching_sigmas, mastering_sigmas = (schedule_params_to_sigmas(get_template_schedule(
        template=template,
        model_sigma_min=model_sigma_min,
        model_sigma_max=model_sigma_max,
        device=model_sigmas.device,
        dtype=model_sigmas.dtype,
    )) for template in (KarrasScheduleTemplate.Searching, KarrasScheduleTemplate.Mastering))
    searching_timesteps, mastering_timesteps = (log_sigmas_to_t(get_log_sigmas(sigmas[:-1]), model_log_sigmas) for sigmas in (searching_sigmas, mastering_sigmas))
    favourite_timesteps = torch.cat([searching_timesteps, mastering_timesteps]).unique()

    test_prompts = [
        f'photo of {placeholder_token}',
        f'photo of hatsune miku {placeholder_token}',
        f'photo of hakurei reimu {placeholder_token}',
        f'photo of aynami rei {placeholder_token}, evangelion',
        f'photo of asuka langley {placeholder_token}, evangelion',
        f'photo of steins;gate mayuri {placeholder_token}',
        f'photo of spice and wolf holo {placeholder_token}',
        f'photo of rin fate {placeholder_token}',
        f'photo of ruby rwby {placeholder_token}',
        f'photo of weiss rwby {placeholder_token}',
        f'photo of yang rwby {placeholder_token}',
        f'photo of blake rwby {placeholder_token}',
        f'photo of saber anime fate {placeholder_token}',
        f'photo of nero anime fate {placeholder_token}',
    ]
    model_needs: ModelNeeds = get_model_needs(args.pretrained_model_name_or_path, unet.dtype)
    unet_k_wrapped = DiffusersSD2Denoiser(unet, alphas_cumprod, sampling_dtype) if model_needs.needs_vparam else DiffusersSDDenoiser(unet, alphas_cumprod, sampling_dtype)
    denoiser_factory = DenoiserFactory(unet_k_wrapped)
    latents_to_bchw: LatentsToBCHW = make_latents_to_bchw(vae)

    batch_size = 1
    num_images_per_prompt = 1
    width = 768 if model_needs.is_768 else 512
    height = width
    latents_shape = (batch_size * num_images_per_prompt, unet.in_channels, height // 8, width // 8)

    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(text_encoder):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device).to(dtype=weight_dtype)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                # timesteps = torch.randint(
                #     0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                # ).long()
                timesteps = favourite_timesteps.index_select(0, torch.randint(0, favourite_timesteps.size(0), (bsz,), device=favourite_timesteps.device))

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.arange(len(tokenizer)) < placeholder_token_ids[0]
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(text_encoder, placeholder_token, placeholder_token_ids, accelerator, args, save_path)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            
            if global_step % args.visualization_steps == 0:
                if accelerator.is_main_process:
                    with torch.no_grad():
                        embed: Embed = get_embedder(
                            tokenizer,
                            accelerator.unwrap_model(text_encoder),
                            subtract_hidden_state_layers=int(model_needs.needs_penultimate_clip_hidden_state)
                        )
                        tracker: SummaryWriter = accelerator.get_tracker("tensorboard")

                        sampled_train_prompts = sample(batch['prompt'], min(args.visualization_train_samples, args.train_batch_size))
                        sampled_test_prompts = sample(test_prompts, min(args.visualization_test_samples, len(test_prompts)))

                        for sampled_prompts, provenance in zip([sampled_train_prompts, sampled_test_prompts], ['train', 'test']):
                            embeds: Tensor = embed(['', *sampled_prompts])
                            uc, *cs = embeds.split(1)

                            for prompt, c in zip(sampled_prompts, cs):
                                denoiser: Denoiser = denoiser_factory(uncond=uc, cond=c, cond_scale=7.5)
                                seed = get_seed()
                                generator = Generator(device='cpu').manual_seed(seed)
                                latents = randn(latents_shape, generator=generator, device='cpu', dtype=sampling_dtype).to(denoiser.denoiser.inner_model.device)
                                latents: Tensor = sample_dpmpp_2m(
                                    denoiser,
                                    latents * searching_sigmas[0],
                                    searching_sigmas,
                                ).to(vae.dtype)
                                bchw: Tensor = latents_to_bchw(latents)
                                chw, *_ = bchw
                                tracker.add_image('[%s][%d] %s' % (provenance, seed, prompt.replace(placeholder_token, '*')), np.asarray(chw.cpu()), global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using using the trained modules and save it.
    if accelerator.is_main_process:
        if args.push_to_hub and args.only_save_embeds:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = not args.only_save_embeds
        if save_full_model:
            pipeline = StableDiffusionPipeline(
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
                scheduler=PNDMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler"),
                safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_progress(text_encoder, placeholder_token, placeholder_token_ids, accelerator, args, save_path)

        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

    accelerator.end_training()


if __name__ == "__main__":
    main()
