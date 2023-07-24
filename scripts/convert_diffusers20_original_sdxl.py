# convert Diffusers v1.x/v2.0 model to original Stable Diffusion

import argparse
import os
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection

import library.sdxl_model_util as model_util


def convert(args):
    # 引数を確認する
    load_dtype = torch.float16 if args.fp16 else None

    save_dtype = None
    if args.fp16 or args.save_precision_as == "fp16":
        save_dtype = torch.float16
    elif args.bf16 or args.save_precision_as == "bf16":
        save_dtype = torch.bfloat16
    elif args.float or args.save_precision_as == "float":
        save_dtype = torch.float

    is_load_ckpt = os.path.isfile(args.model_to_load)
    is_save_ckpt = len(os.path.splitext(args.model_to_save)[1]) > 0

    # assert (
    #     is_save_ckpt or args.reference_model is not None
    # ), f"reference model is required to save as Diffusers / Diffusers形式での保存には参照モデルが必要です"

    # モデルを読み込む
    msg = "checkpoint" if is_load_ckpt else ("Diffusers" + (" as fp16" if args.fp16 else ""))
    print(f"loading {msg}: {args.model_to_load}")

    if is_load_ckpt:
        (
            text_model1,
            text_model2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = model_util.load_models_from_sdxl_checkpoint(model_util.MODEL_VERSION_SDXL_BASE_V0_9, args.model_to_load, 'cpu')
    else:
        pipe: StableDiffusionXLPipeline = StableDiffusionXLPipeline.from_pretrained(
            args.model_to_load, torch_dtype=load_dtype, tokenizer=None, safety_checker=None
        )
        text_model1: CLIPTextModel = pipe.text_encoder
        text_model2: CLIPTextModelWithProjection = pipe.text_encoder_2
        vae: AutoencoderKL = pipe.vae
        unet: UNet2DConditionModel = pipe.unet

    # 変換して保存する
    msg = ("checkpoint" + ("" if save_dtype is None else f" in {save_dtype}")) if is_save_ckpt else "Diffusers"
    print(f"converting and saving as {msg}: {args.model_to_save}")

    if is_save_ckpt:
        key_count = model_util.save_stable_diffusion_checkpoint(
            args.model_to_save,
            text_model1,
            text_model2,
            unet,
            args.epoch,
            args.global_step,
            ckpt_info,
            vae,
            logit_scale,
            save_dtype,
        )
        print(f"model saved. total converted state_dict keys: {key_count}")
    else:
        print(f"copy scheduler/tokenizer config from: {args.reference_model if args.reference_model is not None else 'default model'}")
        model_util.save_diffusers_checkpoint(
            args.model_to_save,
            text_model1,
            text_model2,
            unet,
            args.reference_model,
            vae,
            args.use_safetensors,
            save_dtype,
        )
        print(f"model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="load as fp16 (Diffusers only) and save as fp16 (checkpoint only) / fp16形式で読み込み（Diffusers形式のみ対応）、保存する（checkpointのみ対応）",
    )
    parser.add_argument("--bf16", action="store_true", help="save as bf16 (checkpoint only) / bf16形式で保存する（checkpointのみ対応）")
    parser.add_argument(
        "--float", action="store_true", help="save as float (checkpoint only) / float(float32)形式で保存する（checkpointのみ対応）"
    )
    parser.add_argument(
        "--save_precision_as",
        type=str,
        default="no",
        choices=["fp16", "bf16", "float"],
        help="save precision, do not specify with --fp16/--bf16/--float / 保存する精度、--fp16/--bf16/--floatと併用しないでください",
    )
    parser.add_argument("--epoch", type=int, default=0, help="epoch to write to checkpoint / checkpointに記録するepoch数の値")
    parser.add_argument(
        "--global_step", type=int, default=0, help="global_step to write to checkpoint / checkpointに記録するglobal_stepの値"
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="scheduler/tokenizerのコピー元Diffusersモデル、Diffusers形式で保存するときに使用される、省略時は`runwayml/stable-diffusion-v1-5` または `stabilityai/stable-diffusion-2-1` / reference Diffusers model to copy scheduler/tokenizer config from, used when saving as Diffusers format, default is `runwayml/stable-diffusion-v1-5` or `stabilityai/stable-diffusion-2-1`",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="use safetensors format to save Diffusers model (checkpoint depends on the file extension) / Duffusersモデルをsafetensors形式で保存する（checkpointは拡張子で自動判定）",
    )

    parser.add_argument(
        "model_to_load",
        type=str,
        default=None,
        help="model to load: checkpoint file or Diffusers model's directory / 読み込むモデル、checkpointかDiffusers形式モデルのディレクトリ",
    )
    parser.add_argument(
        "model_to_save",
        type=str,
        default=None,
        help="model to save: checkpoint (with extension) or Diffusers model's directory (without extension) / 変換後のモデル、拡張子がある場合はcheckpoint、ない場合はDiffusesモデルとして保存",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    convert(args)
