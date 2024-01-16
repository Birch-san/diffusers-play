from diffusers import AutoencoderKL
from diffusers.models.attention import Attention
import torch
from torch.nn import Linear

def fuse_qkv(attn: Attention) -> None:
    has_bias = attn.to_q.bias is not None
    # throughout, we assume MHA (as opposed to GQA)
    qkv = Linear(in_features=attn.to_q.in_features, out_features=attn.to_q.out_features*3, bias=has_bias, dtype=attn.to_q.weight.dtype, device=attn.to_q.weight.device)
    qkv.weight.data.copy_(torch.cat([attn.to_q.weight.data * attn.scale, attn.to_k.weight.data, attn.to_v.weight.data]))
    if has_bias:
        qkv.bias.data.copy_(torch.cat([attn.to_q.bias.data * attn.scale, attn.to_k.bias.data, attn.to_v.bias.data]))
    setattr(attn, 'qkv', qkv)
    del attn.to_q, attn.to_k, attn.to_v

def fuse_vae_qkv(vae: AutoencoderKL) -> None:
    for attn in [*vae.encoder.mid_block.attentions, *vae.decoder.mid_block.attentions]:
        fuse_qkv(attn)