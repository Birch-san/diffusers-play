from torch import FloatTensor, BoolTensor
import torch.nn.functional as F
from diffusers.models.attention import Attention
from typing import Optional
import math

class LogitScalingAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    Based on:
    https://github.com/huggingface/diffusers/blob/3105c710ba16fa2cf54d8deb158099a4146da511/src/diffusers/models/attention_processor.py
    Once complete: this will implement the logit scaling from:
    "Training-free Diffusion Model Adaptation for Variable-Sized Text-to-Image Synthesis"
    https://arxiv.org/abs/2306.08645
    to scale self-attention logits when query length is OOD, to output attention probabilities with entropy closer to original distribution.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("LogitScalingAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: FloatTensor,
        encoder_hidden_states: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        temb: Optional[FloatTensor] = None,
        self_attn_key_length_factor: Optional[float] = None,
    ) -> FloatTensor:
        residual = hidden_states

        assert self_attn_key_length_factor is not None

        is_self_attn = encoder_hidden_states is None

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        train_key_len = int(sequence_length / self_attn_key_length_factor)
        self_attn_logit_scale_factor: float = math.log(sequence_length, train_key_len) ** .5

        scale = attn.scale*self_attn_logit_scale_factor if is_self_attn else attn.scale

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # note: use of the `scale` param requires PyTorch 2.1-era scaled_dot_product_attention.
        #       if you would like to use this in older PyTorch, then you could instead:
        #       - remove the `scale=scale` kwarg
        #       - fuse the scale factor into the query (or into the key would work too):
        #         if is_self_attn:
        #           query *= self_attn_logit_scale_factor
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False, scale=scale
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states