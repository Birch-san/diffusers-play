from diffusers.models.attention import Attention
import torch
from torch import FloatTensor, BoolTensor
from typing import Optional

class WackySoftmaxAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    Based on:
    https://github.com/huggingface/diffusers/blob/3105c710ba16fa2cf54d8deb158099a4146da511/src/diffusers/models/attention_processor.py
    Once complete: this will fiddle with self-attention softmax, to try and make it do unspeakable things for out-of-distribution generation.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(
            query,
            key,
            upcast_attention=attn.upcast_attention,
            upcast_softmax=attn.upcast_softmax,
            scale=attn.scale,
            attention_mask=attention_mask,
        )
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

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

    def get_attention_scores(
        self,
        query: FloatTensor,
        key: FloatTensor,
        upcast_attention: bool,
        upcast_softmax: bool,
        scale: float,
        attention_mask: Optional[BoolTensor] = None,
    ):
        dtype = query.dtype
        if upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            batch_x_heads, query_tokens, _ = query.shape
            _, key_tokens, _ = key.shape
            # expanding dims isn't strictly necessary (baddbmm supports broadcasting bias),
            # but documents the expected shape without allocating any additional memory
            attention_bias = torch.zeros(1, 1, 1, dtype=query.dtype, device=query.device).expand(
                batch_x_heads, query_tokens, key_tokens
            )
            beta = 0
        else:
            attention_bias = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            attention_bias,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=scale,
        )
        del attention_bias

        if upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs
