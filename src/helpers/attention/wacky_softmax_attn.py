from diffusers.models.attention import Attention
import torch
from torch import FloatTensor, BoolTensor
from typing import Optional
from enum import Enum, auto

class SoftmaxMode(Enum):
    Original = auto()
    Reimpl = auto()
    Topk = auto()
    DenomTopk = auto()
    CrudeResample = auto()

class WackySoftmaxAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    Based on:
    https://github.com/huggingface/diffusers/blob/3105c710ba16fa2cf54d8deb158099a4146da511/src/diffusers/models/attention_processor.py
    Once complete: this will fiddle with self-attention softmax, to try and make it do unspeakable things for out-of-distribution generation.
    """

    softmax_mode: SoftmaxMode
    rescale_softmax_output: bool
    log_entropy: bool

    def __init__(
        self,
        softmax_mode: SoftmaxMode,
        rescale_softmax_output: bool,
        log_entropy: bool,
    ) -> None:
        self.softmax_mode = softmax_mode
        self.rescale_softmax_output = rescale_softmax_output
        self.log_entropy = log_entropy

    def __call__(
        self,
        attn: Attention,
        hidden_states: FloatTensor,
        encoder_hidden_states: Optional[FloatTensor] = None,
        attention_mask: Optional[BoolTensor] = None,
        temb: Optional[FloatTensor] = None,
        key_length_factor: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> FloatTensor:
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
            heads=attn.heads,
            attention_mask=attention_mask,
            key_length_factor=key_length_factor,
            sigma=sigma,
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
        heads: int,
        attention_mask: Optional[BoolTensor] = None,
        key_length_factor: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> FloatTensor:
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

        if key_length_factor is None or key_length_factor == 1.0 or self.softmax_mode in [SoftmaxMode.Original, SoftmaxMode.Reimpl]:
            # normal attention
            match self.softmax_mode:
                case SoftmaxMode.Original:
                    attention_probs = attention_scores.softmax(dim=-1)
                case SoftmaxMode.Reimpl:
                    attention_probs = softmax(attention_scores)
        else:
            key_tokens = attention_scores.size(-1)
            preferred_token_count = int(key_tokens/key_length_factor)
            match self.softmax_mode:
                case SoftmaxMode.Topk:
                    attention_probs = softmax_topk(attention_scores, k=preferred_token_count)
                case SoftmaxMode.DenomTopk:
                    attention_probs = softmax_denom_topk(attention_scores, k=preferred_token_count)
                case SoftmaxMode.CrudeResample:
                    attention_probs = resample_crude_softmax(attention_scores, k=preferred_token_count)
                case _:
                    raise ValueError(f'Never heard of softmax mode "{self.softmax_mode}"')
        del attention_scores

        if self.log_entropy:
            entropy: FloatTensor = compute_attn_weight_entropy(attention_probs)
            entropy = entropy.unflatten(0, sizes=(-1, heads)).mean(-1)
            print(f'Entropy, sigma {sigma:02f}:')
            print(entropy)

        if self.rescale_softmax_output and key_length_factor is not None and key_length_factor != 1.0:
            attention_probs = attention_probs * key_length_factor

        attention_probs = attention_probs.to(dtype)

        return attention_probs

def softmax(x: torch.FloatTensor, dim=-1) -> torch.FloatTensor:
    """A normal softmax"""
    maxes = x.max(dim, keepdim=True).values
    diffs = x-maxes
    del x, maxes
    x_exp = diffs.exp()
    del diffs
    x_exp_sum = x_exp.sum(dim, keepdim=True)
    quotient = x_exp/x_exp_sum
    return quotient

def softmax_denom_topk(x: torch.FloatTensor, k: int, dim=-1) -> torch.FloatTensor:
    """
    Softmax whose denominator sums only the topk scores
    Sometimes fixes long-distance composition when generating larger-than-trained-distribution samples.
    https://twitter.com/Birchlabs/status/1643020670912045057
    """
    maxes = x.max(dim, keepdim=True).values
    diffs = x-maxes
    del x, maxes
    x_exp = diffs.exp()
    del diffs
    x_exp_sum = x_exp.topk(k=k, dim=dim).values.sum(dim, keepdim=True)
    quotient = x_exp/x_exp_sum
    return quotient

def softmax_topk(x: FloatTensor, k: int, dim=-1) -> FloatTensor:
    """
    Softmax employed in topk attention
    By Katherine Crowson
    """
    values, indices = torch.topk(x, k, dim=dim)
    return torch.full_like(x, float("-inf")).scatter_(dim, indices, values).softmax(dim=dim)

def resample_crude_softmax(x: torch.FloatTensor, k: int, dim=-1) -> torch.FloatTensor:
    """
    Softmax with a modified denominator. for each query token: resamples key dimension to size k; you can use this to increase/decrease denominator to the magnitude on which the model was trained.
    I think the results were bad, but can't really remember.
    """
    maxes = x.max(dim, keepdim=True).values
    diffs = x-maxes
    del maxes
    x_exp = diffs.exp()
    diffs_resampled = torch.nn.functional.interpolate(diffs, scale_factor=k/diffs.size(-1), mode='nearest-exact', antialias=False)
    del diffs
    diffs_exp_sum = diffs_resampled.exp().sum(dim, keepdim=True)
    del diffs_resampled
    quotient = x_exp/diffs_exp_sum
    return quotient

def compute_attn_weight_entropy(weights: FloatTensor) -> FloatTensor:
    """
    By Katherine Crowson.
    """
    entropy: FloatTensor = torch.sum(torch.special.entr(weights), dim=-1)
    return entropy