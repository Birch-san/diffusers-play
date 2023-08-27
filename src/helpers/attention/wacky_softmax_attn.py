from diffusers.models.attention import Attention
import torch
from torch import FloatTensor, BoolTensor
from typing import Optional, Protocol
from enum import Enum, auto
from dataclasses import dataclass

class SoftmaxMode(Enum):
    Original = auto()
    Reimpl = auto()
    Topk = auto()
    DenomTopk = auto()
    CrudeResample = auto()

class IdentifyAttn(Protocol):
    def __call__(self, attn: Attention) -> str: ...

class ReportVariance(Protocol):
    def __call__(self, sigma: float, attn_key: str, variance: FloatTensor) -> None: ...

class GetVarianceScale(Protocol):
    def __call__(self, sigma: float, attn_key: str) -> FloatTensor: ...

@dataclass
class VarianceReporting:
    identify_attn: IdentifyAttn
    report_variance: ReportVariance

@dataclass
class VarianceCompensation:
    identify_attn: IdentifyAttn
    get_variance_scale: GetVarianceScale

@dataclass
class WackySoftmaxAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    Based on:
    https://github.com/huggingface/diffusers/blob/3105c710ba16fa2cf54d8deb158099a4146da511/src/diffusers/models/attention_processor.py
    Once complete: this will fiddle with self-attention softmax, to try and make it do unspeakable things for out-of-distribution generation.
    """

    softmax_mode: SoftmaxMode = SoftmaxMode.DenomTopk
    rescale_softmax_output: bool = False
    rescale_sim_variance: bool = False
    log_entropy: bool = False
    log_variance: bool = False
    variance_comp: Optional[VarianceCompensation] = None
    variance_report: Optional[VarianceReporting] = None

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
            attn=attn,
            is_self_attn=is_self_attn,
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
        attn: Attention,
        is_self_attn: bool,
        attention_mask: Optional[BoolTensor] = None,
        key_length_factor: Optional[float] = None,
        sigma: Optional[float] = None,
    ) -> FloatTensor:
        dtype = query.dtype
        if attn.upcast_attention:
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
            alpha=attn.scale,
        )
        del attention_bias

        if self.variance_report is not None and is_self_attn:
            attn_key: str = self.variance_report.identify_attn(attn=attn)
            variance: FloatTensor = attention_scores.unflatten(0, sizes=(-1, attn.heads)).var(-1).mean(-1)
            self.variance_report.report_variance(sigma=sigma, attn_key=attn_key, variance=variance)

        # we limit to mid-high sigmas only, to not destroy so much fine detail (we are only trying to influence the composition stage)
        if self.rescale_sim_variance and is_self_attn and sigma > 3:
            # scale factor that was found empirically to work (for generating 768x768 images, on a model which prefers 512x512 images).
            # this is significantly higher than the variance ratio that I measured in practice (1.0590 averaged over all heads):
            # https://gist.github.com/Birch-san/f394e5e069943fd5566b5e45a6888cd9
            attention_scores = attention_scores * 1.35

        if self.log_variance and is_self_attn:
            # print just per-head variance for final batch item (which we expect to be cond rather than uncond)
            variance: FloatTensor = attention_scores.unflatten(0, sizes=(-1, attn.heads))[-1].var(-1).mean(-1)
            print(', '.join(['%.4f' % s.item() for s in variance]))

        if attn.upcast_softmax:
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

        if self.log_entropy and is_self_attn:
            entropy: FloatTensor = compute_attn_weight_entropy(attention_probs.unflatten(0, sizes=(-1, attn.heads))[-1])
            entropy = entropy.mean(-1)
            # print just per-head entropy for final batch item (which we expect to be cond rather than uncond)
            print(', '.join(['%.4f' % s.item() for s in entropy]))

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