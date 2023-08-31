import torch.nn.functional as F
from diffusers.models.attention import Attention
from torch import FloatTensor, BoolTensor, arange
from typing import Optional, NamedTuple
import torch
from enum import Enum, auto
from logging import getLogger
from dataclasses import dataclass

class Dimensions(NamedTuple):
    height: int
    width: int

class BiasMode(Enum):
    Original = auto()
    LogBias = auto()
    NeighbourhoodMask = auto()

LOG = getLogger(__name__)

def make_wacky_bias(size: Dimensions, factor: float, device="cpu") -> FloatTensor:
    h, w = size
    h_ramp = arange(h, device=device, dtype=torch.float16)
    w_ramp = arange(w, device=device, dtype=torch.float16)

    hdist = h_ramp.reshape(1, 1, h, 1) - h_ramp.reshape(h, 1, 1, 1)
    wdist = w_ramp.reshape(1, 1, 1, w) - w_ramp.reshape(1, w, 1, 1)
    sq_dist = hdist**2 + wdist**2
    dist = sq_dist**.5
    log_dist = dist.log().clamp(min=0)
    bias = log_dist*factor
    bias = bias.reshape(h*w, h*w)

    return bias

# by Katherine Crowson
def make_neighbourhood_mask(size: Dimensions, size_orig: Dimensions, device="cpu") -> torch.BoolTensor:
    h, w = size
    h_orig, w_orig = size_orig

    h_ramp = torch.arange(h, device=device)
    w_ramp = torch.arange(w, device=device)
    h_pos, w_pos = torch.meshgrid(h_ramp, w_ramp, indexing="ij")

    # Compute start_h and end_h
    start_h = torch.clamp(h_pos - h_orig // 2, 0, h - h_orig)[..., None, None]
    end_h = start_h + h_orig

    # Compute start_w and end_w
    start_w = torch.clamp(w_pos - w_orig // 2, 0, w - w_orig)[..., None, None]
    end_w = start_w + w_orig

    # Broadcast and create the mask
    h_range = h_ramp.reshape(1, 1, h, 1)
    w_range = w_ramp.reshape(1, 1, 1, w)
    mask = (h_range >= start_h) & (h_range < end_h) & (w_range >= start_w) & (w_range < end_w)

    return mask.view(h * w, h * w)

def make_perimeter_mask(size: Dimensions, canvas_edge: Optional[int] = None, device='cpu') -> torch.BoolTensor:
    h, w = size

    h_ramp = torch.arange(h, device=device)
    w_ramp = torch.arange(w, device=device)

    # Broadcast and create the mask
    h_range = h_ramp.reshape(h, 1)
    w_range = w_ramp.reshape(1, w)
    
    mask: BoolTensor = (h_range < canvas_edge) | (h_range >= h-canvas_edge) | (w_range < canvas_edge) | (w_range >= w-canvas_edge)

    return mask.flatten()

@dataclass
class DistBiasedAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    Based on:
    https://github.com/huggingface/diffusers/blob/3105c710ba16fa2cf54d8deb158099a4146da511/src/diffusers/models/attention_processor.py
    Once complete: this will bias attention as a function of key token's distance from query token.
    """
    bias_mode: BiasMode = BiasMode.LogBias
    rescale_softmax_output: bool = False
    # suggested value of 2
    canvas_edge_thickness: Optional[int] = None
    neighbourhood_subtracts_canvas_edge: bool = True

    def __post_init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

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

        if is_self_attn and key_length_factor is not None and key_length_factor != 1.0 and self.bias_mode is not BiasMode.Original:
            assert attention_mask is None
            # TODO: access aspect ratio. for now we just assume a square
            current_h = current_w = int(sequence_length**.5)
            current_size = Dimensions(height=current_h, width=current_w)
            match self.bias_mode:
                case BiasMode.LogBias:
                    if sigma > 4:
                        # during high sigmas (i.e. when composition is being decided):
                        # bias self-attn towards distant tokens (global coherence)
                        factor=1
                    else:
                        # during low sigmas (i.e. when fine detail is being created):
                        # bias self-attn slightly towards nearby tokens (local coherence)
                        # this is pretty subtle; you could even consider just using attention_mask=None
                        factor=-.1
                    attention_mask: FloatTensor = make_wacky_bias(size=current_size, factor=factor, device=query.device)
                case BiasMode.NeighbourhoodMask:
                    preferred_token_count = int(sequence_length/attn.key_length_factor)
                    # TODO: access aspect ratio. for now we just assume a square
                    preferred_h = preferred_w = int(preferred_token_count**.5)

                    # if we are using LM-Infinite mode: let's attend to canvas edge, at the expense of attending to a smaller local neighbourhood
                    if self.canvas_edge_thickness is not None and self.neighbourhood_subtracts_canvas_edge:
                        preferred_h = max(0, preferred_h-self.canvas_edge_thickness*2)
                        preferred_w = max(0, preferred_w-self.canvas_edge_thickness*2)
                            
                    preferred_size = Dimensions(height=preferred_h, width=preferred_w)
                    attention_mask: BoolTensor = make_neighbourhood_mask(size=current_size, size_orig=preferred_size, device=query.device)

                    # LM-Infinite mode
                    # https://arxiv.org/abs/2308.16137
                    if self.canvas_edge_thickness is not None:
                        perimeter_mask: BoolTensor = make_perimeter_mask(size=current_size, canvas_edge=self.canvas_edge_thickness, device=query.device)
                        attention_mask |= perimeter_mask
                case _:
                    raise ValueError(f'Never heard of bias mode "{self.bias_mode}"')
            # broadcast over batch and head dims
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        if self.rescale_softmax_output and key_length_factor is not None and key_length_factor != 1.0:
            hidden_states = hidden_states * key_length_factor

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
