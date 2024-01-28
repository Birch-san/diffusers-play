import torch
from torch import FloatTensor, LongTensor
from torch.nn import Module
from typing import Optional, Dict, Any
from diffusers.utils import maybe_allow_in_graph
from diffusers.models.attention import BasicTransformerBlock

@maybe_allow_in_graph
class TransformerBlockUnNorm(Module):
    delegate: BasicTransformerBlock
    def __init__(self, delegate: BasicTransformerBlock) -> None:
        super().__init__()
        self.delegate = delegate

    def forward(
        self,
        hidden_states: FloatTensor,
        attention_mask: Optional[FloatTensor] = None,
        encoder_hidden_states: Optional[FloatTensor] = None,
        encoder_attention_mask: Optional[FloatTensor] = None,
        timestep: Optional[LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[LongTensor] = None,
    ):
        unnormed_hidden_states = hidden_states
        
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        if self.delegate.use_ada_layer_norm:
            norm_hidden_states = self.delegate.norm1(hidden_states, timestep)
        elif self.delegate.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.delegate.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        else:
            norm_hidden_states = self.delegate.norm1(hidden_states)

        # 0. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        self_attn_kwargs = {**cross_attention_kwargs, 'unnormed_hidden_states': unnormed_hidden_states}

        attn_output = self.delegate.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.delegate.only_cross_attention else None,
            attention_mask=attention_mask,
            **self_attn_kwargs,
        )
        del unnormed_hidden_states
        if self.delegate.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = attn_output + hidden_states

        # 1.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.delegate.fuser(hidden_states, gligen_kwargs["objs"])
        # 1.5 ends

        # 2. Cross-Attention
        if self.delegate.attn2 is not None:
            norm_hidden_states = (
                self.delegate.norm2(hidden_states, timestep) if self.delegate.use_ada_layer_norm else self.delegate.norm2(hidden_states)
            )

            attn_output = self.delegate.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 3. Feed-forward
        norm_hidden_states = self.delegate.norm3(hidden_states)

        if self.delegate.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.delegate._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            if norm_hidden_states.shape[self.delegate._chunk_dim] % self.delegate._chunk_size != 0:
                raise ValueError(
                    f"`hidden_states` dimension to be chunked: {norm_hidden_states.shape[self.delegate._chunk_dim]} has to be divisible by chunk size: {self.delegate._chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
                )

            num_chunks = norm_hidden_states.shape[self.delegate._chunk_dim] // self.delegate._chunk_size
            ff_output = torch.cat(
                [self.delegate.ff(hid_slice) for hid_slice in norm_hidden_states.chunk(num_chunks, dim=self.delegate._chunk_dim)],
                dim=self.delegate._chunk_dim,
            )
        else:
            ff_output = self.delegate.ff(norm_hidden_states)

        if self.delegate.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states