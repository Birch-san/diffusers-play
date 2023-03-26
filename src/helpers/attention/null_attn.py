from diffusers.models.cross_attention import CrossAttention
from diffusers.models.attention import BasicTransformerBlock, FeedForward
from torch import nn, Tensor, FloatTensor
from typing import Optional, Callable
from functools import partial
from .attn_compatible import CrossAttnCompatible
from ..tap.tap_module import TapModule

# compatible, but doesn't give us viable denoiser outputs
class NullAttnProcessor:
    def __call__(
        self,
        attn: CrossAttention,
        hidden_states: FloatTensor,
        encoder_hidden_states: Optional[FloatTensor] = None,
        attention_mask: Optional[FloatTensor] = None,
        encoder_attention_bias: Optional[FloatTensor] = None,
    ) -> FloatTensor:
        # lol
        return hidden_states

# compatible, but doesn't give us viable denoiser outputs
class NullAttention(nn.Module, CrossAttnCompatible):
    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        **cross_attention_kwargs,
    ) -> Tensor:
        # lol
        return hidden_states

def to_null_attn(ca: CrossAttention) -> NullAttention:
    return NullAttention()

# you need to keep at least self-attention for the results to be somewhat usable
class NullTransformerBlock(nn.Module):
    norm1: nn.LayerNorm
    self_attn: CrossAttention
    norm3: nn.LayerNorm
    ff: FeedForward
    def __init__(
        self,
        norm1: nn.LayerNorm,
        self_attn: CrossAttention,
        norm3: nn.LayerNorm,
        ff: FeedForward,
    ) -> None:
       super().__init__()
       self.norm1 = norm1
       self.self_attn = self_attn
       self.norm3 = norm3
       self.ff = ff

    def forward(
        self,
        hidden_states: FloatTensor,
        encoder_hidden_states=None,
        timestep=None,
        attention_mask=None,
        cross_attention_kwargs=None,
        class_labels=None,
    ) -> FloatTensor:
        norm_hidden_states: FloatTensor = self.norm1(hidden_states)
        attn_output: FloatTensor = self.self_attn(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states: FloatTensor = attn_output + hidden_states
        norm_hidden_states: FloatTensor = self.norm3(hidden_states)
        ff_output: FloatTensor = self.ff(norm_hidden_states)
        hidden_states = ff_output + hidden_states
        return hidden_states
    
def to_null_basic_transformer_block(b: BasicTransformerBlock) -> NullTransformerBlock:
    return NullTransformerBlock(
        norm1=b.norm1,
        self_attn=b.attn1,
        norm3=b.norm3,
        ff=b.ff,
    )

ReplaceBasicTransformerBlock = Callable[[BasicTransformerBlock], nn.Module]

def _replace_basic_transformer_block(replace_module: ReplaceBasicTransformerBlock, module: nn.Module) -> None:
  for name, m in module.named_children():
    if isinstance(m, BasicTransformerBlock):
      replacement: nn.Module = replace_module(m)
      setattr(module, name, replacement)

def replace_basic_transformer_block_to_tap_module(tap_basic_transformer_block: ReplaceBasicTransformerBlock) -> TapModule:
   return partial(_replace_basic_transformer_block, tap_basic_transformer_block)
