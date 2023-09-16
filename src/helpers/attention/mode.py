from enum import Enum, auto

class AttentionMode(Enum):
    # accept diffusers' default attention processor (picks AttnProcessor2_0 if PyTorch supports it)
    Standard = auto()
    # AttnProcessor; uses baddbmm(), bmm()
    Classic = auto()
    # SlicedAttnProcessor; uses baddbmm(), bmm()
    Sliced = auto()
    # https://github.com/huggingface/diffusers/issues/1892
    # [not available on current branch] "memory-efficient" in pure PyTorch: torch.narrow()'d baddbmm(), bmm()s
    Chunked = auto()
    # replaces diffusers' Attention layers with torch.nn.MultiheadAttention
    TorchMultiheadAttention = auto()
    # AttnProcessor2_0; uses torch.nn.functional.scaled_dot_product_attention
    ScaledDPAttn = auto()
    # fork of AttnProcessor2_0 with a custom self-attn bias to encourage long-distance associations
    ScaledDPAttnDistBiased = auto()
    # fork of AttnProcessor with a custom softmax to try and bring the attention weights in-distribution when test-time context length does not match train-time
    ClassicWackySoftmax = auto()
    # fork of AttnProcessor2_0 which scales self-attention logits when query length is OOD, to output attention probabilities with entropy closer to original distribution.
    LogitScaled = auto()
    # XFormersAttnProcessor
    Xformers = auto()