from enum import Enum, auto

class AttentionMode(Enum):
    # usual diffusers Attention layer, AttnProcessor via baddbmm(), bmm()
    Standard = auto()
    # https://github.com/huggingface/diffusers/issues/1892
    # usual diffusers Attention layer, AttnProcessor via torch.narrow()'d baddbmm(), bmm()s ("memory-efficient" in pure PyTorch)
    Chunked = auto()
    # replaces diffusers' Attention layers with torch.nn.MultiheadAttention
    TorchMultiheadAttention = auto()
    # usual diffusers Attention layer, AttnProcessor via torch.nn.functional.scaled_dot_product_attention
    ScaledDPAttn = auto()
    # usual diffusers Attention layer, AttnProcessor via Xformers
    Xformers = auto()