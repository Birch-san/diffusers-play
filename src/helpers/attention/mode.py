from enum import Enum, auto

class AttentionMode(Enum):
    # usual diffusers CrossAttention layer, CrossAttnProcessor via baddbmm(), bmm()
    Standard = auto()
    # https://github.com/huggingface/diffusers/issues/1892
    # usual diffusers CrossAttention layer, CrossAttnProcessor via torch.narrow()'d baddbmm(), bmm()s ("memory-efficient" in pure PyTorch)
    Chunked = auto()
    # replaces diffusers' CrossAttention layers with torch.nn.MultiheadAttention
    TorchMultiheadAttention = auto()
    # usual diffusers CrossAttention layer, CrossAttnProcessor via torch.nn.functional.scaled_dot_product_attention
    ScaledDPAttn = auto()
    # usual diffusers CrossAttention layer, CrossAttnProcessor via Xformers
    Xformers = auto()