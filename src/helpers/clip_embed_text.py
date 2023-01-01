import torch
from transformers import CLIPTextModel, PreTrainedTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.tokenization_utils_base import BatchEncoding
from torch import Tensor, BoolTensor, no_grad
from .embed_text_types import Embed, EmbeddingAndMask, Prompts

def get_embedder(
  tokenizer: PreTrainedTokenizer,
  text_encoder: CLIPTextModel,
  subtract_hidden_state_layers = 0,
) -> Embed:
  def embed(prompts: Prompts) -> EmbeddingAndMask:
    tokens: BatchEncoding = tokenizer(prompts, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt", return_attention_mask=True)
    text_input_ids: Tensor = tokens.input_ids
    attention_mask: BoolTensor = tokens.attention_mask.to(dtype=torch.bool)
    with no_grad():
      encoder_outputs: BaseModelOutputWithPooling = text_encoder.forward(
        text_input_ids.to(text_encoder.device),
        output_hidden_states=subtract_hidden_state_layers != 0,
        return_dict=True,
      )
      text_embeddings: Tensor = encoder_outputs.last_hidden_state if subtract_hidden_state_layers == 0 else (
        text_encoder.text_model.final_layer_norm.forward(encoder_outputs.hidden_states[-1 - subtract_hidden_state_layers])
      )
    return text_embeddings, attention_mask
  return embed