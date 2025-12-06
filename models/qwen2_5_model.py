from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RotaryEmbedding,
    Qwen2MLP,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
    QWEN2_START_DOCSTRING,
    QWEN2_INPUTS_DOCSTRING,
)
from dataclasses import dataclass


from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings
)
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
import math
import os
import numpy as np
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "meta-qwen2/Qwen2-2-7b-hf"
_CONFIG_FOR_DOC = "Qwen2Config"


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def get_projector(
    projector_type: str, in_dim: int, out_dim: int, projector_kwargs: dict
) -> nn.Module:
    if projector_type == "linear":
        return nn.Linear(in_dim, out_dim, bias=True)
    elif projector_type == "mlp":
        class MLP(nn.Module):
            def __init__(self, in_dim, out_dim, projector_kwargs):
                super().__init__()
                mlp_hidden_dim = projector_kwargs["mlp_hidden_dim"]
                self.up_proj = nn.Linear(in_dim, mlp_hidden_dim, bias=False)
                self.down_proj = nn.Linear(mlp_hidden_dim, out_dim, bias=False)
                self.act_fn = nn.SiLU()
            
            def forward(self, hidden_state):
                return self.down_proj(self.act_fn(self.up_proj(hidden_state)))
        return MLP(in_dim, out_dim, projector_kwargs)
    else:
        raise NotImplementedError(f"Projector type {projector_type} not found")
    

# Copied from transformers.models.llama.modeling_llama._prepare_4d_causal_attention_mask_with_cache_position
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

    return causal_mask

@dataclass
class RetrievalModelOutputWithPast(BaseModelOutputWithPast):
    """
    Base class for model's outputs that also contains a retr_logits attribute.
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    retr_logits: Optional[Tuple[torch.FloatTensor, ...]] = None


PADDING_VALUE = torch.finfo(torch.bfloat16).min

def get_kb_recall_stats(config, layer_idx, knowledge_info, ori_top_idx, topk_size):
    if getattr(config, "answer_idx", None) and layer_idx in knowledge_info:
        answer_idx = config.answer_idx
        if answer_idx:
            hit = 0
            hit_10 = 0
            hit_100 = 0
            hit_top = 0
            for i in answer_idx:
                hit += 1 if i in ori_top_idx[0] else 0
                hit_10 += 1 if i in ori_top_idx[0][:10] else 0
                hit_100 += 1 if i in ori_top_idx[0][:100] else 0
                hit_top += 1 if i in ori_top_idx[0][:len(answer_idx)] else 0
            setattr(config, f"recall_{layer_idx}@top", hit_top/len(answer_idx))
            setattr(config, f"recall_{layer_idx}@10", hit_10/len(answer_idx))
            setattr(config, f"recall_{layer_idx}@100", hit_100/len(answer_idx))
            setattr(config, f"recall_{layer_idx}@{topk_size}", hit/len(answer_idx))
            
            logger.info(f"TOP INDEX STATISTICS in Layer {layer_idx} @{topk_size}: {hit/len(answer_idx)}")
            logger.info(f"TOP INDEX STATISTICS in Layer {layer_idx} @100: {hit_100/len(answer_idx)}")
            logger.info(f"TOP INDEX STATISTICS in Layer {layer_idx} @10: {hit_10/len(answer_idx)}")
            logger.info(f"TOP INDEX STATISTICS in Layer {layer_idx} @top: {hit_top/len(answer_idx)}")
    return config

class SRKIQwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.sep_query_head = config.sep_query_head
        self.kb_layer_frequency = config.kb_layer_frequency
        self.dynamic_sparsify = config.dynamic_sparsify
        self.topk_size = config.top_k_kb
        self.projector_kwargs = config.projector_kwargs if hasattr(config, "projector_kwargs") else None
        self.return_retr_logits = config.return_retr_logits
        self.reuse_kb = getattr(config, "reuse_kb", None)
        self.knowledge_info = getattr(config, "knowledge_info", [])
        
        # add a new query projection for the new query head, only when the layer_idx is divisible by kb_layer_frequency
        if self.layer_idx % config.kb_layer_frequency == 0:
            self.embed_dim = config.embed_dim
            if self.sep_query_head:
                self.q_proj_new = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
            self.k_proj_new = get_projector(config.project_type, self.embed_dim, self.num_heads * self.head_dim, self.projector_kwargs)  
            self.v_proj_new = get_projector(config.project_type, self.embed_dim, self.num_heads * self.head_dim, self.projector_kwargs)
            self.key_layernorm = nn.LayerNorm(self.num_heads * self.head_dim, elementwise_affine=False, bias=False)
        
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def prune_key_value(self, query, kb_keys, kb_values, topk_size=20):
        batch_size, num_heads, kb_len, head_dim = kb_keys.shape
        
        attn_weights = torch.matmul(query, kb_keys.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )  # Batchsize, num_heads, query_size, key_size
        
        if topk_size >= kb_len:
            return kb_keys, kb_values, attn_weights
        
        top_idx = None
        if self.reuse_kb and self.layer_idx > min(self.knowledge_info) and not self.training:
            top_idx = self.cached_top_idx
            ori_top_idx = top_idx[:, 0, :, 0].squeeze(1).squeeze(-1)
        
        if top_idx is None:            
            # Seq-wise pruning
            ori_top_idx = attn_weights.sum((1, 2)).topk(min(kb_len, topk_size), -1)[1] # (bsz, topk_size)
            
            # get the kb recall stats
            self.config = get_kb_recall_stats(self.config, self.layer_idx, self.knowledge_info, ori_top_idx, topk_size)
            
            ori_top_idx = ori_top_idx.sort(dim=-1)[0]
            
            # During training, if the answer index is not in the top-k, we need to concat 0 to the first position
            if self.training and self.layer_idx == min(self.knowledge_info): # If current layer is exactly the retrieval layer, we concat the correct index into top_idx            
                # concat [0,1,2,3] directly, [0,1,2,3] is always the correct index during training
                ori_top_idx = torch.concat([torch.tensor([[0,1,2,3]]).expand(ori_top_idx.shape[0], 4).to(ori_top_idx.device), ori_top_idx[:, 4:]], dim=-1)
            
            top_idx = ori_top_idx.view(batch_size, -1, topk_size, 1).expand(
                batch_size, num_heads, topk_size, head_dim
            )    
                       
        self.ori_top_idx = ori_top_idx
        # cache the top_idx
        if self.reuse_kb and self.layer_idx in self.knowledge_info and not self.training:
            self.cached_top_idx = top_idx # [bsz, num_heads, topk_size, head_dim]

        kb_keys = kb_keys.gather(-2, top_idx)
        kb_values = kb_values.gather(-2, top_idx)

        top_idx_attn = ori_top_idx.unsqueeze(1).unsqueeze(2).expand(
            batch_size, num_heads, attn_weights.shape[2], ori_top_idx.shape[-1]
        )
        
        attn_weights = attn_weights.gather(-1, top_idx_attn)
        return kb_keys, kb_values, attn_weights
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        key_embds: Optional[torch.Tensor] = None,
        value_embds: Optional[torch.Tensor] = None,
        save_attention_weights: bool = True,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # config
        sep_query_head = self.sep_query_head
        kb_layer_frequency = self.kb_layer_frequency
        dynamic_sparsify = self.dynamic_sparsify
        topk_size = self.topk_size

        if self.layer_idx % kb_layer_frequency == 0 and sep_query_head:
            query_states_2 = self.q_proj_new(hidden_states)
            query_states_2 = query_states_2.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights_2 = None
        if key_embds is not None and position_ids.nelement() > 1: 
            if self.layer_idx % kb_layer_frequency == 0:
                kb_keys_embeddings, kb_values_embeddings = (
                    key_embds, value_embds  # (batch_size, kb_len, out_dim)
                )                
                kb_keys = self.k_proj_new(kb_keys_embeddings)
                kb_keys = self.key_layernorm(kb_keys)
                kb_values = self.v_proj_new(kb_values_embeddings)
                kb_keys = kb_keys.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
                kb_values = kb_values.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2)
                if dynamic_sparsify:
                    kb_keys, kb_values, pruned_attn_weights = self.prune_key_value(
                        query_states_2, kb_keys, kb_values, topk_size
                    )
                    attn_weights_2 = pruned_attn_weights
                key_states = torch.concat([kb_keys, key_states], dim=2)
                value_states = torch.concat([kb_values, value_states], dim=2)

                # Modify the attention matrix: Appendx a (seq_len, kb_len) block to the left
                kb_len = kb_keys.shape[2]
                self.kb_len = kb_len # add a new attribute to store the kb_len

                kb_atten_mask = attention_mask.new_zeros(bsz, 1, q_len, kb_len)
                padding_mask = torch.all(
                    attention_mask < 0, -1, keepdim=True
                )  # (bsz, num_heads, q_len, 1)
                kb_atten_mask = (
                    padding_mask * PADDING_VALUE + (~padding_mask) * kb_atten_mask
                )
                attention_mask = torch.concat([kb_atten_mask, attention_mask], dim=-1)
        
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)


        if sep_query_head and key_embds is not None and self.layer_idx % kb_layer_frequency == 0:
            if attn_weights_2 is None:
                attn_weights_2 = torch.matmul(
                    query_states_2, key_states[:, :, :self.kb_len].transpose(2, 3)
                ) / math.sqrt(self.head_dim)
            
            attn_weights = torch.matmul(
                query_states, key_states[:, :, self.kb_len:].transpose(2, 3)
            ) / math.sqrt(self.head_dim)
            
            attn_weights = torch.concat([attn_weights_2, attn_weights], -1)
        else:
            attn_weights = torch.matmul(
                query_states, key_states.transpose(2, 3)
            ) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            if attention_mask.shape[-1] != key_states.shape[-2]: # deal with the case that kb_kvs are injected
                kb_atten_mask = attention_mask.new_zeros(bsz, 1, q_len, self.kb_len)
                attention_mask = torch.cat([kb_atten_mask, attention_mask], dim=-1)
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        if save_attention_weights and q_len > 1:
            attn_weights_to_save = attn_weights_2.clone()
            attn_weights_to_save = nn.functional.softmax(attn_weights_to_save, dim=-1, dtype=torch.float32)
            save_path = os.path.join(
                attention_save_loc,
                f"{attention_file_base_name}_{self.layer_idx}.npy",
            )
            np.save(
                save_path,
                attn_weights_to_save.cpu().detach().numpy(),
            )
        attn_weights = attn_weights.to(query_states.dtype)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.attention_dropout, training=self.training
        )
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        
        if self.return_retr_logits:
            return attn_output, attn_weights, past_key_value, pruned_attn_weights
        else:
            return attn_output, attn_weights, past_key_value
        

QWEN2_ATTENTION_CLASSES = {
    "eager": SRKIQwen2Attention,
    "flash_attention_2": SRKIQwen2Attention,
    "sdpa": SRKIQwen2Attention,
}

class SRKIQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.return_retr_logits = getattr(config, "return_retr_logits", False)
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        key_embds: Optional[torch.Tensor] = None,
        value_embds: Optional[torch.Tensor] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            key_embds=key_embds,
            value_embds=value_embds,
            save_attention_weights=save_attention_weights,
            attention_save_loc=attention_save_loc,
            attention_file_base_name=attention_file_base_name,
        )
        if self.return_retr_logits:
            hidden_states, self_attn_weights, present_key_value, attn_weights_2 = attn_outputs
        else:
            hidden_states, self_attn_weights, present_key_value = attn_outputs

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if self.self_attn.return_retr_logits:
            outputs += (attn_weights_2,)

        return outputs


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class SRKIQwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [SRKIQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._no_split_modules = ["SRKIQwen2DecoderLayer"]
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.return_retr_logits = getattr(config, "return_retr_logits", False)
        self.knowledge_info = getattr(config, "knowledge_info", [])
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        key_embds: Optional[torch.Tensor] = None,
        value_embds: Optional[torch.Tensor] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache) and not self.training:
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_retr_logits = [] if self.return_retr_logits else None
        
        for l_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    key_embds,
                    value_embds,
                    save_attention_weights,
                    attention_save_loc,
                    attention_file_base_name,
                )
            else:
                if getattr(self.config, "reuse_kb", None) and l_idx > min(self.knowledge_info) and not self.training:
                    decoder_layer.self_attn.cached_top_idx = self.cached_top_idx
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    key_embds=key_embds,
                    value_embds=value_embds,
                    save_attention_weights=save_attention_weights,
                    attention_save_loc=attention_save_loc,
                    attention_file_base_name=attention_file_base_name,
                )
                if getattr(self.config, "reuse_kb", None) and l_idx in self.knowledge_info and not self.training:
                    self.cached_top_idx = decoder_layer.self_attn.cached_top_idx
                             
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if self.return_retr_logits:
                all_retr_logits.append(layer_outputs[-1])

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return RetrievalModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            retr_logits=all_retr_logits,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class SRKIQwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = SRKIQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.return_retr_logits = getattr(config, "return_retr_logits", False)
        
        self.knowledge_info = getattr(config, "knowledge_info", [])
        logger.info(f"knowledge_info: {self.knowledge_info}")

        # Initialize weights and apply final processing
        self.post_init()

        if config._attn_implementation == "flash_attention_2":
            raise NotImplementedError(
                "Flash Attention 2 is not yet supported for SRKI."
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
        
    # Retrieval loss from logits
    def retrieval_loss_from_logits(self, logits, qa_types, tau: float = 0.05):
        """
        logits:   [B, num_heads, q_len, kb_len]
        qa_types: QA types, used to determine the positive indices
        tau:      temperature parameter, used to scale the logits
        """
        B, H, Q, K = logits.shape
        logits = logits.mean((1, 2))
        logits = logits / tau
        
        total_loss = 0.0
        num_valid_samples = 0
        
        for i in range(B):
            qa_type = qa_types[i].item()
            
            # determine the positive indices according to the qa_type
            if qa_type == 0:
                positive_indices = [0, 1]
            elif qa_type == 1:
                positive_indices = [0, 1, 2, 3]
            elif qa_type == 2:
                continue # skip the unanswerable qa type
            else:
                raise ValueError(f"unsupported qa_type {qa_type}")
            
            sample_losses = []
            for pos_idx in positive_indices:
                # Create a mask, exclude other positive indices, only keep the current positive index and all negative indices
                mask = torch.ones(K, dtype=torch.bool, device=logits.device)
                for other_pos_idx in positive_indices:
                    if other_pos_idx != pos_idx:
                        mask[other_pos_idx] = False
                
                filtered_logits = logits[i][mask]  # only keep the current positive index and all negative indices
                
                # target label is 0 (because the current positive index is the first in the filtered logits)
                target = torch.tensor(0, device=logits.device, dtype=torch.long)
                
                loss = F.cross_entropy(filtered_logits.unsqueeze(0), target.unsqueeze(0))
                sample_losses.append(loss)
            
            if sample_losses:
                sample_loss = torch.stack(sample_losses).mean()
                total_loss += sample_loss
                num_valid_samples += 1
        
        if num_valid_samples > 0:
            final_loss = total_loss / num_valid_samples
        else:
            final_loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        return final_loss

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        key_embds: Optional[torch.Tensor] = None,
        value_embds: Optional[torch.Tensor] = None,
        qa_types:Optional[torch.Tensor] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            key_embds=key_embds,
            value_embds=value_embds,
            save_attention_weights=save_attention_weights,
            attention_save_loc=attention_save_loc,
            attention_file_base_name=attention_file_base_name,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if getattr(outputs, "retr_logits", None):             
                layer_weights = self.config.retrieval_layer_weights
                retr_logits = outputs.retr_logits
                loss_retr = 0.0
                for l_idx, (retr_logit, w) in enumerate(zip(retr_logits, layer_weights)):
                    if l_idx in self.knowledge_info: # only calculate the retrieval layer loss
                        loss_retr += self.retrieval_loss_from_logits(retr_logit, qa_types) * w
                        logger.info(f"debug: layer idx: {l_idx}, loss_retr: {loss_retr.item()}, w: {w}")
                loss = loss + loss_retr
                logger.info(f"debug: loss_retr: {loss_retr.item()}, final loss: {loss.item()}")

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        key_embds: Optional[torch.Tensor] = None,
        value_embds: Optional[torch.Tensor] = None,
        save_attention_weights: bool = False,
        attention_save_loc: Optional[str] = None,
        attention_file_base_name: Optional[str] = None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "key_embds": key_embds,
                "value_embds": value_embds,
                "save_attention_weights": save_attention_weights,
                "attention_save_loc": attention_save_loc,
                "attention_file_base_name": attention_file_base_name,
            }
        )
        return model_inputs