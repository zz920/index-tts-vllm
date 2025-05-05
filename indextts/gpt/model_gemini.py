import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import (assert_device_map,
                                                     get_device_map)
from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.arch_util import AttentionBlock
# from indextts.utils.typical_sampling import TypicalLogitsWarper # Keep for reference, but not used by vLLM

# ****** ADD VLLM and ASYNCIO Imports ******
import asyncio
import os
import tempfile
import logging
try:
    from vllm import LLM, SamplingParams
    _vllm_available = True
except ImportError:
    print("Warning: vLLM not found. vLLM inference will not be available.")
    LLM = None
    SamplingParams = None
    _vllm_available = False
# ******************************************

logger = logging.getLogger(__name__)


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)


class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)

# --- GPT2InferenceModel remains unchanged, but won't be directly used by vLLM ---
class GPT2InferenceModel(GPT2PreTrainedModel):
    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=False):
        super().__init__(config)
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb # Note: renamed from mel_pos_embedding in original inference code
        self.embeddings = embeddings # Note: Mel Embeddings in original inference code
        self.final_norm = norm
        # self.lm_head = nn.Sequential(norm, linear) # Original - sequential norm+linear
        self.lm_head = linear # Modified for consistency with vLLM loading lm_head separately
        self.norm = norm # Keep norm separate

        self.kv_cache = kv_cache

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(max(1, torch.cuda.device_count())))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.norm = self.norm.to(self.transformer.first_device) # Norm needs parallelization too
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.norm = self.norm.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            # "inputs_embeds": inputs_embeds # Added potential embeds pass-through
        }

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None, # Changed from None assertion
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        # assert self.cached_mel_emb is not None # No longer needed if embeds passed directly
        # assert inputs_embeds is None  # Not supported by this inference model. # Now supported
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            # Create embedding (original logic if needed for compatibility)
            assert self.cached_mel_emb is not None, "Cached mel emb needed if not passing embeds directly"
            mel_len = self.cached_mel_emb.shape[1]
            if input_ids.shape[1] != 1: # Initial call
                text_inputs = input_ids[:, mel_len:]
                text_emb = self.embeddings(text_inputs) # These are MEL embeddings in this context
                # Assuming text_pos_embedding here applies to the MEL tokens being generated
                text_emb = text_emb + self.text_pos_embedding(text_emb) # Uses text_pos_embedding instance

                if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                    mel_emb = self.cached_mel_emb.repeat_interleave(
                        text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
                    )
                else:
                    mel_emb = self.cached_mel_emb
                emb = torch.cat([mel_emb, text_emb], dim=1)
            else: # Subsequent calls (autoregressive step)
                emb = self.embeddings(input_ids) # Mel embeddings
                # Get fixed embedding for the current position relative to text start
                # This assumes attention_mask covers the full context length
                current_pos_index = attention_mask.shape[1] - 1 # Index of the token being generated
                emb = emb + self.text_pos_embedding.get_fixed_embedding(
                    current_pos_index, attention_mask.device
                )
        else:
             # If inputs_embeds is provided, use it directly
             emb = inputs_embeds

        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            # Note: MPS lacks set_device, handle differently if needed
            if torch.cuda.is_available():
                 torch.cuda.set_device(self.transformer.first_device)

            hidden_states = hidden_states.to(self.norm.weight.device) # Move to the device of norm/lm_head

        hidden_states = self.norm(hidden_states) # Apply final norm
        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            # return h # Original: Returns (B, C, S)
            return h.permute(0, 2, 1) # Return (B, S, C) for consistency with Perceiver/Conformer

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        # Input x can be token_ids (b, seq) or embeddings (b, seq, dim)
        sl = x.shape[1]
        # Ensure range is within embedding size
        positions = torch.arange(0, sl, device=x.device)
        # Clamp positions to avoid index out of bounds if sl > seq_len
        positions = torch.clamp(positions, 0, self.emb.num_embeddings - 1)
        pos_emb = self.emb(positions) # (seq, dim)
        return pos_emb # Return (seq, dim), will be broadcast added

    def get_fixed_embedding(self, ind, dev):
         # Ensure ind is within bounds
        clamped_ind = torch.clamp(torch.tensor([ind], device=dev), 0, self.emb.num_embeddings - 1)
        return self.emb(clamped_ind) # (1, dim), remove unsqueeze

def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing, activation_function):
    """
    GPT-2 implemented by the HuggingFace library.
    Returns the GPT2Model and necessary position embeddings.
    """
    total_seq_len = max_mel_seq_len + max_text_seq_len + 2 # +2 for start/stop tokens maybe? Adjust if needed
    gpt_config = GPT2Config(vocab_size=1,  # Vocab size is not used by the core model, only wrapper/heads
                            n_positions=total_seq_len,
                            # n_ctx=total_seq_len, # n_ctx is deprecated, use n_positions
                            n_embd=model_dim,
                            n_layer=layers,
                            n_head=heads,
                            activation_function=activation_function or "gelu_new",
                            use_cache=not checkpointing, # Use cache for inference
                            # gradient_checkpointing=checkpointing # Only relevant for training
                           )
    gpt = GPT2Model(gpt_config)
    # Override the built-in positional embeddings - we handle them externally
    # del gpt.wpe # Keep wpe, but we won't use it directly with token IDs
    gpt.wpe = nn.Embedding(gpt_config.n_positions, gpt_config.n_embd) # Keep standard pos embedding layer
    # Built-in token embeddings are unused by the core model if inputs_embeds is provided
    # del gpt.wte # Keep wte, but we won't use it directly with token IDs
    gpt.wte = nn.Embedding(1, gpt_config.n_embd) # Dummy token embedding

    # Create separate position embeddings for MEL and Text parts
    # We need these based on the *maximum* expected lengths for each modality
    # Add buffer for start/stop tokens and conditioning inputs if they affect position indices
    # max_mel_seq_len should account for start/stop + conditioning
    mel_pos_emb = LearnedPositionEmbeddings(max_mel_seq_len, model_dim)
    # max_text_seq_len should account for start/stop
    text_pos_emb = LearnedPositionEmbeddings(max_text_seq_len, model_dim)

    return gpt, mel_pos_emb, text_pos_emb, None, None # Return None for layer pos embeddings


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1),
                                     nn.Sequential(*[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 16, channels // 2),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 8, channels),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
                                     )
        self.reduction = 4

    def forward(self, x):
        # Input x: (B, C, S) e.g. (B, 80, S_mel)
        for e in self.encoder:
            x = e(x)
        # Output: (B, C_model, S_reduced)
        return x.permute(0, 2, 1) # (B, S_reduced, C_model) for transformer

class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=0, stop_text_token=1, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193,
                 train_solo_embeddings=False, use_mel_codes_as_input=True,
                 checkpointing=True, types=1, activation_function=None,
                 condition_num_latent=32, condition_type="perceiver", condition_module=None,
                 # ****** ADD VLLM related params ******
                 model_dir=None,
                 initialize_vllm=False,
                 vllm_gpu_memory_utilization=0.50,
                 vllm_tensor_parallel_size=1
                 # ************************************
                ):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
            condition_type: perceiver, gst or default encoder
            initialize_vllm (bool): If True, initialize the vLLM engine on creation.
            vllm_gpu_memory_utilization (float): GPU memory utilization for vLLM.
            vllm_tensor_parallel_size (int): Tensor parallel size for vLLM.
        """
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        # Adjust max lengths for positional embeddings to account for start/stop tokens
        # and potentially conditioning tokens if they occupy sequence positions
        self.max_mel_tokens = max_mel_tokens # Max *generated* mel tokens
        self.max_text_tokens = max_text_tokens # Max *input* text tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs # Max *number* of conditioning clips, not tokens
        self.cond_num_latents = condition_num_latent # Number of latent vectors from conditioning
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.use_mel_codes_as_input = use_mel_codes_as_input
        self.types = types # For multi-speaker type embedding expansion

        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num_latents, 0), True) # Pad mask for perceiver

        # --- Conditioning Encoder ---
        if condition_type == "perceiver":
            # Input spec_dim is 100 for the Perceiver conditioning encoder
            self.conditioning_encoder = ConditioningEncoder(100, model_dim, num_attn_heads=heads)
            # Output of conditioning_encoder is (B, S_cond, D_model)
            # Perceiver takes context_dim=model_dim
            self.perceiver_encoder = PerceiverResampler(dim=model_dim, depth=2, dim_context=model_dim, num_latents=self.cond_num_latents)
        elif condition_type == "conformer_perceiver" or condition_type == "conformer_encoder":
             assert condition_module is not None, "condition_module config needed for conformer"
             # Input is (B, S_spec, D_spec=100)
             self.conditioning_encoder = ConformerEncoder(input_size=100,
                                                          output_size=condition_module['output_size'],
                                                          linear_units=condition_module['linear_units'],
                                                          attention_heads=condition_module['attention_heads'],
                                                          num_blocks=condition_module['num_blocks'],
                                                          input_layer=condition_module['input_layer'])
             # Output is (B, S_conf, D_conf_out)
             if condition_type == "conformer_perceiver":
                 self.perceiver_encoder = PerceiverResampler(dim=model_dim, # Output dim of Perceiver is model_dim
                                                             depth=2,
                                                             dim_context=condition_module['output_size'], # Context is output of conformer
                                                             ff_mult=condition_module.get('perceiver_mult', 4),
                                                             heads=condition_module['attention_heads'],
                                                             num_latents=self.cond_num_latents)
                 # Output is (B, N_latent, D_model)
             else: # conformer_encoder only
                  # Need a projection from conformer output to model dim if different
                  if condition_module['output_size'] != model_dim:
                       self.cond_proj = nn.Linear(condition_module['output_size'], model_dim)
                  else:
                       self.cond_proj = nn.Identity()
                  self.perceiver_encoder = None # No perceiver used
                  # Output is (B, S_conf, D_model) after projection
                  # We might want to mean-pool or take first token? Assume mean for now.
                  # Will handle pooling/selection in get_conditioning
        else: # "default" or other simple encoder type (e.g., mean pooling)
            # Assume input spec_dim=100
            self.conditioning_encoder = ConditioningEncoder(100, model_dim, num_attn_heads=heads, mean=True)
            # Output is (B, D_model)
            self.perceiver_encoder = None

        # --- Embeddings ---
        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim) # +1 for stop token? Check usage
        if use_mel_codes_as_input:
            # Ensure number_mel_codes includes start/stop tokens if they have unique IDs
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            # MelEncoder expects input (B, mel_channels, S_mel), output (B, S_reduced, model_dim)
            self.mel_embedding = MelEncoder(model_dim, mel_channels=80, resblocks_per_reduction=1) # Assuming 80 mel channels

        # --- Transformer ---
        # Max length for pos embedding needs to account for ALL possible tokens in sequence
        # Rough estimate: cond_latents + text_tokens + start/stop + mel_tokens + start/stop
        # Use generous estimates for max lengths passed to build_hf_gpt_transformer
        max_pos_emb_len_mel = self.cond_num_latents + self.max_mel_tokens + 2 # Cond + Mel + Start/Stop
        max_pos_emb_len_text = self.max_text_tokens + 2 # Text + Start/Stop
        self.gpt, self.mel_pos_embedding, self.text_pos_embedding, _, _ = \
            build_hf_gpt_transformer(layers, model_dim, heads,
                                     max_pos_emb_len_mel, # Max positions needed *after* conditioning/text
                                     max_pos_emb_len_text, # Max positions needed for text part
                                     checkpointing, activation_function)

        # --- Solo Embeddings (Optional) ---
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        # --- Output Heads ---
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1) # Matching text embedding size
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes) # Matching mel embedding size

        # Initialize the embeddings per the GPT-2 scheme
        embeddings_to_init = [self.text_embedding]
        if use_mel_codes_as_input and isinstance(self.mel_embedding, nn.Embedding):
            embeddings_to_init.append(self.mel_embedding)
        for module in embeddings_to_init:
            module.weight.data.normal_(mean=0.0, std=.02)

        # --- VLLM Setup ---
        self.vllm_engine = None
        self.vllm_save_dir = os.path.join(model_dir, "vllm")
        self.initialize_vllm = initialize_vllm and _vllm_available
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = vllm_tensor_parallel_size

    def init_vllm(self):
        """Initializes the vLLM engine by saving weights and loading them."""
        if not _vllm_available:
            raise RuntimeError("vLLM library is not installed. Cannot initialize vLLM engine.")
        if self.vllm_engine is not None:
            logger.warning("vLLM engine already initialized.")
            return

        # 1. Create a temporary directory to save the model in HF format
        logger.info(f"Saving model components for vLLM to {self.vllm_save_dir}")

        # 2. Get the GPT2Config matching our transformer
        #    We need to ensure the vocab size used by vLLM's tokenizer/output mapping
        #    matches our *MEL* head output size.
        gpt_config = self.gpt.config
        # *** Crucially, set vocab_size to the target vocab size (mel codes) ***
        # vLLM uses this for the output layer dimension check and token mapping
        gpt_config.vocab_size = self.number_mel_codes
        # Adjust n_positions if necessary based on how vLLM handles it, but usually uses the config's value
        # gpt_config.n_positions = self.max_mel_tokens + self.max_text_tokens + self.cond_num_latents + 4 # A safe upper bound

        # 3. Save the transformer weights and config
        #    vLLM expects a standard HF model structure. We save the core `gpt` model.
        self.gpt.save_pretrained(self.vllm_save_dir, safe_serialization=False) # Use False for wider compat
        logger.info(f"GPT transformer saved to {self.vllm_save_dir}")

        # 4. Save the MEL head weights as 'lm_head.weight' and 'lm_head.bias' if it exists
        #    vLLM automatically looks for 'lm_head.weight' based on the config's vocab size.
        lm_head_state_dict = {
            "weight": self.mel_head.weight.detach().cpu().clone(),
        }
        if self.mel_head.bias is not None:
             lm_head_state_dict["bias"] = self.mel_head.bias.detach().cpu().clone()

        # Need to save it within the PyTorch model file expected by HF's from_pretrained
        torch.save(lm_head_state_dict, os.path.join(self.vllm_save_dir, "pytorch_model.bin")) # Overwrite/add to existing
        logger.info(f"MEL head weights saved as lm_head to {os.path.join(self.vllm_save_dir, 'pytorch_model.bin')}")

        # Save the final norm weights? vLLM might apply its own or expect it fused.
        # Generally, vLLM handles the final layer norm internally based on the model type.
        # We assume vLLM's GPT2 implementation includes the final LayerNorm before the lm_head.
        # If not, this might require a custom vLLM model runner or modifications.

        # 5. Initialize the vLLM engine
        logger.info(f"Loading vLLM engine with model: {self.vllm_save_dir}, "
                    f"TP size: {self.vllm_tensor_parallel_size}, "
                    f"GPU memory util: {self.vllm_gpu_memory_utilization}")
        # We need trust_remote_code=True because we are loading from a local, non-hub path
        # We pass the config's model type explicitly if needed.
        self.vllm_engine = LLM(
            model=self.vllm_save_dir,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            trust_remote_code=True, # Important for local paths/custom code
            # dtype='auto', # or 'half', 'bfloat16'
            # max_num_seqs=256, # Adjust batch size/concurrency if needed
        )
        logger.info("vLLM engine created successfully.")

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        """Initializes the HF inference model wrapper (for non-vLLM inference)."""
        logger.info("Initializing Hugging Face GPT2InferenceModel (for non-vLLM usage or comparison)")
        # Determine sequence length for HF config (needs to be large enough)
        seq_length = self.gpt.config.n_positions # Use length from core model's config

        # Create a config compatible with GPT2InferenceModel
        # Vocab size should match the *MEL* embedding/head size for this specific model
        gpt_config_inf = GPT2Config(
            vocab_size=self.number_mel_codes, # Set vocab size for the LM head
            n_positions=seq_length,
            # n_ctx=seq_length, # deprecated
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            activation_function=self.gpt.config.activation_function,
            # gradient_checkpointing=False, # Not used in inference wrapper
            use_cache=True, # Enable KV cache for HF inference
        )

        # Note: Passing self.mel_pos_embedding and self.mel_embedding here.
        # This was based on the original inference_speech logic which seemed to treat
        # the generated part as "text-like" using text_pos_embedding.
        # Let's align this with the vLLM approach which uses mel embeddings/head for generation.
        # We need a position embedding suitable for the MEL generation part.
        self.inference_model = GPT2InferenceModel(
            gpt_config_inf,
            self.gpt, # The core transformer
            self.mel_pos_embedding, # Positional embedding for the generated MEL tokens
            self.mel_embedding, # Token embedding lookup for MEL codes
            self.final_norm, # Final normalization layer
            self.mel_head, # The actual MEL prediction head
            kv_cache=kv_cache,
        )
        # Put model in eval mode
        self.inference_model = self.inference_model.eval()
        # If using half precision with HF model
        if half:
             self.inference_model = self.inference_model.half()

        # Assign mel_embedding to gpt.wte for potential internal use by HF generate, though we use embeds
        # self.gpt.wte = self.mel_embedding # This might cause issues if wte is used unexpectedly

        logger.info("Hugging Face GPT2InferenceModel initialized.")

        if self.initialize_vllm:
            logger.info("Initializing vLLM engine...")
            try:
                # Ensure model is on CPU before saving for vLLM to load onto its GPUs
                self.cpu()
                self.init_vllm()
                logger.info(f"vLLM engine initialized from {self.vllm_save_dir}")
            except Exception as e:
                logger.error(f"Failed to initialize vLLM: {e}", exc_info=True)
                self.vllm_engine = None
                self.initialize_vllm = False # Disable if init fails


    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        # Add start token at the beginning
        inp = F.pad(input, (1, 0), value=start_token)
        # Add stop token at the end for targets
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """Pad with stop_mel_token."""
        # (B, S_mel_padded)
        for b in range(mel_input_tokens.shape[0]):
            actual_end = mel_lengths[b] # Length includes the actual tokens
            if actual_end < mel_input_tokens.shape[1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """Pad with stop_text_token."""
         # (B, S_text_padded)
        for b in range(text_input_tokens.shape[0]):
            actual_end = text_lengths[b] # Length includes the actual tokens
            if actual_end < text_input_tokens.shape[1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(self, speech_conditioning_latent, first_inputs_embeds, first_head, second_inputs_embeds=None, second_head=None, get_attns=False, return_latent=False):
        """
        Unified forward pass through the transformer for training/evaluation.
        Assumes inputs are already embedded.
        """
        if second_inputs_embeds is not None:
            # Concatenate along sequence dim: (B, S_cond, D), (B, S1, D), (B, S2, D) -> (B, S_cond+S1+S2, D)
            emb = torch.cat([speech_conditioning_latent, first_inputs_embeds, second_inputs_embeds], dim=1)
        else:
            emb = torch.cat([speech_conditioning_latent, first_inputs_embeds], dim=1)

        # Pass concatenated embeddings through the transformer
        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)

        if get_attns:
            return gpt_out.attentions # Return attention weights if requested

        # Get hidden states, excluding the conditioning latent part
        # Shape: (B, S_cond+S1[+S2], D)
        hidden_states = gpt_out.last_hidden_state
        offset = speech_conditioning_latent.shape[1]
        # Shape: (B, S1[+S2], D)
        logits_hidden_states = hidden_states[:, offset:]

        # Apply final normalization
        logits_hidden_states = self.final_norm(logits_hidden_states)

        # --- Calculate logits for the first part ---
        len1 = first_inputs_embeds.shape[1]
        first_logits_hidden = logits_hidden_states[:, :len1] # (B, S1, D)
        if return_latent:
            first_result = first_logits_hidden # Return latent representations
        else:
            first_logits = first_head(first_logits_hidden) # (B, S1, V1)
            first_result = first_logits.permute(0, 2, 1) # (B, V1, S1) for CrossEntropyLoss

        # --- Calculate logits for the second part (if provided) ---
        if second_inputs_embeds is not None:
            len2 = second_inputs_embeds.shape[1]
            second_logits_hidden = logits_hidden_states[:, len1:len1+len2] # (B, S2, D)
            if return_latent:
                 second_result = second_logits_hidden # Return latent representations
            else:
                 second_logits = second_head(second_logits_hidden) # (B, S2, V2)
                 second_result = second_logits.permute(0, 2, 1) # (B, V2, S2) for CrossEntropyLoss
            return first_result, second_result
        else:
            return first_result

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        """
        Computes the conditioning embeddings based on the configured method.

        Args:
            speech_conditioning_input: (B, D_spec, S_spec) or (B, N_clips, D_spec, S_spec)
            cond_mel_lengths: (B,) Optional lengths for masking if using conformer.

        Returns:
            conds: (B, N_latent or S_cond, D_model) Conditioning embeddings.
        """
        # Ensure input is (B, D_spec, S_spec)
        if speech_conditioning_input.ndim == 4:
            # If multiple clips provided (B, N, D, S), process the first one for now
            # TODO: Handle multiple conditioning inputs properly (e.g., mean, concat?)
            logger.warning("Multiple conditioning clips provided, only using the first.")
            speech_conditioning_input = speech_conditioning_input[:, 0] # Take the first clip: (B, D_spec, S_spec)

        # Transpose to (B, S_spec, D_spec) for most encoders
        speech_conditioning_input = speech_conditioning_input.transpose(1, 2)

        if self.condition_type == "perceiver":
             # ConditioningEncoder expects (B, D_spec, S_spec), output (B, S_cond, D_model)
            cond_encoded = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2))
             # Perceiver expects (B, S_cond, D_model), output (B, N_latent, D_model)
            conds = self.perceiver_encoder(cond_encoded)
        elif self.condition_type == "conformer_encoder" or self.condition_type == "conformer_perceiver":
             # ConformerEncoder expects (B, S_spec, D_spec), lengths (B,)
             # Output: hidden_states (B, S_conf, D_conf_out), mask (B, 1, S_conf)
            cond_encoded, mask = self.conditioning_encoder(speech_conditioning_input, cond_mel_lengths)

            if self.condition_type == "conformer_perceiver":
                 # Perceiver expects context (B, S_conf, D_conf_out) and mask (B, S_conf)
                 # mask needs to be (B, N_latent + S_conf) for cross-attention
                 conds_mask = self.cond_mask_pad(mask.squeeze(1)) # (B, N_latent + S_conf)
                 conds = self.perceiver_encoder(context=cond_encoded, mask=conds_mask) # (B, N_latent, D_model)
            else: # conformer_encoder only
                 cond_encoded = self.cond_proj(cond_encoded) # Project to D_model: (B, S_conf, D_model)
                 # Use mean pooling over valid time steps based on mask
                 mask_expanded = mask.transpose(1, 2).float() # (B, S_conf, 1)
                 conds_sum = (cond_encoded * mask_expanded).sum(dim=1) # (B, D_model)
                 valid_lengths = mask_expanded.sum(dim=1) # (B, 1)
                 conds = conds_sum / torch.clamp(valid_lengths, min=1.0) # (B, D_model)
                 # Add sequence dim for compatibility: (B, 1, D_model)
                 conds = conds.unsqueeze(1)

        else: # Default mean-pooling encoder
             # ConditioningEncoder expects (B, D_spec, S_spec), mean=True gives (B, D_model)
            conds = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2))
             # Add sequence dim: (B, 1, D_model)
            conds = conds.unsqueeze(1)

        # conds should be (B, S_cond_effective, D_model)
        # S_cond_effective is N_latent for perceiver types, 1 for mean pooling
        return conds

    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, wav_lengths,
                cond_mel_lengths=None, types=None, text_first=True, raw_mels=None, return_attentions=False,
                return_latent=False, clip_inputs=False):
        """
        Forward pass for training or evaluation.

        Args:
            speech_conditioning_latent: (B, D_spec, S_spec) or (B, N_clips, D_spec, S_spec) Raw conditioning input (e.g., Mel spectrogram).
            text_inputs: (B, S_text_padded) Long tensor of text token IDs.
            text_lengths: (B,) Long tensor of actual text lengths.
            mel_codes: (B, S_mel_padded) Long tensor of mel code IDs.
            wav_lengths: (B,) Long tensor of actual waveform lengths (used to derive mel lengths).
            cond_mel_lengths: (B,) Optional. Actual lengths of conditioning spectrograms if needed by encoder.
            types: (B,) Optional. Speaker/style type IDs for embedding modulation.
            text_first (bool): Whether text comes before MEL in the transformer sequence.
            raw_mels: (B, D_mel, S_mel_raw) Optional. Raw Mel spectrograms if `use_mel_codes_as_input=False`.
            return_attentions (bool): If True, return attention weights instead of logits/loss.
            return_latent (bool): If True, return latent representations instead of logits/loss.
            clip_inputs (bool): If True, clip padded inputs to max actual length in the batch.

        Returns:
            if return_attentions: Attention weights.
            if return_latent: Latent tensors (mel_latents or text_latents depending on text_first).
            else: loss_text, loss_mel, mel_logits (for potential metrics).
        """

        device = text_inputs.device
        # --- Conditioning ---
        # Shape: (B, S_cond, D_model)
        conds = self.get_conditioning(speech_conditioning_latent.to(device), cond_mel_lengths.to(device) if cond_mel_lengths is not None else None)

        # --- Input Clipping (Optional) ---
        if clip_inputs:
            max_text_len = text_lengths.max()
            text_inputs = text_inputs[:, :max_text_len]
            # Calculate max mel length based on wav lengths and compression
            # Add 1 for the extra token prediction as in the original code
            max_mel_len = (torch.ceil(wav_lengths.float() / self.mel_length_compression).long() + 1).max()
            mel_codes = mel_codes[:, :max_mel_len]
            if raw_mels is not None:
                # Raw mels might have different compression, adjust accordingly if needed
                # Assuming MelEncoder reduction is 4
                max_raw_mel_len = max_mel_len * self.mel_embedding.reduction if not self.use_mel_codes_as_input else max_mel_len
                raw_mels = raw_mels[:, :, :max_raw_mel_len]

        # --- Prepare Text Inputs ---
        # Apply type modulation if needed
        if types is not None:
            type_offset = self.number_text_tokens * types.unsqueeze(-1) # (B, 1)
            text_inputs = text_inputs + type_offset
        # Pad text inputs appropriately
        text_inputs = self.set_text_padding(text_inputs, text_lengths) # Pad with stop_text_token
        # Add start/stop tokens for aligned input/target creation
        # Input: [start, t1, t2, ..., tn, stop]
        # Target: [t1, t2, ..., tn, stop, stop] (predict next token including stop)
        text_inputs_aligned, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        # Embed text tokens
        # Shape: (B, S_text_aligned, D_model)
        text_emb = self.text_embedding(text_inputs_aligned)
        # Add positional embeddings for the text part
        # Shape depends only on seq length: (S_text_aligned, D_model) -> broadcast add
        text_emb = text_emb + self.text_pos_embedding(text_inputs_aligned) + self.text_solo_embedding

        # --- Prepare Mel Inputs ---
        # Calculate mel lengths from wav lengths
        # +1 matches original logic, potentially predicting one token past the end
        mel_codes_lengths = torch.ceil(wav_lengths.float() / self.mel_length_compression).long() + 1
        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths) # Pad with stop_mel_token
        # Add start/stop tokens
        # Input: [start, m1, m2, ..., mk, stop]
        # Target: [m1, m2, ..., mk, stop, stop]
        mel_codes_aligned, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)

        # Embed mel tokens/features
        if self.use_mel_codes_as_input:
            # Shape: (B, S_mel_aligned, D_model)
             mel_emb = self.mel_embedding(mel_codes_aligned)
        else:
            # MelEncoder expects (B, D_mel, S_mel), need to prepare input appropriately
            # Assume raw_mels is (B, D_mel, S_mel_raw), pad it to match mel_codes_aligned length if needed
            # This part needs careful alignment based on MelEncoder's expected input/output shapes & reduction
            if raw_mels is None:
                raise ValueError("raw_mels required when use_mel_codes_as_input is False")
            # Pad raw_mels like mel_codes were padded before alignment? Requires thought.
            # Simplified: Assume MelEncoder handles padding or input matches aligned length
            # Input shape: (B, D_mel, S_mel_aligned * reduction_factor?) -> Needs check
            # Output shape: (B, S_mel_aligned, D_model)
            mel_emb = self.mel_embedding(raw_mels) # Pass raw mels to the encoder
            # Need to ensure output sequence length matches mel_codes_aligned
            if mel_emb.shape[1] != mel_codes_aligned.shape[1]:
                 # This indicates a mismatch in length handling between MelEncoder and token alignment
                 # Might need interpolation or adjustments to mel_codes_lengths calculation
                 logger.warning(f"Mel embedding seq length ({mel_emb.shape[1]}) != target seq length ({mel_codes_aligned.shape[1]}). Check MelEncoder reduction and length calculations.")
                 # Simple truncation/padding:
                 target_len = mel_codes_aligned.shape[1]
                 if mel_emb.shape[1] > target_len:
                      mel_emb = mel_emb[:, :target_len, :]
                 else: # Pad if shorter
                      padding_size = target_len - mel_emb.shape[1]
                      mel_emb = F.pad(mel_emb, (0, 0, 0, padding_size))


        # Add positional embeddings for the mel part
        # Shape depends only on seq length: (S_mel_aligned, D_model) -> broadcast add
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes_aligned) + self.mel_solo_embedding


        # --- Run through Transformer ---
        if text_first:
            # Input order: [conds, text_emb, mel_emb]
            results = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head,
                                       get_attns=return_attentions, return_latent=return_latent)
            if return_attentions: return results # Return attention weights
            if return_latent: return results[1] # Return mel latents (second part)
            text_logits, mel_logits = results # (B, V_text, S_text), (B, V_mel, S_mel)
        else:
            # Input order: [conds, mel_emb, text_emb]
            results = self.get_logits(conds, mel_emb, self.mel_head, text_emb, self.text_head,
                                      get_attns=return_attentions, return_latent=return_latent)
            if return_attentions: return results # Return attention weights
            if return_latent: return results[1] # Return text latents (second part)
            mel_logits, text_logits = results # (B, V_mel, S_mel), (B, V_text, S_text)

        # --- Calculate Losses ---
        # Ensure targets are long tensors
        text_targets = text_targets.long()
        mel_targets = mel_targets.long()

        # Text loss: compares predictions (B, V_text, S_text) with targets (B, S_text)
        loss_text = F.cross_entropy(text_logits, text_targets, ignore_index=self.stop_text_token) # Ignore padding? Check if stop token should be ignored

        # Mel loss: compares predictions (B, V_mel, S_mel) with targets (B, S_mel)
        loss_mel = F.cross_entropy(mel_logits, mel_targets, ignore_index=self.stop_mel_token) # Ignore padding? Check if stop token should be ignored

        # Return losses and mel_logits (useful for metrics like accuracy)
        return loss_text.mean(), loss_mel.mean(), mel_logits


    # --- Original HuggingFace-based Inference ---
    def inference_speech(self, speech_conditioning_latent, text_inputs, text_lengths=None, # Added text_lengths
                         cond_mel_lengths=None, input_tokens=None, num_return_sequences=1,
                         max_generate_length=None, typical_sampling=False, typical_mass=.9, **hf_generate_kwargs):
        """
        Original inference method using Hugging Face generate.
        Requires post_init_gpt2_config to be called first.
        """
        if self.inference_model is None:
            raise RuntimeError("Run `post_init_gpt2_config()` first to initialize the Hugging Face inference model.")
        # if typical_sampling:
        #     if 'logits_processor' in hf_generate_kwargs:
        #          hf_generate_kwargs['logits_processor'].append(TypicalLogitsWarper(mass=typical_mass))
        #     else:
        #          hf_generate_kwargs['logits_processor'] = LogitsProcessorList([TypicalLogitsWarper(mass=typical_mass)])

        device = next(self.parameters()).device
        self.inference_model.to(device) # Ensure HF model is on correct device

        # --- Prepare Conditioning and Text Prefix ---
        with torch.no_grad():
            # 1. Get Conditioning Embeddings
            # Shape: (B, S_cond, D_model)
            conds = self.get_conditioning(speech_conditioning_latent.to(device),
                                        cond_mel_lengths.to(device) if cond_mel_lengths is not None else None)

            # 2. Prepare Text Inputs
            # Pad text, add start/stop tokens
            if text_lengths is None:
                 text_lengths = torch.tensor([text_inputs.shape[1]] * text_inputs.shape[0], device=device) # Assume full length if not provided
            text_inputs = self.set_text_padding(text_inputs.to(device), text_lengths.to(device))
            text_inputs_aligned, _ = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)

            # 3. Embed Text Tokens
            text_emb = self.text_embedding(text_inputs_aligned)
            text_emb = text_emb + self.text_pos_embedding(text_inputs_aligned) + self.text_solo_embedding # (B, S_text_aligned, D)

            # 4. Combine Conditioning and Text Embeddings -> Prefix `emb`
            # Shape: (B, S_cond + S_text_aligned, D_model)
            emb = torch.cat([conds, text_emb], dim=1)

            # Store prefix embeddings in the HF inference model (using its cache mechanism)
            self.inference_model.store_mel_emb(emb)

            # --- Prepare Input IDs for HF Generate ---
            # `generate` expects token IDs. We create "fake" inputs that correspond
            # in shape to our prefix embedding `emb`. The actual embeddings will be
            # used via the `cached_mel_emb` and the custom forward pass.
            prefix_len = emb.shape[1]
            batch_size = emb.shape[0]

            # Need to provide the *last* token ID of the prefix to start generation.
            # This should be the *start_mel_token*.
            # Create input_ids: (B, prefix_len + 1)
            # Fill with placeholder (e.g., 0), then set the last one to start_mel_token
            fake_input_ids = torch.zeros((batch_size, prefix_len + 1), dtype=torch.long, device=device)
            fake_input_ids[:, -1] = self.start_mel_token

            start_token_len = 1 # Length of the starting token(s) we provide

            # Handle provided input_tokens (autoregressive continuation)
            if input_tokens is not None:
                 input_tokens = input_tokens.to(device)
                 # Repeat prefix embeddings and fake inputs if num_return_sequences > batch_size
                 if num_return_sequences > batch_size:
                      assert num_return_sequences % batch_size == 0, "num_return_sequences must be a multiple of batch size"
                      num_repeats = num_return_sequences // batch_size
                      emb = emb.repeat_interleave(num_repeats, dim=0)
                      fake_input_ids = fake_input_ids.repeat_interleave(num_repeats, dim=0)
                      self.inference_model.store_mel_emb(emb) # Update cache with repeated embeds

                 # Repeat input_tokens if necessary
                 if input_tokens.shape[0] != num_return_sequences:
                      assert num_return_sequences % input_tokens.shape[0] == 0
                      num_repeats_tokens = num_return_sequences // input_tokens.shape[0]
                      input_tokens = input_tokens.repeat_interleave(num_repeats_tokens, dim=0)

                 # Concatenate fake prefix IDs with the provided input tokens
                 # fake_input_ids shape: (B * num_repeats, prefix_len + 1)
                 # input_tokens shape: (B * num_repeats, S_in)
                 # We need to remove the start_mel_token from fake_input_ids if it's also the start of input_tokens
                 if fake_input_ids[0, -1] == input_tokens[0, 0]:
                      inputs = torch.cat([fake_input_ids[:, :-1], input_tokens], dim=1)
                 else:
                      # This case is less common, implies input_tokens doesn't start with start_mel_token
                      inputs = torch.cat([fake_input_ids, input_tokens], dim=1)

                 current_len = inputs.shape[1]

            else:
                 # If no input_tokens provided, start generation right after the prefix + start_mel_token
                 inputs = fake_input_ids
                 # Repeat prefix embeddings and inputs if num_return_sequences > 1
                 if num_return_sequences > 1:
                      assert num_return_sequences % batch_size == 0, "num_return_sequences must be multiple of batch size"
                      num_repeats = num_return_sequences // batch_size
                      emb = emb.repeat_interleave(num_repeats, dim=0)
                      inputs = inputs.repeat_interleave(num_repeats, dim=0)
                      self.inference_model.store_mel_emb(emb) # Update cache

                 current_len = inputs.shape[1]


            # --- Determine Max Generation Length ---
            # Max length for HF generate is the *total* length (prefix + generated)
            gen_len = max_generate_length if max_generate_length is not None else self.max_mel_tokens
            max_len_hf = prefix_len + gen_len # Target total sequence length

            # Ensure max_len doesn't exceed model's positional embedding capacity
            max_pos_embed = self.inference_model.config.n_positions
            if max_len_hf > max_pos_embed:
                 logger.warning(f"Requested max_length ({max_len_hf}) exceeds model's position embedding size ({max_pos_embed}). Truncating.")
                 max_len_hf = max_pos_embed

            # --- Generate with Hugging Face model ---
            # We pass `inputs` which includes the fake prefix IDs + start_mel_token [+ input_tokens]
            # The custom forward pass inside GPT2InferenceModel uses `cached_mel_emb`
            gen = self.inference_model.generate(
                inputs=inputs, # Contains fake prefix + start token + maybe more input tokens
                # We don't need input_embeds here because the forward pass handles it
                max_length=max_len_hf,
                eos_token_id=self.stop_mel_token,
                pad_token_id=self.stop_mel_token, # Treat padding same as EOS for mel
                # bos_token_id=self.start_mel_token, # Usually not needed if start token is in inputs
                num_return_sequences=num_return_sequences, # Already handled by repeating inputs/embeds
                **hf_generate_kwargs,
            )

            # --- Process Output ---
            # `gen` contains the full sequence: [fake_prefix_ids, start_mel_token, generated_mel_tokens, stop_mel_token/pad]
            # We need to return only the generated MEL tokens, excluding the prefix and start token.
            # The starting point of actual generation in `gen` is after the initial `inputs`.
            start_gen_index = current_len
            generated_tokens = gen[:, start_gen_index:]

            # Clean up stop/pad tokens at the end if needed
            # (Optional, depends on whether consumer expects them)
            # Example: Find first stop token and truncate
            clean_tokens = []
            for seq in generated_tokens:
                stop_idx = (seq == self.stop_mel_token).nonzero(as_tuple=True)[0]
                if len(stop_idx) > 0:
                    clean_tokens.append(seq[:stop_idx[0]])
                else:
                    clean_tokens.append(seq) # No stop token found

        # Return list of tensors (or pad them back to tensor if required)
        # For now, return the raw output after slicing the prefix
        return generated_tokens # Shape (num_return_sequences, generated_seq_len)


    # --- NEW Asynchronous vLLM Inference ---
    async def inference_speech_vllm(
        self,
        speech_conditioning_latent: torch.Tensor,
        text_inputs: torch.Tensor,
        text_lengths: torch.Tensor = None, # Added text_lengths
        cond_mel_lengths: torch.Tensor = None,
        num_return_sequences: int = 1,
        max_generate_tokens: int = None,
        # VLLM Sampling Params (map from hf_generate_kwargs and typical_sampling)
        temperature: float = 0.8,
        top_p: float = 1.0, # Set to 1.0 to disable top-p by default
        top_k: int = -1, # Set to -1 to disable top-k by default
        use_beam_search: bool = False, # vLLM also supports beam search
        best_of: int = None, # For beam search or stochastic sampling
        length_penalty: float = 1.0, # For beam search
        stop_token_ids: list[int] = None,
        **kwargs # Catch any other unused kwargs
        ) -> list[list[int]]:
        """
        Performs asynchronous speech generation using the vLLM engine.

        Args:
            speech_conditioning_latent: (B, D_spec, S_spec) Conditioning input.
            text_inputs: (B, S_text) Text token IDs.
            text_lengths: (B,) Actual lengths of text inputs.
            cond_mel_lengths: (B,) Optional lengths for conditioning encoder.
            num_return_sequences: Number of sequences to generate per input.
            max_generate_tokens: Maximum number of *new* MEL tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p nucleus sampling probability.
            top_k: Top-k sampling K.
            use_beam_search: Whether to use beam search.
            best_of: Number of sequences to generate internally (beam width for beam search).
            length_penalty: Length penalty for beam search.
            stop_token_ids: Optional list of token IDs to stop generation. Defaults to [stop_mel_token].
            kwargs: Ignored.

        Returns:
            A list of lists, where each inner list contains the generated MEL token IDs
            for one sequence (batch_size * num_return_sequences total sequences).
            Returns None if vLLM is not available or not initialized.
        """
        if not _vllm_available or self.vllm_engine is None:
            logger.error("vLLM is not available or not initialized. Cannot run vLLM inference.")
            return None

        device = next(self.parameters()).device # Get current device (likely CPU after init)
        self.to(device) # Ensure model parts (embeddings etc) are on the right device if needed

        batch_size = speech_conditioning_latent.shape[0]
        if max_generate_tokens is None:
            max_generate_tokens = self.max_mel_tokens
        if stop_token_ids is None:
            stop_token_ids = [self.stop_mel_token]

        # --- Prepare Prefix Embeddings (on the correct device) ---
        with torch.no_grad():
            # 1. Get Conditioning Embeddings
            # Shape: (B, S_cond, D_model)
            conds = self.get_conditioning(speech_conditioning_latent.to(device),
                                        cond_mel_lengths.to(device) if cond_mel_lengths is not None else None)

            # 2. Prepare Text Inputs
            if text_lengths is None:
                 text_lengths = torch.tensor([text_inputs.shape[1]] * text_inputs.shape[0], device=device)
            text_inputs = self.set_text_padding(text_inputs.to(device), text_lengths.to(device))
            # Add start/stop but only need the input part for embedding
            text_inputs_aligned, _ = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)

            # 3. Embed Text Tokens
            text_emb = self.text_embedding(text_inputs_aligned)
            text_emb = text_emb + self.text_pos_embedding(text_inputs_aligned) + self.text_solo_embedding

            # 4. Combine Conditioning and Text Embeddings -> Prefix `emb`
            # Shape: (B, S_prefix = S_cond + S_text_aligned, D_model)
            prefix_embeddings = torch.cat([conds, text_emb], dim=1)

            # 5. Prepare dummy token IDs for the prefix shape (required by vLLM for prompt_embeds)
            prefix_len = prefix_embeddings.shape[1]
            # Use a non-special token ID like 0, assuming it's unused or benign
            prefix_token_ids = torch.zeros((batch_size, prefix_len), dtype=torch.long, device=device)

            # 6. Add the start_mel_token ID to the *end* of the prefix_token_ids
            # This tells vLLM what the *next* token to predict is after the embeds
            start_mel_tensor = torch.full((batch_size, 1), self.start_mel_token, dtype=torch.long, device=device)
            prompt_token_ids_with_start = torch.cat([prefix_token_ids, start_mel_tensor], dim=1)


        # --- Prepare vLLM Sampling Parameters ---
        # Handle num_return_sequences vs best_of
        if use_beam_search:
             if best_of is None:
                 best_of = num_return_sequences
             sampling_params = SamplingParams(
                 n=num_return_sequences, # Number of sequences to return
                 best_of=best_of,        # Beam width
                 use_beam_search=True,
                 length_penalty=length_penalty,
                 temperature=0, # Beam search typically uses temperature 0
                 top_p=1.0,
                 top_k=-1,
                 max_tokens=max_generate_tokens,
                 stop_token_ids=stop_token_ids,
                 # skip_special_tokens=False, # Keep special tokens like stop
             )
        else: # Nucleus/Top-K sampling
             if best_of is None:
                 best_of = num_return_sequences # Sample `best_of` sequences, return `n`
             sampling_params = SamplingParams(
                 n=num_return_sequences,
                 best_of=best_of, # Sample `best_of` sequences if > n
                 temperature=temperature,
                 top_p=top_p,
                 top_k=top_k,
                 max_tokens=max_generate_tokens,
                 stop_token_ids=stop_token_ids,
                 # skip_special_tokens=False,
             )

        # --- Add requests to vLLM ---
        # vLLM expects lists of inputs
        # Ensure embeddings are on CPU or correct device format expected by vLLM (usually CUDA tensors work)
        # vLLM handles batching internally.
        request_id_counter = 0
        requests = []
        for i in range(batch_size):
            request_id = str(request_id_counter)
            requests.append(
                (
                 request_id,
                 { # prompt_embeds dict format
                     "prompt_embeds": prefix_embeddings[i].unsqueeze(0), # Shape (1, S_prefix, D)
                     "prompt_token_ids": prompt_token_ids_with_start[i].unsqueeze(0) # Shape (1, S_prefix + 1)
                 },
                 sampling_params
                 )
            )
            # Add the request using the low-level API to handle prompt_embeds correctly
            # Note: This API might change slightly between vLLM versions. Refer to docs.
            # Using generate directly should now support prompt_embeds dictionary
            # self.vllm_engine.add_request(
            #     request_id=request_id,
            #     inputs={
            #         "prompt_embeds": prefix_embeddings[i].unsqueeze(0), # Shape (1, S_prefix, D)
            #         "prompt_token_ids": prompt_token_ids_with_start[i].unsqueeze(0) # Shape (1, S_prefix + 1)
            #     },
            #     params=sampling_params
            # )
            request_id_counter += 1


        # --- Generate asynchronously using generate ---
        # generate takes 'prompts' (text) or 'prompt_token_ids', or 'inputs' dict
        vllm_inputs = [{ "prompt_embeds": prefix_embeddings[i], "prompt_token_ids": prompt_token_ids_with_start[i] } for i in range(batch_size)]

        outputs = await self.vllm_engine.generate(inputs=vllm_inputs, sampling_params=sampling_params)

        # --- Process vLLM Outputs ---
        # Outputs is a list of RequestOutput objects
        all_generated_tokens = []
        for output in outputs:
            # Each output corresponds to one prompt in the batch
            # output.outputs is a list of CompletionOutput objects (one for each sequence in n)
            for completion in output.outputs:
                 # completion.token_ids contains the *generated* tokens (excluding the prompt)
                 all_generated_tokens.append(completion.token_ids)

        # Expected output: list (size B * n) of lists of ints
        return all_generated_tokens

    # --- Helper to run the async vLLM inference ---
    def inference_speech_vllm_sync(self, *args, **kwargs):
        """Synchronous wrapper for the async vLLM inference."""
        if not _vllm_available or self.vllm_engine is None:
            logger.error("vLLM not available or initialized.")
            return None
        # If running in an existing event loop, just await
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.inference_speech_vllm(*args, **kwargs))
        except RuntimeError: # No running event loop
            # Create a new event loop to run the async function
            return asyncio.run(self.inference_speech_vllm(*args, **kwargs))