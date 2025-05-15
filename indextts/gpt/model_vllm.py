import uuid
import os
import patch_vllm  # ⚠️ Monkey Patch, do not delete this line

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import (assert_device_map,
                                                     get_device_map)
from transformers import GPT2Config, GPT2Model

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.arch_util import AttentionBlock
from indextts.utils.typical_sampling import TypicalLogitsWarper

from vllm import AsyncLLMEngine, SamplingParams, TokensPrompt
from vllm.engine.arg_utils import AsyncEngineArgs
import asyncio


def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)
    

class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=0, stop_text_token=1, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193,
                 types=1, activation_function=None,
                 model_dir=None,
                 condition_num_latent=32, condition_module=None, **kwargs):
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
            checkpointing:
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
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)

        self.conditioning_encoder = ConformerEncoder(input_size=100,
                                                        output_size=condition_module['output_size'],
                                                        linear_units=condition_module['linear_units'],
                                                        attention_heads=condition_module['attention_heads'],
                                                        num_blocks=condition_module['num_blocks'],
                                                        input_layer=condition_module['input_layer'])
        self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=condition_module['output_size'],
                                                    ff_mult=condition_module['perceiver_mult'],
                                                    heads=condition_module['attention_heads'],
                                                    num_latents=self.cond_num)

        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)

        max_mel_seq_len = self.max_mel_tokens + 2 + self.max_conditioning_inputs
        max_text_seq_len = self.max_text_tokens + 2
        self.mel_pos_embedding, self.text_pos_embedding  = LearnedPositionEmbeddings(max_mel_seq_len, model_dim), LearnedPositionEmbeddings(max_text_seq_len, model_dim)

        self.mel_solo_embedding = 0
        self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)  # , dtype=torch.float16
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding, self.mel_embedding]
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)

        # init vllm engine
        vllm_dir = os.path.join(model_dir, "vllm")
        engine_args = AsyncEngineArgs(
            model=vllm_dir,
            tensor_parallel_size=1,
            dtype="auto",
            gpu_memory_utilization=0.25,
            # enforce_eager=True,
        )
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(
            temperature=1.0,
            top_p=0.8,
            top_k=30,  # 5, 30
            repetition_penalty=10.0,  # 8.0
            max_tokens=768,  # 605
        )

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                    cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.cond_mask_pad(mask.squeeze(1))
        conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 32, d)
        return conds

    async def inference_speech(self, speech_conditioning_latent, text_inputs, cond_mel_lengths=None):

        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)
        text_inputs, _ = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)

        # speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent, cond_mel_lengths)
        emb = torch.cat([speech_conditioning_latent, text_emb], dim=1)
        trunc_index = emb.shape[1] + 1

        mel_start_emb = self.mel_embedding(torch.full((emb.shape[0], 1,), fill_value=self.start_mel_token, dtype=torch.long, device=text_inputs.device))
        mel_start_emb = mel_start_emb + self.mel_pos_embedding(mel_start_emb)
        inputs_embeds = torch.cat([emb, mel_start_emb], dim=1)

        fake_inputs = [idx for idx in range(inputs_embeds.shape[1])]
        multi_modal_data = {"image": inputs_embeds}
        tokens_prompt = TokensPrompt(prompt_token_ids=fake_inputs, multi_modal_data=multi_modal_data)
        output_generator = self.llm.generate(tokens_prompt, sampling_params=self.sampling_params, request_id=uuid.uuid4())
        latent = []
        async for output in output_generator:
            latent.append(output.hidden_states.clone())
        codes = output.outputs[0].token_ids[:-2]

        latent = torch.cat(latent[:-2], dim=0).unsqueeze(0)
        # latent = self.final_norm(latent.float())
        latent = latent.float()
        print("codes", len(codes), codes)
        print("latent", latent.shape, latent)
        return codes, latent  # [:, trunc_index:]
