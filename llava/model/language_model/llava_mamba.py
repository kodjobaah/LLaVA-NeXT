from typing import List, Optional, Tuple, Union

import torch

# Import the Mamba components directly from mamba_ssm
from mamba_ssm.models.mixer_seq_simple import MambaConfig, MambaLMHeadModel, MixerModel
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.model.llava_arch import LlavaMetaForCausalLM, LlavaMetaModel

# Also import RMSNorm and related functions if needed for manual forward pass
try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    print(
        "Warning: Triton LayerNorm / RMSNorm kernels not found. Some functionality might be limited."
    )


class LlavaMambaConfig(MambaConfig):  # Inherit from Mamba's config
    model_type = "llava_mamba"
    # Mamba models don't use rope_scaling, so explicitly set it to None.
    rope_scaling: Optional[dict] = None


class LlavaMambaModel(
    LlavaMetaModel, MixerModel
):  # LlavaMetaModel for multimodal, MixerModel for Mamba backbone
    config_class = LlavaMambaConfig

    def __init__(self, config: MambaConfig):
        super().__init__(config)
        # MixerModel's __init__ will set up self.embedding, self.layers, self.norm_f.

    def embed_tokens(self, x):
        """
        Exposes the embedding layer of MixerModel, as expected by LlavaMetaModel.
        MixerModel's embedding layer is named 'embedding'.
        """
        return self.embedding(x)


class LlavaMambaForCausalLM(
    MambaLMHeadModel, LlavaMetaForCausalLM
):  # Inherit from MambaLMHeadModel for Mamba's Causal LM, and LlavaMetaForCausalLM for multimodal capabilities
    config_class = LlavaMambaConfig
    supports_gradient_checkpointing = True  # Standard Hugging Face attribute for models supporting gradient checkpointing.

    def __init__(self, config: MambaConfig):
        # Initialize MambaLMHeadModel first.
        # This will set up self.backbone (an instance of MixerModel) and self.lm_head.
        super().__init__(config)

        # Explicitly set self.model to point to the Mamba backbone (MixerModel instance).
        # LlavaMetaModel expects the core model to be at self.model for multimodal processing.
        # MambaLMHeadModel stores it at self.backbone.
        self.model = self.backbone

        # self.lm_head is already set by MambaLMHeadModel's __init__.
        # No need to re-assign or re-create it unless custom initialization is required.

        # self.post_init() is handled by the `transformers.PreTrainedModel`
        # parent class of `MambaLMHeadModel` (which is an alias of `MambaForCausalLM`).

    def get_model(self):
        # This method is used by LlavaMetaForCausalLM to get the core model backbone.
        # It returns our `LlavaMambaModel` instance, which is stored in `self.model`.
        return self.model

    def _set_gradient_checkpointing(self, module, value=False):
        """
        Configures gradient checkpointing.
        Mamba's `Block` (layers within MixerModel) supports a form of checkpointing.
        This function is primarily for compatibility with Hugging Face's `PreTrainedModel` system.
        """
        # Mamba's Blocks internally handle checkpointing. We don't typically set a global flag
        # on the entire MixerModel for this in the `mamba_ssm` library.
        # If specific fine-grained control is needed, it would be within the `Block` forward pass.
        # For HF compatibility, this can remain a placeholder or be removed if it causes issues.
        pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[
            torch.Tensor
        ] = None,  # Mamba models do not use attention_mask for their core computation
        position_ids: Optional[
            torch.LongTensor
        ] = None,  # Mamba models do not use position_ids
        past_key_values: Optional[
            List[torch.FloatTensor]
        ] = None,  # Mamba uses `inference_params` for state management
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[
            bool
        ] = None,  # Controls Mamba's internal state via `inference_params`
        output_attentions: Optional[
            bool
        ] = None,  # Mamba models do not produce attention weights
        output_hidden_states: Optional[
            bool
        ] = None,  # MixerModel can output hidden states
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,  # Mamba models do not use cache_position
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # 1. Prepare multimodal inputs using LlavaMetaForCausalLM's logic.
        # This will process images and inject embeddings into the sequence (`inputs_embeds`).
        # It also generates Transformer-style `attention_mask`, `position_ids`, `past_key_values`
        # which are largely irrelevant for Mamba's core `forward` pass.
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values_llava_prepped,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            modalities,
            image_sizes,
        )

        # Mamba's `inference_params` is its internal state. `past_key_values_llava_prepped` (from LlavaMetaForCausalLM)
        # will be `None` for the first token (full prompt) and will contain the Mamba state (from `generate`)
        # in subsequent steps for single-token decoding.
        mamba_inference_params = None
        if use_cache and past_key_values_llava_prepped is None:
            # Allocate cache for all layers of the model only on the first step of generation.
            # `current_sequence_length` is derived from `inputs_embeds.shape[1]`.
            mamba_inference_params = self.model.allocate_inference_cache(
                inputs_embeds.shape[0],
                inputs_embeds.shape[1],
                dtype=inputs_embeds.dtype,
            )
        elif past_key_values_llava_prepped is not None:
            # If Llava's `past_key_values` (which is Mamba's state) is provided, use it.
            mamba_inference_params = past_key_values_llava_prepped

        # Determine the input to the Mamba backbone.
        # If `inputs_embeds` is provided (which it will be if images are present or for DPO_forward),
        # we bypass the standard embedding lookup of `MambaLMHeadModel` and directly feed `inputs_embeds`
        # into the `MixerModel` layers.

        if inputs_embeds is not None:
            # Manual forward pass through MixerModel's layers, starting with `inputs_embeds`.
            # This handles multimodal inputs correctly.
            current_hidden_states = inputs_embeds
            residual = None  # MixerModel's layers start with residual=None

            # Iterate through the MixerModel's layers (self.model.layers).
            for layer in self.model.layers:
                # Each layer is a `Block`, which contains the Mamba/Mamba2 mixer.
                # Block's forward takes `hidden_states`, `residual`, `inference_params`.
                current_hidden_states, residual = layer(
                    current_hidden_states,
                    residual,
                    inference_params=mamba_inference_params,
                )

            # Apply final normalization, mirroring MixerModel's end processing.
            if not self.model.fused_add_norm:  # MixerModel uses this flag
                hidden_states_after_norm = (
                    (current_hidden_states + residual)
                    if residual is not None
                    else current_hidden_states
                )
                hidden_states_after_norm = self.model.norm_f(
                    hidden_states_after_norm.to(dtype=self.model.norm_f.weight.dtype)
                )  # MixerModel's final norm
            else:
                if rms_norm_fn is None:
                    raise ImportError(
                        "RMSNorm / layer_norm_fn required for fused_add_norm path and not found."
                    )
                hidden_states_after_norm = rms_norm_fn(  # Use Triton RMSNorm if fused
                    current_hidden_states,
                    self.model.norm_f.weight,
                    self.model.norm_f.bias,
                    eps=self.model.norm_f.eps,
                    residual=residual,
                    prenorm=False,  # As per MixerModel's forward
                    residual_in_fp32=self.model.residual_in_fp32,  # From MixerModel config
                    is_rms_norm=isinstance(self.model.norm_f, RMSNorm),
                )

            logits = self.lm_head(
                hidden_states_after_norm
            )  # Apply the language model head

            if dpo_forward:
                return logits, labels  # DPO expects (logits, labels) tuple

            # For standard CausalLMOutputWithPast
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

            # Mamba's `past_key_values` for CausalLMOutputWithPast should be its `inference_params`.
            # `mamba_inference_params` will hold the final state if `use_cache` is True.
            next_cache = mamba_inference_params if use_cache else None

            if not return_dict:
                # Output structure: (loss, logits, past_key_values, hidden_states, attentions)
                # Mamba models do not have attention weights.
                output = (
                    logits,
                    next_cache,
                    hidden_states_after_norm if output_hidden_states else None,
                    None,
                )
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=next_cache,
                hidden_states=hidden_states_after_norm
                if output_hidden_states
                else None,
                attentions=None,  # Mamba has no attention mechanism
            )

        else:
            # This branch is for when `inputs_embeds` is None (i.e., pure text input, no images).
            # In this case, `input_ids` should be available, and we can directly call
            # the base `MambaLMHeadModel`'s forward method.
            # It handles its own embedding lookup and basic state management.

            # MambaLMHeadModel's forward expects `input_ids` and `inference_params`.
            # `past_key_values_llava_prepped` here, if not None, should be Mamba's `inference_params`.

            mamba_output = super().forward(
                input_ids=input_ids,
                inference_params=past_key_values_llava_prepped,  # This should be Mamba's `inference_params` for stateful decoding
                # position_ids is ignored by MambaLMHeadModel.forward
            )

            logits = (
                mamba_output.logits
            )  # MambaLMHeadModel.forward returns a namedtuple with 'logits'

            # The base MambaLMHeadModel.forward doesn't return hidden_states or an updated inference_params
            # explicitly in its output tuple. For CausalLMOutputWithPast, these will be None.
            # `generate` method handles the full state management.

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            if not return_dict:
                # MambaLMHeadModel returns a namedtuple `CausalLMOutput(logits=lm_logits)`.
                # So, we return (logits, None, None, None) for (logits, pkvs, hidden_states, attentions).
                output = (logits, None, None, None)
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,  # MambaLMHeadModel.forward doesn't return this directly
                hidden_states=None,  # MambaLMHeadModel.forward doesn't return this directly
                attentions=None,  # Mamba has no attention mechanism
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[
            torch.Tensor
        ] = None,  # Corresponds to input_ids from tokenizer
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # `position_ids` and `attention_mask` are typically not passed directly to Mamba's core `forward`
        # during generation. They might be in kwargs from higher-level generation utilities.
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)

        if "inputs_embeds" in kwargs:
            # We handle `inputs_embeds` in `prepare_inputs_for_generation`.
            # If it's explicitly passed here, it might be an issue.
            raise NotImplementedError(
                "`inputs_embeds` should be handled by `prepare_inputs_for_generation` for multimodal inputs."
            )

        # `super().generate` (from `MambaLMHeadModel` which inherits `GenerationMixin`)
        # will call our `prepare_inputs_for_generation` and then our `forward`.
        return super().generate(
            inputs=inputs,  # This will be `input_ids` to `prepare_inputs_for_generation`
            images=images,  # Passed through to `prepare_inputs_for_generation`
            image_sizes=image_sizes,  # Passed through to `prepare_inputs_for_generation`
            modalities=modalities,  # Passed through to `prepare_inputs_for_generation`
            position_ids=position_ids,  # Pass through, will be filtered in `prepare_inputs_for_generation`
            attention_mask=attention_mask,  # Pass through, will be filtered in `prepare_inputs_for_generation`
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        modalities = kwargs.pop("modalities", ["image"])  # Get modalities from kwargs

        # This method is called by `generate`. It needs to return a dict that `self.forward` can use.
        # The `prepare_inputs_labels_for_multimodal` function from `LlavaMetaForCausalLM` is central here.

        # `past_key_values` from `generate` will be Mamba's `inference_params`
        # for subsequent tokens, or `None` for the first token.

        if images is not None:
            # This path is for multimodal inputs (first token or full prompt).
            # `prepare_inputs_labels_for_multimodal` processes images and injects embeddings.
            (
                input_ids_multimodal,
                position_ids_multimodal,
                attention_mask_multimodal,
                past_key_values_multimodal,
                inputs_embeds_multimodal,
                labels_multimodal,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,  # Original input_ids from tokenizer, potentially with placeholders
                kwargs.get("position_ids", None),
                kwargs.get("attention_mask", None),
                past_key_values,  # This will be Mamba's inference_params or None
                None,  # labels are None for generation
                images,
                modalities,
                image_sizes=image_sizes,
            )

            # For Mamba, we will use `inputs_embeds_multimodal` directly in the `forward` pass,
            # bypassing the standard embedding lookup. So, `input_ids` for `forward` should be `None`.
            model_inputs = {
                "inputs_embeds": inputs_embeds_multimodal,
                "input_ids": None,  # Force `forward` to use `inputs_embeds`
                "past_key_values": past_key_values_multimodal,  # This will be Mamba's `inference_params` for stateful decoding
                "use_cache": kwargs.get(
                    "use_cache", False
                ),  # Ensure use_cache is passed correctly
                # Pass original Transformer-style args, though Mamba's core `forward` ignores them.
                "position_ids": position_ids_multimodal,
                "attention_mask": attention_mask_multimodal,
            }

        else:
            # Standard text-only generation. Call the base MambaLMHeadModel's method.
            # This will prepare `input_ids` and Mamba's `inference_params` (mapped from `past_key_values`).
            model_inputs = super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        # Add any other kwargs that should be passed to `forward`
        model_inputs.update(kwargs)
        return model_inputs


AutoConfig.register("llava_mamba", LlavaMambaConfig)
AutoModelForCausalLM.register(LlavaMambaConfig, LlavaMambaForCausalLM)
