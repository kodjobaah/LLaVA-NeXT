import os

AVAILABLE_MODELS = {
    "llava_llama": "LlavaLlamaForCausalLM, LlavaConfig",
    "llava_qwen": "LlavaQwenForCausalLM, LlavaQwenConfig",
    "llava_mistral": "LlavaMistralForCausalLM, LlavaMistralConfig",
    "llava_mixtral": "LlavaMixtralForCausalLM, LlavaMixtralConfig",
    # "llava_qwen_moe": "LlavaQwenMoeForCausalLM, LlavaQwenMoeConfig",    
    "llava_mamba" : "LlavaMambaModel, LlavaMambaConfig",
    
    # Add other models as needed
}

__all__ = []

for model_name, model_classes in AVAILABLE_MODELS.items():
    try:
        exec(f"from .language_model.{model_name} import {model_classes}")
        __all__.extend(model_classes.replace(" ", "").split(","))
    except Exception as e:
        print(f"Failed to import {model_name} from llava.language_model.{model_name}. Error: {e}")
