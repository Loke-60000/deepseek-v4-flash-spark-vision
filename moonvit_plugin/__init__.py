"""Registers the MoonViT variant with vLLM.

Loaded through the vllm.general_plugins entry point, which vLLM calls in every
worker before the model is built.
"""


def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "DeepseekV4MoonViTForCausalLM",
        "dsv4_moonvit_vllm.model:DeepseekV4MoonViTForCausalLM",
    )
