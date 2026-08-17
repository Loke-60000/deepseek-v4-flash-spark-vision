# MoonViT variant (untested)

Pairs [`0xSero/deepseek-v4-flash-0731-spark`](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) with the MoonViT-3d tower and
PatchMerger projector published by
[`webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4`](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4).

WebBrain's own integration targets SGLang. vLLM already carries the vision half:
`MoonViT3dPretrainedModel` and `KimiK25MultiModalProjector` live in
`vllm/model_executor/models/kimi_k25.py` and `kimi_k25_vit.py`, together with
the KimiK25 processing pipeline. `model.py` reuses those and replaces the
language model with DeepSeek V4, then substitutes the 64-entry routing palette
into `input_ids` at image positions.

This code has not been run. The DeepEncoderV2 path in the parent repository is
the one with measured output. Two things need checking first: the tower weights
in `vision_tower.safetensors` carry a `vision_tower.` prefix and the loader here
does not yet remap them, and `image_placeholder_token_id` is 129280 while
`vocab_size` is 129280, so the sentinel sits one past the end of the vocabulary
and must never reach the embedding lookup unsubstituted.

Build a model directory the same way as the DeepEncoder variant: symlink the
backbone shards, then merge `vision_config` and `deepseek_vision` from
WebBrain's `config.json` into the backbone `config.json` and set
`architectures` to `["DeepseekV4MoonViTForCausalLM"]`.
