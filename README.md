# deepseek-v4-flash-spark-vision

Container build, compose file, and patches that run a vision tower on top of
[`0xSero/deepseek-v4-flash-0731-spark`](https://huggingface.co/0xSero/deepseek-v4-flash-0731-spark) on one NVIDIA DGX Spark. One vLLM server
answers both text and image requests at 262144 context with DSpark speculative
decoding on.

No weights live here. The backbone comes from `0xSero`, the tower and projector
from [`FlyCockpit/DeepSeek-V4-Flash-0731-vision`](https://huggingface.co/FlyCockpit/DeepSeek-V4-Flash-0731-vision).

Measurements, model provenance, and the quality limits are on the Hugging Face
side: [Loke-60000/deepseek-v4-flash-0731-spark-vision-exp](https://huggingface.co/Loke-60000/deepseek-v4-flash-0731-spark-vision-exp)

## Pull the prebuilt image

```bash
docker pull ghcr.io/loke-60000/deepseek-v4-flash-spark-vision:latest
```

arm64 only, built for GB10. It carries the SparkInfer runtime, the xgrammar
fix, the FlyCockpit plugin, and the EAGLE3 patch, so nothing needs building.
Point `compose.vision.yaml` at it by setting `image:` to that tag, then supply
the model directory, tower, and adapter as below. `.github/workflows/publish-image.yml`
builds it; the base layer is about 21.7 GB, so the job wants an arm64 machine
with real disk rather than a hosted runner.

## Build it yourself

```bash
# model directory: symlinks to the backbone shards plus a patched config.json
./make_tp1_vision.sh data/tp1 data/tp1-vision

# tower and projector from the FlyCockpit repository
mkdir -p vision-assets/tower vision-assets/adapter
# deepencoder_v2_tower.safetensors  -> vision-assets/tower/
# merged-004800-5af0c5.pt           -> vision-assets/adapter/

docker build -f Dockerfile.vision -t deepseek-v4-flash-sparkinfer:vision .
docker build -f Dockerfile.eagle3 -t deepseek-v4-flash-sparkinfer:eagle3 .
docker compose -f compose.vision.yaml up -d
```

The server binds port 8000. `VLLM_PORT` is read by the compose file and ignored
by the launcher script inside the image.

## The two patches

### patch_eagle3.py

Submitted upstream as
[FlyCockpit PR #1](https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks/pull/1), applied
in place on the fork at
[the fork](https://github.com/Loke-60000/deepseek-v4-vision-2x-dgx-sparks). The script here
rewrites the installed plugin at image build time, which is what to use until
the change lands upstream.

`DeepseekV4VisionForCausalLM` in the FlyCockpit plugin subclasses `nn.Module`,
`SupportsMultiModal`, and `SupportsPP`, and does not implement the EAGLE3
interface. vLLM sets `use_aux_hidden_state_outputs` for any speculative method
in `("eagle3", "dflash", "dspark")` and then calls
`set_eagle3_aux_hidden_state_layers()`, which raises:

```
RuntimeError: Model does not support EAGLE3 interface
```

The text backbone passes that check by inheriting from `DeepseekV2ForCausalLM`.
The wrapper forwards `compute_logits` and `get_mtp_target_hidden_states` to
`self.language_model` but not the EAGLE3 hooks. The patch adds
`supports_eagle3 = True` and three delegating methods.

### MAX_MODEL_LEN

Reported upstream as
[SparkInfer issue #4](https://github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer/issues/4).

`MODE=off` (mtp0) fails on this build:

```
RuntimeError: swa_k_cache page stride 37376 is smaller than DSV4 page width 37440
```

The same crash reproduces on the plain text model with no vision plugin loaded,
at `MAX_MODEL_LEN=16384`. It does not appear at 262144.
`kv_cache_interface.py` returns `storage_block_size * 584 = 37376` unpadded
while `models/deepseek_v4/nvidia/b12x.py` requires
`round_up(37376, 576) = 37440`. `VLLM_DSV4_PADDED_NVFP4=1` does not change it,
because the spec that reaches `b12x` carries `cache_dtype_str = None` and falls
through both DeepSeek branches. Use `MODE=dspark` with a large context.

At `GPU_MEMORY_UTILIZATION=0.93` the engine reports 7.93 GiB of KV cache needed
against 7.53 GiB available, because the tower takes about 0.9 GiB. The compose
file uses 0.945, which gives a KV cache of 298,241 tokens on 121.69 GiB of
unified memory.

## probes/

`vision_test.py` sends a text prompt, a generated image of a red circle and a
blue square, and an image of rendered text, then prints the three answers.
`vision_img.py` takes a path and asks two questions about it. Both talk to
`http://127.0.0.1:8000` and need `pillow`.

## scripts/

`palette_check.py` maps a routing palette through the backbone's `tid2eid`
tables and reports how many of the 216 surviving experts it reaches, with a
random-token baseline.

`calib_local.py` measures the projector's output norm against the norm of
`embed.weight` and writes a rescaled checkpoint. The rescale makes output worse;
the reasoning and the numbers are in the Hugging Face card. It is kept because
the measurement is useful, not because the result should be deployed.

## moonvit_plugin/

An untested variant that pairs the backbone with the MoonViT-3d tower and
PatchMerger projector from
[`webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4`](https://huggingface.co/webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4). WebBrain's integration
targets SGLang; vLLM already carries `MoonViT3dPretrainedModel` and
`KimiK25MultiModalProjector`, so this reuses `KimiK25ForConditionalGeneration`
and swaps the language model for DeepSeek V4, then substitutes the 64-entry
routing palette into `input_ids` at image positions.

Two gaps before it can run: the tower weights carry a `vision_tower.` prefix
that the loader does not remap, and `image_placeholder_token_id` is 129280 while
`vocab_size` is 129280, so the sentinel sits one past the end of the vocabulary
and has to be substituted before any embedding lookup.

## Licenses

Each upstream artifact keeps its own license: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
for the backbone lineage, the FlyCockpit repository for the tower and projector,
and [`moonshotai/Kimi-K2.6`](https://huggingface.co/moonshotai/Kimi-K2.6) for the MoonViT weights the untested variant expects.
