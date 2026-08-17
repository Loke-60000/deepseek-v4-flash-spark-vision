# Single-pass build of the image published by .github/workflows/publish-image.yml.
#
# Chain, in order:
#   1. pinned SparkInfer runtime (arm64, GB10)
#   2. xgrammar 0.2.4, without which any request carrying `tools` returns HTTP 500
#   3. the FlyCockpit DeepEncoderV2 plugin, fetched at build time
#   4. the EAGLE3 delegation the plugin needs before DSpark will load it
#
# The plugin is cloned rather than vendored, so its source stays under its own
# repository and license.

ARG BASE=ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4
FROM ${BASE}

ARG PLUGIN_REPO=https://github.com/FlyCockpit/DeepSeek-V4-Vision-2x-DGX-Sparks.git
ARG PLUGIN_REF=main

# The base ships xgrammar 0.1.27 in dist-packages and /opt/runtime-venv was made
# with --system-site-packages, so it inherits that copy. vLLM imports
# normalize_tool_choice, which only exists in 0.2.x. --no-deps keeps pip away
# from the pinned torch 2.12.0+cu130 ABI.
RUN /opt/runtime-venv/bin/pip install --no-deps --no-cache-dir xgrammar==0.2.4 \
 && /opt/runtime-venv/bin/python -c "from xgrammar import StructuralTag, normalize_tool_choice" \
 && /opt/runtime-venv/bin/python -c "import torch; assert torch.__version__.startswith('2.12.0'), torch.__version__"

RUN git clone --depth 1 --branch "${PLUGIN_REF}" "${PLUGIN_REPO}" /tmp/fc \
 && cp -r /tmp/fc/plugin /opt/vision-plugin \
 && rm -rf /tmp/fc \
 && /opt/runtime-venv/bin/pip install --no-deps -e /opt/vision-plugin \
 && /opt/runtime-venv/bin/python -c "import dsv4_vision_vllm"

COPY patch_eagle3.py /tmp/patch_eagle3.py
RUN /opt/runtime-venv/bin/python /tmp/patch_eagle3.py \
 && /opt/runtime-venv/bin/python -c "\
import dsv4_vision_vllm.model as m; \
c = m.DeepseekV4VisionForCausalLM; \
assert getattr(c, 'supports_eagle3', False), 'EAGLE3 flag missing'; \
assert hasattr(c, 'set_aux_hidden_state_layers'), 'EAGLE3 setter missing'; \
print('eagle3 ok')" \
 && rm -f /tmp/patch_eagle3.py
