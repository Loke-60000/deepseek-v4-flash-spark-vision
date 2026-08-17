"""DSV4-Flash-0731-spark with the MoonViT-3d tower and WebBrain's PatchMerger.

WebBrain publishes this pairing for SGLang. vLLM already carries every vision
piece it needs: MoonViT3dPretrainedModel and KimiK25MultiModalProjector in
vllm/model_executor/models/kimi_k25*.py, plus the whole KimiK25 multimodal
processing path. So this wrapper reuses KimiK25ForConditionalGeneration and
changes two things.

First, the language model. KimiK25ForConditionalGeneration hardcodes
architectures=["DeepseekV2ForCausalLM"] in __init__, so __init__ is rewritten
here to pass DeepseekV4ForCausalLM and to build the vision configs from the raw
hf_config rather than a registered KimiK25Config.

Second, routing. DSV4's MoE picks experts from tid2eid[input_ids], so image rows
carry no routing information. WebBrain assigns them IDs from a fixed 64-entry
palette, cycling by position within each image:

    palette[(absolute_position - image_start) % 64]

Layers 0, 1 and 2 are the only ones that read tid2eid; layers 3 through 42 gate
on hidden states. On the REAP-pruned 216-expert table that palette reaches 210
of 216 experts on layer 0, against 176 for 64 randomly drawn IDs.

requires_raw_input_tokens keeps input_ids alive next to inputs_embeds, which is
what makes the substitution possible without patching vLLM.

UNTESTED. The FlyCockpit DeepEncoderV2 path in the parent repository is the one
with measured output.
"""

from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import SupportsEagle3
from vllm.model_executor.models.kimi_k25 import (
    KimiK25DummyInputsBuilder,
    KimiK25ForConditionalGeneration,
    KimiK25MultiModalProcessor,
    KimiK25MultiModalProjector,
    KimiK25ProcessingInfo,
)
from vllm.model_executor.models.kimi_k25_vit import MoonViT3dPretrainedModel
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform


def _vision_config(hf_config):
    """KimiK25VisionConfig built from the vision_config block in config.json."""
    from vllm.transformers_utils.configs.kimi_k25 import KimiK25VisionConfig

    raw = getattr(hf_config, "vision_config", None)
    if raw is None:
        raise ValueError("config.json has no vision_config block")
    if not isinstance(raw, dict):
        raw = raw.to_dict()
    return KimiK25VisionConfig(**raw)


def _vision_meta(hf_config) -> dict:
    meta = getattr(hf_config, "deepseek_vision", None)
    if meta is None:
        raise ValueError("config.json has no deepseek_vision block")
    return meta if isinstance(meta, dict) else dict(meta)


@MULTIMODAL_REGISTRY.register_processor(
    KimiK25MultiModalProcessor,
    info=KimiK25ProcessingInfo,
    dummy_inputs=KimiK25DummyInputsBuilder,
)
class DeepseekV4MoonViTForCausalLM(KimiK25ForConditionalGeneration, SupportsEagle3):
    # Keep input_ids beside inputs_embeds so tid2eid routing still works.
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        model_config = vllm_config.model_config
        config = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        vis_cfg = _vision_config(config)
        meta = _vision_meta(config)

        self.route_palette = tuple(int(v) for v in meta["routing_palette"])
        if not self.route_palette:
            raise ValueError("routing_palette is empty")
        self.media_placeholder = int(meta["image_placeholder_token_id"])
        self.max_image_tokens = int(meta.get("max_image_tokens", 512))

        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.hidden_size
        self.device = current_platform.current_device()

        self.vision_tower = MoonViT3dPretrainedModel(
            vis_cfg,
            quant_config=None,
            prefix=maybe_prefix(prefix, "vision_tower"),
        ).to(device=self.device, dtype=model_config.dtype)

        self.mm_projector = KimiK25MultiModalProjector(
            config=vis_cfg,
            use_data_parallel=self.use_data_parallel,
            quant_config=None,
            prefix=maybe_prefix(prefix, "mm_projector"),
        ).to(device=self.device, dtype=model_config.dtype)

        if self.mm_projector.linear_2.out_features != self.hidden_size:
            raise ValueError(
                "projector emits %d, backbone hidden_size is %d"
                % (self.mm_projector.linear_2.out_features, self.hidden_size)
            )

        self.quant_config = quant_config
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["DeepseekV4ForCausalLM"],
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # --- EAGLE3, required by DSpark. mtp0 crashes on this build. ---
    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        inner = self.language_model
        fn = getattr(inner, "get_eagle3_default_aux_hidden_state_layers", None)
        return (fn or inner.get_eagle3_aux_hidden_state_layers)()

    def _routing_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replace image-slot IDs with the cycling palette, leave text alone.

        Position within the run, not position in the sequence, picks the palette
        entry, so a prompt split across chunked-prefill batches routes the same
        way it would unsplit.
        """
        out = input_ids.clone()
        mask = out == self.media_placeholder
        if not bool(mask.any()):
            return out
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        # restart the cycle at each contiguous run of image tokens
        starts = torch.ones_like(idx, dtype=torch.bool)
        starts[1:] = (idx[1:] - idx[:-1]) != 1
        run_id = torch.cumsum(starts.long(), 0) - 1
        run_start = torch.zeros_like(idx)
        first = idx[starts]
        run_start = first[run_id]
        phase = (idx - run_start) % len(self.route_palette)
        pal = torch.tensor(self.route_palette, device=out.device, dtype=out.dtype)
        out[idx] = pal[phase]
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        if input_ids is not None:
            input_ids = self._routing_ids(input_ids)
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
