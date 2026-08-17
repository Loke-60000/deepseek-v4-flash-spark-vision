"""Let the vision wrapper run under DSpark.

model_runner sets use_aux_hidden_state_outputs for method in
(eagle3, dflash, dspark), then calls set_eagle3_aux_hidden_state_layers(),
which hard-requires the EAGLE3 interface. The DSV4 backbone satisfies it via
deepseek_v2.DeepseekV2ForCausalLM; the wrapper subclasses only
nn.Module/SupportsMultiModal/SupportsPP, so the interface never reached it.
Delegate the three hooks to self.language_model.

mtp0 is not an alternative: MODE=off reproduces the swa_k_cache page-stride
crash on the plain text model with no vision involved, so dspark is the only
mode that actually works on this build.
"""
import ast
import sys

PATH = "/opt/vision-plugin/src/dsv4_vision_vllm/model.py"

ANCHOR = "    requires_raw_input_tokens = True"
METHODS = '''
    # --- EAGLE3 interface, required by DSpark (see module docstring) ---
    supports_eagle3 = True

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        inner = self.language_model
        fn = getattr(inner, "get_eagle3_default_aux_hidden_state_layers", None)
        if fn is None:
            fn = inner.get_eagle3_aux_hidden_state_layers
        return fn()
'''

src = open(PATH).read()
if "supports_eagle3" in src:
    print("already patched")
    sys.exit(0)
if ANCHOR not in src:
    sys.exit("ANCHOR NOT FOUND in %s" % PATH)

src = src.replace(ANCHOR, ANCHOR + "\n" + METHODS, 1)

# make the class actually advertise the protocol
OLD_CLS = ("class DeepseekV4VisionForCausalLM(nn.Module, SupportsMultiModal, "
           "SupportsPP):")
NEW_CLS = ("class DeepseekV4VisionForCausalLM(nn.Module, SupportsMultiModal, "
           "SupportsPP, SupportsEagle3):")
if OLD_CLS in src:
    src = src.replace(OLD_CLS, NEW_CLS, 1)
    src = src.replace(
        "from vllm.model_executor.models.interfaces import (",
        "from vllm.model_executor.models.interfaces import (\n    SupportsEagle3,",
        1,
    )
else:
    print("WARN: class signature not matched; relying on duck-typed ClassVar")

open(PATH, "w").write(src)
ast.parse(src)
print("patched %s" % PATH)
