# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spyre OOT linear layers and the shared transposed-weight fast path.

`SpyreTransposedWeightMethod` is the common base for the Spyre linear and
lm-head quant methods: it stores each 2-D weight physically transposed as `Wᵀ`
(shape `[in, out]`, contiguous) in `process_weights_after_loading` and runs the
Spyre-fast `x @ Wᵀ` in `apply`. Subclasses parameterize the destination
attribute and an optional output-row padding (needed by the lm-head).
"""

from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearMethodBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod

logger = init_logger(__name__)


def spyre_linear_t(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor | None):
    """Linear forward with a pre-transposed weight: `x @ Wᵀ (+ bias)`.

    `weight_t` is the physically-transposed weight of shape `[in, out]`, so the
    matmul is a plain `x @ A` (the Spyre-fast layout), not `F.linear`'s `x @ Aᵀ`.
    """
    out = torch.matmul(x, weight_t)
    if bias is not None:
        out = out + bias
    return out


class SpyreTransposedWeightMethod:
    """Shared Spyre weight handler: store `Wᵀ` (optionally row-padded) and matmul it.

    A mixin combined *before* a concrete vLLM `Unquantized*Method` (which supplies
    `create_weights` and the `QuantizeMethodBase` lineage), so `super()` calls in
    the methods below reach that concrete method via the MRO. Subclasses set:

    - `WEIGHT_T_ATTR`: layer attribute that holds the transposed weight.
      `"weight"` replaces it in place; a distinct name (e.g. `"padded_weight_t"`)
      preserves the original `weight` — required for a tied lm-head whose
      `weight` IS `embed_tokens.weight` and must keep its gather layout.
    - `ROW_ALIGN`: pad the output (row) dim up to a multiple of this before
      transposing (torch-spyre matmul work-division limit); `None` = no padding.
    """

    WEIGHT_T_ATTR: str = "weight"
    ROW_ALIGN: int | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        w = cast(torch.Tensor, layer.weight).data
        padding = (-w.shape[0]) % self.ROW_ALIGN if self.ROW_ALIGN else 0
        layer.spyre_row_padding = padding
        if padding:
            padded = F.pad(w, (0, 0, 0, padding))
            logger.warning_once(
                "%s: weights padded from %d to %d (torch-spyre limitation) "
                "expect numerical differences to upstream vLLM.",
                layer.__class__.__name__,
                w.shape[0],
                padded.shape[0],
            )
            w = padded

        # Store transposed (`[in, out]`, contiguous) so the forward GEMM is the
        # Spyre-fast `x @ A`. `.t().contiguous()` gives INDEPENDENT storage, so a
        # distinct WEIGHT_T_ATTR leaves the source `weight` untouched.
        setattr(layer, self.WEIGHT_T_ATTR, Parameter(w.t().contiguous(), requires_grad=False))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = spyre_linear_t(x, getattr(layer, self.WEIGHT_T_ATTR), bias)
        padding = cast(int, layer.spyre_row_padding)
        if padding:
            # Drop the trailing pad columns; the slice lowers on-device eagerly
            # (torch-spyre #3578 honors the storage offset).
            out = out[:, :-padding]
        return out


class SpyreUnquantizedLinearMethod(SpyreTransposedWeightMethod, UnquantizedLinearMethod):
    """Unquantized linear method: store `Wᵀ` in place and matmul it (torch-spyre #3512).

    Uses the shared base defaults (`WEIGHT_T_ATTR="weight"`, no padding), so the
    forward GEMM is the Spyre-fast `x @ Wᵀ` instead of `F.linear`'s `x @ Aᵀ`.
    """


class SpyreFp8LinearMethod(LinearMethodBase):
    """FP8 linear method for Spyre: per-tensor W8A8 using torch-spyre FP8 ops.

    Delegates weight creation and loading to vLLM's `Fp8LinearMethod` (which
    handles checkpoint parameter registration and scale merging), then:

    * `process_weights_after_loading` — DMAs the pre-quantized `float8_e4m3fn`
      weight directly to Spyre using `_dma_to_spyre_fp8_kernel`, which places it
      into the `[2, 64]` QFP8WT KERNEL layout expected by `_scaled_mm`.
      No runtime re-quantization is needed or performed.

    * `apply` — quantizes the activation via
      `quantize_fp8_with_scale(x, scale_a)` → QFP8CH, then calls
      `aten._scaled_mm(q_x, w_kernel, scale_a, scale_b)`.  The weight is
      already in QFP8WT KERNEL layout from DMA; `quantize_weight_fp8_with_scale`
      is intentionally NOT called.

    Only non-block, non-marlin per-tensor quantization is supported — the
    standard path for FP8 checkpoints like `granite-3.3-8b-instruct-FP8`.
    Block-wise and Marlin paths are tracked separately in issue #259.
    """

    def __init__(self, quant_config):
        self._delegate = Fp8LinearMethod(quant_config)

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        # Fp8LinearMethod.create_weights registers weight / weight_scale /
        # input_scale parameters and then calls init_fp8_linear_kernel to
        # select a CUDA/XPU kernel — that last step raises ValueError on the
        # Spyre OOT platform where none of those kernels are available.
        # All parameter registration happens before the kernel-selection call,
        # so it is safe to swallow that specific ValueError.
        try:
            self._delegate.create_weights(
                layer, input_size_per_partition, output_partition_sizes,
                input_size, output_size, params_dtype, **extra_weight_attrs,
            )
        except ValueError as exc:
            if "Failed to find a kernel" not in str(exc):
                raise

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """DMA the pre-quantized FP8 weight to Spyre in QFP8WT KERNEL layout.

        Steps:
        1. For fused modules (MergedColumn: gate_up_proj) the checkpoint
           provides one scale per logical shard.  We take the max scale and
           re-express each shard's weight in that unified scale (CPU fp32,
           no CUDA helpers) so a single `_scaled_mm` covers the whole fused
           weight with one scalar scale.

        2. Normalise weight_scale to a scalar fp32 and set input_scale.

        3. Transpose to [K, N] = [in_features, out_features] — the shape
           `_scaled_mm` expects for the weight operand.

        4. Call `_dma_to_spyre_fp8_kernel(weight_KN)` — this DMAs the
           `float8_e4m3fn` tensor ([K, N]) directly to Spyre with the `[2, 64]`
           QFP8WT KERNEL layout and `dim_order=[0, 1]`.  This is the exact
           layout `_qfp8wt_stl` in torch-spyre's propagate_layouts produces,
           so no ReStickify is needed at compile time.
        """
        from vllm.model_executor.utils import replace_parameter
        from torch_spyre.model_utils import _dma_to_spyre_fp8_kernel

        weight: torch.Tensor = layer.weight        # [N, K], float8_e4m3fn
        weight_scale: torch.Tensor = layer.weight_scale  # scalar or [num_shards]
        input_scale = getattr(layer, "input_scale", None)

        # --- merge per-shard scales for fused modules (CPU, fp32, no CUDA) ---
        if weight_scale.numel() > 1:
            # weight_scale is [num_shards]; each shard covers logical_widths[i] rows.
            # Re-quantize each shard to the global max scale so _scaled_mm sees
            # a single uniform scale.
            max_scale = weight_scale.max().to(torch.float32)
            weight_f32 = weight.to(torch.float32)

            from torch_spyre._inductor.constants import FP8_E4M3FN_MAX
            new_w = torch.empty_like(weight_f32)
            offset = 0
            for shard_rows, shard_scale in zip(layer.logical_widths, weight_scale):
                shard = weight_f32[offset:offset + shard_rows]
                new_w[offset:offset + shard_rows] = (
                    shard * shard_scale.to(torch.float32) / max_scale
                ).clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
                offset += shard_rows

            weight = new_w.to(torch.float8_e4m3fn)
            weight_scale = max_scale
        else:
            weight_scale = weight_scale.reshape([]).to(torch.float32)

        if self._delegate.act_q_static and input_scale is not None:
            input_scale = input_scale.max()

        # Transpose to [K, N] = [in_features, out_features] for _scaled_mm,
        # then DMA to Spyre with QFP8WT KERNEL layout ([2,64] sticks, dim_order=[0,1]).
        # _dma_to_spyre_fp8_kernel expects the weight in the shape _scaled_mm sees
        # it: [K, N] (matches _qfp8wt_stl dim_order=[0,1] with identity layout).
        weight = _dma_to_spyre_fp8_kernel(weight.t().contiguous())

        # Store scales as fp16 on device now so apply() needs no dtype conversion
        # inside the compiled region (a .to(fp16) inside the graph creates a
        # STANDARD-layout buffer that triggers a mixed-EA error when fused with
        # the QFP8WT weight).
        weight_scale_fp16 = weight_scale.to(torch.float16)

        replace_parameter(layer, "weight", weight.data)
        replace_parameter(layer, "weight_scale", weight_scale_fp16)

        if input_scale is not None:
            replace_parameter(layer, "input_scale",
                              input_scale.to(torch.float16))
        else:
            layer.input_scale = None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight              # [K, N] QFP8WT KERNEL on Spyre
        weight_scale = layer.weight_scale  # scalar, float16

        # Scales are already fp16 (stored as such in process_weights_after_loading).
        scale_a = layer.input_scale if layer.input_scale is not None \
            else torch.ones(1, dtype=torch.float16, device=x.device)
        scale_b = weight_scale  # already fp16

        # Quantize activation to QFP8CH layout.
        q_x = torch.ops.spyre.quantize_fp8_with_scale(x, scale_a)

        # weight is already in QFP8WT KERNEL layout from process_weights_after_loading.
        # Do NOT call quantize_weight_fp8_with_scale — the weight is pre-placed.
        out = torch.ops.aten._scaled_mm(
            q_x, weight,
            scale_a=scale_a,
            scale_b=scale_b,
            bias=None,
            out_dtype=torch.float16,
        )
        if bias is not None:
            out = out + bias
        return out


class _SpyreTransposedLinearMixin:
    """Swaps in Spyre-specific linear methods.

    - `UnquantizedLinearMethod` → `SpyreUnquantizedLinearMethod` (transposed weight)
    - `Fp8LinearMethod`         → `SpyreFp8LinearMethod` (FP8 matmul via torch-spyre)

    The swap must happen *before* `LinearBase.__init__` calls
    `self.quant_method.create_weights`, because `Fp8LinearMethod.create_weights`
    ends by calling `init_fp8_linear_kernel` which raises `ValueError` on the
    Spyre OOT platform (no CUDA/XPU kernels available).  We intercept by
    monkey-patching `quant_config.get_quant_method` on the kwargs instance so
    that `LinearBase.__init__` installs `SpyreFp8LinearMethod` directly instead
    of `Fp8LinearMethod`.
    """

    def __init__(self, *args, **kwargs):
        quant_config = kwargs.get("quant_config") or (args[6] if len(args) > 6 else None)

        # Patch quant_config.get_quant_method so LinearBase.__init__ installs
        # SpyreFp8LinearMethod instead of Fp8LinearMethod before create_weights.
        _orig_get_quant_method = None
        if quant_config is not None and hasattr(quant_config, "get_quant_method"):
            _orig = quant_config.get_quant_method

            def _patched_get_quant_method(layer, prefix=""):
                method = _orig(layer, prefix=prefix)
                if isinstance(method, Fp8LinearMethod):
                    return SpyreFp8LinearMethod(method.quant_config)
                return method

            quant_config.get_quant_method = _patched_get_quant_method
            _orig_get_quant_method = _orig

        try:
            super().__init__(*args, **kwargs)
        finally:
            # Restore original method to avoid side-effects on shared config.
            if _orig_get_quant_method is not None:
                quant_config.get_quant_method = _orig_get_quant_method

        if isinstance(self.quant_method, UnquantizedLinearMethod):
            self.quant_method = SpyreUnquantizedLinearMethod()


@ColumnParallelLinear.register_oot(name="ColumnParallelLinear")
class SpyreColumnParallelLinear(_SpyreTransposedLinearMixin, ColumnParallelLinear):
    """OOT ColumnParallelLinear storing `Wᵀ` for the Spyre-fast GEMM."""


@MergedColumnParallelLinear.register_oot(name="MergedColumnParallelLinear")
class SpyreMergedColumnParallelLinear(_SpyreTransposedLinearMixin, MergedColumnParallelLinear):
    """OOT MergedColumnParallelLinear (e.g. gate_up_proj) storing `Wᵀ`."""


@RowParallelLinear.register_oot(name="RowParallelLinear")
class SpyreRowParallelLinear(_SpyreTransposedLinearMixin, RowParallelLinear):
    """OOT RowParallelLinear (e.g. o_proj, down_proj) storing `Wᵀ`."""


@ReplicatedLinear.register_oot(name="ReplicatedLinear")
class SpyreReplicatedLinear(_SpyreTransposedLinearMixin, ReplicatedLinear):
    """OOT ReplicatedLinear storing `Wᵀ` for the Spyre-fast GEMM."""


@QKVParallelLinear.register_oot(name="QKVParallelLinear")
class SpyreQKVParallelLinear(_SpyreTransposedLinearMixin, QKVParallelLinear):
    """OOT QKVParallelLinear for IBM's Spyre device.

    The fused QKV output is returned whole; the model splits it on-device with
    the unmodified `qkv.split(...)` idiom (no CPU-side unfusing).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.gather_output, (
            f"{self.__class__.__name__} requires gather_output=False; "
            "all_gather is not yet supported on Spyre"
        )
