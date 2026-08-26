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

"""Tests for SpyreFp8LinearMethod (custom_ops/linear.py).

`process_weights_after_loading` is a host-side weight mutation tested on CPU.
`apply` delegates to torch-spyre FP8 ops (quantize_fp8_with_scale,
quantize_weight_fp8_with_scale, aten._scaled_mm) which are Spyre-device-only;
those tests are guarded by `spyre_available()`.
"""

import pytest
import torch
import torch.nn.functional as F

from torch_spyre._inductor.constants import FP8_E4M3FN_MAX
from spyre_testing_plugin.pytest_plugin import spyre_available


def _make_fp8_layer(in_features: int, out_features: int, *,
                    activation_scheme: str = "dynamic",
                    bias: bool = False):
    """Build a ColumnParallelLinear with Fp8Config and return the layer."""
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme=activation_scheme,
    )
    layer = ColumnParallelLinear(
        input_size=in_features,
        output_size=out_features,
        bias=bias,
        params_dtype=torch.float16,
        quant_config=quant_config,
        disable_tp=True,
        prefix="test",
    )
    return layer


def _cpu_fp8_linear(x: torch.Tensor, weight_fp8: torch.Tensor,
                    weight_scale: torch.Tensor,
                    scale_a: float = 1.0,
                    bias: torch.Tensor | None = None) -> torch.Tensor:
    """CPU reference for SpyreFp8LinearMethod.apply.

    Mirrors what _scaled_mm computes:
      q_x  = clamp(x / scale_a, -448, 448).to(fp8)  → dequant: q_x_f32 * scale_a
      q_w  = weight_fp8 codes                        → dequant: w_fp8_f32 * scale_b
      out  = (q_x_dequant @ q_w_dequant.T)

    weight_fp8 is [N, K] = [out_features, in_features] (vLLM/checkpoint convention).
    F.linear expects weight in [out_features, in_features] — no transpose needed.
    """
    # Simulate activation quantization (clamp to FP8 range, round-trip through fp8)
    q_x = (x.float() / scale_a).clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX).to(torch.float8_e4m3fn)
    x_dequant = q_x.to(torch.float32) * scale_a
    w_f32 = weight_fp8.to(torch.float32) * weight_scale.item()
    out = F.linear(x_dequant, w_f32)
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out.to(torch.float16)


@pytest.mark.fp8
def test_spyre_fp8_linear_method_installed(tp_group):
    """_SpyreTransposedLinearMixin installs SpyreFp8LinearMethod for Fp8Config."""
    from spyre_inference.custom_ops.linear import SpyreFp8LinearMethod

    layer = _make_fp8_layer(128, 256)
    assert isinstance(layer.quant_method, SpyreFp8LinearMethod), (
        f"Expected SpyreFp8LinearMethod, got {type(layer.quant_method)}"
    )


@pytest.mark.fp8
def test_spyre_fp8_linear_method_not_installed_for_unquantized(tp_group):
    """Unquantized layers still get SpyreUnquantizedLinearMethod, not Fp8."""
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from spyre_inference.custom_ops.linear import (
        SpyreFp8LinearMethod,
        SpyreUnquantizedLinearMethod,
    )

    layer = ColumnParallelLinear(
        input_size=128, output_size=256,
        bias=False, params_dtype=torch.float16,
        quant_config=None, disable_tp=True, prefix="test",
    )
    assert isinstance(layer.quant_method, SpyreUnquantizedLinearMethod)
    assert not isinstance(layer.quant_method, SpyreFp8LinearMethod)


@pytest.mark.fp8
@pytest.mark.parametrize("num_tokens", [1, 4, 16])
@pytest.mark.parametrize("in_features,out_features", [(128, 256), (256, 128)])
def test_fp8_weight_shape_after_loading(tp_group, num_tokens, in_features, out_features):
    """process_weights_after_loading DMAs weight to Spyre in QFP8WT KERNEL layout.

    Weight is transposed to [K, N] = [in_features, out_features] before DMA.
    _dma_to_spyre_fp8_kernel places it in [2,64] QFP8WT layout with dim_order=[0,1].
    """
    layer = _make_fp8_layer(in_features, out_features)

    # Initialise with random FP8 weights (simulate checkpoint load)
    layer.weight.data = (
        torch.randn(out_features, in_features, dtype=torch.float16) * 0.1
    ).to(torch.float8_e4m3fn)
    if hasattr(layer, "weight_scale"):
        layer.weight_scale.data = torch.tensor([1.0])

    layer.quant_method.process_weights_after_loading(layer)

    # After loading: weight is [K, N] = [in_features, out_features] (transposed for _scaled_mm).
    assert layer.weight.shape == (in_features, out_features), (
        f"Expected [{in_features}, {out_features}], got {layer.weight.shape}"
    )
    assert layer.weight.dtype == torch.float8_e4m3fn, (
        f"Expected float8_e4m3fn, got {layer.weight.dtype}"
    )


@pytest.mark.fp8
@pytest.mark.parametrize("num_tokens,in_features,out_features,scale_b",
                         [(1, 128, 128, 1.0),
                          (4, 128, 256, 2.0),
                          (2, 256, 128, 0.5)])
def test_fp8_apply_matches_reference(tp_group, num_tokens, in_features,
                                     out_features, scale_b):
    """SpyreFp8LinearMethod.apply output matches dequantized CPU reference.

    Requires Spyre hardware: apply() calls torch.ops.spyre.quantize_fp8_with_scale
    (activation → QFP8CH) and aten._scaled_mm with the pre-placed QFP8WT weight.
    quantize_weight_fp8_with_scale is NOT called — the weight is already in
    QFP8WT KERNEL layout from process_weights_after_loading via _dma_to_spyre_fp8_kernel.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.linear import SpyreFp8LinearMethod
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    torch.manual_seed(0)
    layer = _make_fp8_layer(in_features, out_features)

    # Simulate post-loading state: FP8 weight [out, in] + scalar scale
    w_fp8 = (
        torch.randn(out_features, in_features, dtype=torch.float32)
        .clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
        .to(torch.float8_e4m3fn)
    )
    layer.weight.data = w_fp8
    layer.weight_scale.data = torch.tensor([scale_b])

    # Compute CPU reference BEFORE process_weights_after_loading, while weight
    # is still a CPU tensor.  weight is [N,K]=[out,in]; _cpu_fp8_linear expects that.
    x_cpu = torch.randn(num_tokens, in_features, dtype=torch.float16)
    ref = _cpu_fp8_linear(x_cpu, w_fp8, torch.tensor([scale_b]))

    layer.quant_method.process_weights_after_loading(layer)
    # weight is now a Spyre QFP8WT KERNEL tensor ([K,N]=[in_features, out_features])

    layer.input_scale = None  # dynamic activation: scale computed per-call
    layer = layer.to("spyre:0")

    # The pre-DMA'd QFP8WT weight must be a frozen constant in the compiled graph,
    # not a graph input.  Compile a wrapper that captures layer from the closure so
    # the weight is a constant, not a symbolic input.
    def _forward(x):
        return layer.quant_method.apply(layer, x, bias=None)

    compiled_forward = torch.compile(_forward, backend="inductor", fullgraph=False)
    layer = layer.to("spyre:0")

    x = x_cpu.to("spyre:0")
    out = compiled_forward(x).cpu()

    torch.testing.assert_close(
        out.float(), ref.float(), atol=2.0, rtol=0.1,
        msg=f"FP8 apply mismatch for shape [{num_tokens},{in_features}]x[{in_features},{out_features}]"
    )


@pytest.mark.fp8
def test_fp8_apply_with_bias(tp_group):
    """Bias is correctly added after the FP8 matmul.

    Requires Spyre hardware: apply() calls torch-spyre FP8 ops.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.linear import SpyreFp8LinearMethod

    torch.manual_seed(1)
    in_f, out_f = 128, 128
    layer = _make_fp8_layer(in_f, out_f, bias=True)

    layer.weight.data = (
        torch.randn(out_f, in_f, dtype=torch.float32)
        .clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
        .to(torch.float8_e4m3fn)
    )
    layer.weight_scale.data = torch.tensor([1.0])
    layer.quant_method.process_weights_after_loading(layer)
    layer.input_scale = None

    x_cpu = torch.randn(2, in_f, dtype=torch.float16)
    bias_cpu = torch.randn(out_f, dtype=torch.float16)

    layer = layer.to("spyre:0")

    def _fwd_no_bias(x):
        return layer.quant_method.apply(layer, x, bias=None)

    def _fwd_with_bias(x):
        return layer.quant_method.apply(layer, x, bias=bias_cpu.to(x.device))

    compiled_no_bias   = torch.compile(_fwd_no_bias,   backend="inductor", fullgraph=False)
    compiled_with_bias = torch.compile(_fwd_with_bias, backend="inductor", fullgraph=False)

    x = x_cpu.to("spyre:0")

    out_no_bias = compiled_no_bias(x).cpu()
    out_bias    = compiled_with_bias(x).cpu()

    torch.testing.assert_close(
        out_bias.float(), (out_no_bias + bias_cpu).float(),
        atol=1e-2, rtol=0.0
    )


@pytest.mark.fp8
def test_fp8_method_type_stability_through_init(tp_group):
    """SpyreFp8LinearMethod survives multiple create_weights + process cycles."""
    from spyre_inference.custom_ops.linear import SpyreFp8LinearMethod

    layer = _make_fp8_layer(64, 64)
    assert isinstance(layer.quant_method, SpyreFp8LinearMethod)

    layer.weight.data = (
        torch.randn(64, 64, dtype=torch.float32)
        .clamp(-FP8_E4M3FN_MAX, FP8_E4M3FN_MAX)
        .to(torch.float8_e4m3fn)
    )
    layer.weight_scale.data = torch.tensor([1.0])
    layer.quant_method.process_weights_after_loading(layer)

    # Still SpyreFp8LinearMethod after processing
    assert isinstance(layer.quant_method, SpyreFp8LinearMethod)
