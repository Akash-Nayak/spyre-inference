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

"""End-to-end tests for FP8 model inference on Spyre.

Verifies the full prequant weight lifecycle with a real FP8 checkpoint:
  - Model loads and SpyreFp8LinearKernel is selected for all FP8 linears
  - layer._qfp8wt_for_mm is populated on all linear layers after warmup
  - Generated text is non-empty (basic sanity)
  - Prequant (default) and fallback (FORCE=0) paths produce the same token ids

The model is loaded from a local NFS path so no HF download is needed.
All tests use the vLLM v1 subprocess path (EngineCore runs in a child process).
Model introspection is done via ``llm.apply_model(fn)`` which sends a callable
into the EngineCore worker subprocess — this is the only safe way to inspect
the model in vLLM v1.

``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` is set so pickle can transport the
module-level callables to the worker.  The callable helpers (_count_fp8_layers,
_check_prequant_cache) are module-level so pickle can locate them by
qualified name (local/nested functions cannot be pickled).

The ``SPYRE_FP8_PREQUANT_FORCE=0`` env var disables pre-quantization and falls
back to running qfp8wt inside the compiled graph every forward (used only in
``test_fp8_prequant_matches_fallback`` for numerical comparison).
"""

from __future__ import annotations

import pytest

_FP8_MODEL = "/nfs_mnt/models/granite-3.3-8b-instruct-FP8"
_PROMPT = "What is the capital of France?"
_MAX_TOKENS = 16
_MAX_MODEL_LEN = 128
_MAX_NUM_SEQS = 2
_MAX_NUM_BATCHED_TOKENS = 16

pytestmark = pytest.mark.uses_subprocess


# ---------------------------------------------------------------------------
# Module-level worker callables (must be importable by name for pickle).
# ---------------------------------------------------------------------------


def _is_fp8_layer(mod, SpyreFp8LinearKernel) -> bool:
    """True if ``mod`` uses SpyreFp8LinearKernel.

    For compressed-tensors FP8 the kernel lives at ``mod.scheme.fp8_linear``
    (CompressedTensorsW8A8Fp8.fp8_linear), not ``mod.quant_method.fp8_linear``.
    Both paths are checked for forward-compatibility.
    """
    # compressed-tensors path: layer.scheme.fp8_linear
    scheme = getattr(mod, "scheme", None)
    if isinstance(getattr(scheme, "fp8_linear", None), SpyreFp8LinearKernel):
        return True
    # direct fp8 path: layer.quant_method.fp8_linear (Fp8LinearMethod)
    qm = getattr(mod, "quant_method", None)
    return isinstance(getattr(qm, "fp8_linear", None), SpyreFp8LinearKernel)


def _count_fp8_layers(model: torch.nn.Module) -> int:  # noqa: F821
    """Count SpyreFp8LinearKernel layers. Runs inside the EngineCore worker."""
    from spyre_inference.custom_ops.fp8_linear_kernel import SpyreFp8LinearKernel

    return sum(1 for mod in model.modules() if _is_fp8_layer(mod, SpyreFp8LinearKernel))


def _check_prequant_cache(model: torch.nn.Module) -> list:  # noqa: F821
    """Return names of FP8 layers missing _qfp8wt_for_mm. Runs in worker."""
    from spyre_inference.custom_ops.fp8_linear_kernel import SpyreFp8LinearKernel

    missing = []
    for name, mod in model.named_modules():
        if not _is_fp8_layer(mod, SpyreFp8LinearKernel):
            continue
        if getattr(mod, "_qfp8wt_for_mm", None) is None:
            missing.append(name)
    return missing


# ---------------------------------------------------------------------------
# LLM factory helper
# ---------------------------------------------------------------------------


def _make_llm(monkeypatch: pytest.MonkeyPatch, *, prequant_force: str | None = None):
    """Create a compiled LLM instance for FP8 testing.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        prequant_force: ``"0"`` forces the fallback path (qfp8wt in-graph every
            forward). ``None`` (default) uses the prequant path.
    """
    from vllm import LLM
    from vllm.config import CompilationConfig

    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3600")
    # Required so apply_model() can serialise plain Python callables to the worker.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    if prequant_force is not None:
        monkeypatch.setenv("SPYRE_FP8_PREQUANT_FORCE", prequant_force)
    else:
        monkeypatch.delenv("SPYRE_FP8_PREQUANT_FORCE", raising=False)

    return LLM(
        model=_FP8_MODEL,
        dtype="float16",
        max_model_len=_MAX_MODEL_LEN,
        max_num_seqs=_MAX_NUM_SEQS,
        max_num_batched_tokens=_MAX_NUM_BATCHED_TOKENS,
        compilation_config=CompilationConfig(compile_sizes=[1, _MAX_NUM_BATCHED_TOKENS]),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.fp8
def test_fp8_model_generates_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """FP8 model loads and produces non-empty output."""
    from vllm import SamplingParams

    llm = _make_llm(monkeypatch)
    output = llm.generate(
        _PROMPT,
        SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS),
        use_tqdm=False,
    )
    assert len(output) == 1
    text = output[0].outputs[0].text
    assert len(text) > 0, f"Model produced empty output for prompt: {_PROMPT!r}"


@pytest.mark.fp8
def test_fp8_kernel_selected(monkeypatch: pytest.MonkeyPatch) -> None:
    """SpyreFp8LinearKernel is registered and selected for all FP8 linears.

    Uses ``llm.apply_model()`` to inspect the model inside the EngineCore
    subprocess — the only reliable introspection API in vLLM v1.
    """
    llm = _make_llm(monkeypatch)
    results = llm.apply_model(_count_fp8_layers)
    # apply_model returns a list (one entry per worker rank).
    count = results[0] if isinstance(results, list) else results
    assert count > 0, "No SpyreFp8LinearKernel layers found — kernel registration may have failed"


@pytest.mark.fp8
def test_fp8_prequant_cache_populated_after_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After warmup, all FP8 linear layers have _qfp8wt_for_mm set (prequant path).

    Uses ``llm.apply_model()`` for in-worker introspection.
    """
    from vllm import SamplingParams

    llm = _make_llm(monkeypatch)

    # Trigger warmup (one forward pass is enough).
    llm.generate(
        _PROMPT,
        SamplingParams(temperature=0.0, max_tokens=1),
        use_tqdm=False,
    )

    results = llm.apply_model(_check_prequant_cache)
    missing = results[0] if isinstance(results, list) else results
    assert not missing, (
        f"_qfp8wt_for_mm not populated on {len(missing)} FP8 layer(s) after warmup: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
    )


@pytest.mark.fp8
def test_fp8_prequant_matches_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prequant (default) and fallback (FORCE=0) produce the same token ids.

    Loads the model twice — default path and SPYRE_FP8_PREQUANT_FORCE=0 — and
    asserts greedy token ids are identical.
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=_MAX_TOKENS)

    # --- prequant path (default) ---
    llm_pre = _make_llm(monkeypatch)
    out_pre = llm_pre.generate(_PROMPT, sp, use_tqdm=False)
    tokens_pre = list(out_pre[0].outputs[0].token_ids)
    del llm_pre

    # --- fallback path ---
    llm_fall = _make_llm(monkeypatch, prequant_force="0")
    out_fall = llm_fall.generate(_PROMPT, sp, use_tqdm=False)
    tokens_fall = list(out_fall[0].outputs[0].token_ids)
    del llm_fall

    assert tokens_pre == tokens_fall, (
        f"Prequant and fallback token ids differ:\n"
        f"  prequant : {tokens_pre}\n"
        f"  fallback : {tokens_fall}"
    )
