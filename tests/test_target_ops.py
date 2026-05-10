# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import KVCache, QuantizedKVCache, RotatingKVCache

from dflash_mlx.engine.target_ops import bind_draft_to_target, resolve_target_ops
from dflash_mlx.engine.target_gemma4 import Gemma4TargetOps
from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

class _FakeLinearAttn:
    conv_kernel_size = 4

class _FakeGdnLayer:
    is_linear = True
    linear_attn = _FakeLinearAttn()

class _FakeFaLayer:
    is_linear = False
    self_attn = object()

class _FakeEmbed:
    pass

class _FakeTarget:
    def __init__(self, *, model_type: str = "qwen3_5") -> None:
        self.args = SimpleNamespace(model_type=model_type, tie_word_embeddings=True)
        self.model = SimpleNamespace(
            layers=[_FakeFaLayer(), _FakeGdnLayer(), _FakeFaLayer()],
            embed_tokens=_FakeEmbed(),
        )

class _FakePureAttentionTarget:
    def __init__(self, *, model_type: str = "qwen3") -> None:
        self.args = SimpleNamespace(model_type=model_type, tie_word_embeddings=True)
        self.model = SimpleNamespace(
            layers=[_FakeFaLayer(), _FakeFaLayer()],
            embed_tokens=_FakeEmbed(),
        )

class _FakeGemmaInner:
    embed_tokens = _FakeEmbed()

    def __init__(self, layer_types: list[str]) -> None:
        self.layers = [SimpleNamespace(layer_type=layer_type) for layer_type in layer_types]

class _FakeGemmaWrapper:
    def __init__(self, *, layer_types: list[str] | None = None) -> None:
        layer_types = layer_types or [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        self.args = SimpleNamespace(
            model_type="gemma4_text",
            layer_types=layer_types,
            sliding_window=1024,
            num_kv_shared_layers=0,
            tie_word_embeddings=True,
        )
        self.model = _FakeGemmaInner(layer_types)

    def make_cache(self):
        caches = []
        for layer_type in self.args.layer_types:
            if layer_type == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(RotatingKVCache(max_size=self.args.sliding_window, keep=0))
        return caches

class _FakeGemmaTarget:
    def __init__(self) -> None:
        self.args = SimpleNamespace(model_type="gemma4")
        self.language_model = _FakeGemmaWrapper()

class _FakeDraft:
    def __init__(self) -> None:
        self.bound = []

    def bind_target_model(self, target_model, *, target_ops):
        self.bound.append((target_model, target_ops))

class _FakeAsLinearEmbed:
    def as_linear(self, hidden):
        return hidden + 2.0

class _FakeLmHead:
    def __call__(self, hidden):
        return (hidden * 3.0) - 1.0

class _FakeGemmaLogitInner:
    embed_tokens = _FakeAsLinearEmbed()

class _FakeGemmaLogitWrapper:
    def __init__(
        self,
        *,
        tied: bool,
        softcap: float | None = None,
        args_softcap: float | None = None,
    ) -> None:
        self.args = SimpleNamespace(
            model_type="gemma4_text",
            layer_types=["full_attention"],
            final_logit_softcapping=args_softcap,
        )
        self.model = _FakeGemmaLogitInner()
        self.tie_word_embeddings = tied
        self.lm_head = _FakeLmHead()
        if softcap is not None:
            self.final_logit_softcapping = softcap

class _FakeGemmaLogitTarget:
    def __init__(
        self,
        *,
        tied: bool,
        softcap: float | None = None,
        args_softcap: float | None = None,
    ) -> None:
        self.args = SimpleNamespace(model_type="gemma4")
        self.language_model = _FakeGemmaLogitWrapper(
            tied=tied,
            softcap=softcap,
            args_softcap=args_softcap,
        )

def _assert_close(actual, expected, *, atol: float = 1e-6) -> None:
    mx.eval(actual, expected)
    assert float(mx.max(mx.abs(actual - expected)).item()) <= atol

def test_resolver_selects_qwen_ops_for_current_target_shape():
    ops = resolve_target_ops(_FakeTarget())

    assert isinstance(ops, QwenGdnTargetOps)
    assert ops.family(_FakeTarget()) == "hybrid_gdn"

def test_resolver_selects_gemma4_ops_for_gemma4_text_shape():
    ops = resolve_target_ops(_FakeGemmaTarget())

    assert isinstance(ops, Gemma4TargetOps)
    assert ops.family(_FakeGemmaTarget()) == "gemma4_swa"

def test_bind_draft_to_target_invokes_optional_target_binding():
    target = _FakeGemmaTarget()
    target_ops = Gemma4TargetOps()
    draft = _FakeDraft()

    returned = bind_draft_to_target(draft, target, target_ops=target_ops)

    assert returned is target_ops
    assert draft.bound == [(target, target_ops)]

def test_resolver_rejects_known_unsupported_family_markers():
    for model_type in ("llama", "mistral", "olmo"):
        target = _FakeTarget(model_type=model_type)

        try:
            resolve_target_ops(target)
        except NotImplementedError as exc:
            message = str(exc)
            assert f"model_type={model_type}" in message
            assert "model_class=_FakeTarget" in message
            assert "supported target backends: qwen_gdn, gemma4" in message
        else:
            raise AssertionError(f"{model_type} marker must not resolve to Qwen ops")

def test_resolver_rejects_unknown_model_type():
    try:
        resolve_target_ops(_FakeTarget(model_type="unknown_arch"))
    except NotImplementedError as exc:
        assert "model_type=unknown_arch" in str(exc)
    else:
        raise AssertionError("unknown non-empty model_type must not resolve to Qwen ops")

def test_resolver_rejects_empty_model_type_even_with_qwen_like_shape():
    try:
        resolve_target_ops(_FakeTarget(model_type=""))
    except NotImplementedError as exc:
        assert "model_type=unknown" in str(exc)
    else:
        raise AssertionError("empty model_type must not resolve to Qwen ops")

def test_capabilities_for_distinguishes_hybrid_and_pure_attention():
    ops = QwenGdnTargetOps()

    hybrid = ops.capabilities_for(_FakeTarget())
    pure = ops.capabilities_for(_FakePureAttentionTarget())

    assert hybrid.supports_recurrent_rollback is True
    assert hybrid.supports_dflash is True
    assert hybrid.supports_prefix_snapshot is True
    assert pure.supports_recurrent_rollback is False
    assert pure.supports_dflash is True
    assert pure.supports_kv_trim is True
    assert pure.supports_rotating_cache_snapshot is True
    assert pure.supports_shared_kv is False

def test_gemma4_capabilities_disable_prefix_snapshot_initially():
    caps = Gemma4TargetOps().capabilities_for(_FakeGemmaTarget())

    assert caps.supports_dflash is True
    assert caps.supports_recurrent_rollback is False
    assert caps.supports_kv_trim is True
    assert caps.supports_prefix_snapshot is False
    assert caps.supports_target_hidden_capture is True
    assert caps.supports_verify_linear is True


def test_gemma4_make_cache_matches_official_shared_kv_cache_types():
    from mlx_lm.models import gemma4_text

    args = gemma4_text.ModelArgs(
        hidden_size=16,
        num_hidden_layers=6,
        intermediate_size=32,
        num_attention_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_key_value_heads=1,
        num_kv_shared_layers=2,
        vocab_size=128,
        vocab_size_per_layer_input=128,
        hidden_size_per_layer_input=0,
        sliding_window=16,
        sliding_window_pattern=3,
        use_double_wide_mlp=False,
        attention_k_eq_v=False,
        final_logit_softcapping=None,
        tie_word_embeddings=True,
    )
    official = gemma4_text.Model(args)
    target = SimpleNamespace(
        args=SimpleNamespace(model_type="gemma4"),
        language_model=official,
    )

    actual = Gemma4TargetOps().make_cache(
        target,
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )
    expected = official.make_cache()

    assert len(actual) == len(expected) == 4
    assert [type(cache) for cache in actual] == [type(cache) for cache in expected]
    assert [type(cache).__name__ for cache in actual] == [
        "RotatingKVCache",
        "RotatingKVCache",
        "KVCache",
        "RotatingKVCache",
    ]

def test_gemma4_logits_use_tied_embedding_head():
    target = _FakeGemmaLogitTarget(tied=True)
    hidden = mx.array([[[1.0, -2.0, 3.0]]])

    logits = Gemma4TargetOps().logits_from_hidden(target, hidden)
    expected = target.language_model.model.embed_tokens.as_linear(hidden)

    _assert_close(logits, expected)

def test_gemma4_logits_use_untied_lm_head_for_a4b_style_target():
    target = _FakeGemmaLogitTarget(tied=False)
    hidden = mx.array([[[1.0, -2.0, 3.0]]])

    logits = Gemma4TargetOps().logits_from_hidden(target, hidden)
    expected = target.language_model.lm_head(hidden)

    _assert_close(logits, expected)

def test_gemma4_logits_apply_final_softcap_after_native_projection():
    target = _FakeGemmaLogitTarget(tied=False, softcap=2.5)
    hidden = mx.array([[[4.0, -4.0, 0.25]]])

    native_logits = target.language_model.lm_head(hidden)
    logits = Gemma4TargetOps().logits_from_hidden(target, hidden)
    expected = mx.tanh(native_logits / 2.5) * 2.5

    _assert_close(logits, expected)

def test_gemma4_logits_fall_back_to_args_final_softcap():
    target = _FakeGemmaLogitTarget(tied=False, args_softcap=2.5)
    hidden = mx.array([[[4.0, -4.0, 0.25]]])

    native_logits = target.language_model.lm_head(hidden)
    logits = Gemma4TargetOps().logits_from_hidden(target, hidden)
    expected = mx.tanh(native_logits / 2.5) * 2.5

    _assert_close(logits, expected)

def test_qwen_make_cache_matches_current_hybrid_cache_types(monkeypatch):
    monkeypatch.setattr(QwenGdnTargetOps, "install_speculative_hooks", lambda self, _model: None)
    caches = QwenGdnTargetOps().make_cache(
        _FakeTarget(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=0,
    )

    assert isinstance(caches[0], KVCache)
    assert isinstance(caches[1], RecurrentRollbackCache)
    assert isinstance(caches[2], KVCache)

def test_qwen_make_cache_quantizes_fa_only(monkeypatch):
    monkeypatch.setattr(
        QwenGdnTargetOps,
        "install_speculative_hooks",
        lambda self, _model: None,
    )
    caches = QwenGdnTargetOps().make_cache(
        _FakeTarget(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=True,
        target_fa_window=0,
    )

    assert isinstance(caches[0], QuantizedKVCache)
    assert isinstance(caches[1], RecurrentRollbackCache)
    assert isinstance(caches[2], QuantizedKVCache)

def test_speculative_gdn_hook_materializes_cached_state():
    root = Path(__file__).resolve().parents[1]
    text = (root / "dflash_mlx/engine/target_qwen_gdn.py").read_text()

    assert (
        "cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :])"
        in text
    )
    assert "cache[1] = mx.contiguous(state)" in text

def test_qwen_make_cache_windows_fa_only(monkeypatch):
    monkeypatch.setattr(QwenGdnTargetOps, "install_speculative_hooks", lambda self, _model: None)
    caches = QwenGdnTargetOps().make_cache(
        _FakeTarget(),
        enable_speculative_linear_cache=True,
        quantize_kv_cache=False,
        target_fa_window=2048,
    )

    assert isinstance(caches[0], RotatingKVCache)
    assert isinstance(caches[1], RecurrentRollbackCache)
    assert isinstance(caches[2], RotatingKVCache)

def test_runtime_and_spec_epoch_do_not_import_deleted_target_modules():
    root = Path(__file__).resolve().parents[1]
    runtime_text = (root / "dflash_mlx/runtime.py").read_text()
    spec_text = (root / "dflash_mlx/engine/spec_epoch.py").read_text()

    for text in (runtime_text, spec_text):
        assert "engine.target_verifier" not in text
        assert "engine.rollback" not in text

def test_model_backend_doc_mentions_workflow():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs/model_backend.md").read_text()

    assert "What is a TargetOps backend?" in text
    assert "register in `TARGET_BACKENDS`" in text
    assert "no backend per model size" in text
