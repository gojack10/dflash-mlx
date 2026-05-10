# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import json
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load, load_model

from dflash_mlx.internal_debug import (
    verify_linear_override as _debug_verify_linear_override,
    verify_qmm_enabled as _debug_verify_qmm_enabled,
)
from dflash_mlx.engine.config import (
    _effective_draft_window_size,
)
from dflash_mlx.engine.target_ops import resolve_target_ops
from dflash_mlx.model import (
    DFlashDraftModel,
    DFlashDraftModelArgs,
)

def resolve_model_ref(model_ref: str | Path | None, *, kind: str) -> str:
    if model_ref:
        candidate = Path(model_ref).expanduser()
        return str(candidate if candidate.exists() else model_ref)
    raise ValueError(f"{kind} model reference is required")

def get_stop_token_ids(tokenizer: Any) -> list[int]:
    eos_token_ids = list(getattr(tokenizer, "eos_token_ids", None) or [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and eos_token_id not in eos_token_ids:
        eos_token_ids.append(int(eos_token_id))
    return eos_token_ids

def default_split_sdpa_enabled(model_ref: str | Path | None) -> bool:
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    return False

def _get_dflash_model_classes(config: dict[str, Any]):
    return DFlashDraftModel, DFlashDraftModelArgs

def _resolve_local_model_path(model_ref: str | Path) -> Path:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return candidate
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise FileNotFoundError(f"Model path does not exist and huggingface_hub is unavailable: {model_ref}") from exc

    snapshot_path = snapshot_download(
        repo_id=str(model_ref),
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "tokenizer*"],
    )
    return Path(snapshot_path)

def _prepare_prompt_tokens(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))

def build_suppress_token_mask(
    vocab_size: int,
    suppress_token_ids: Optional[list[int]],
) -> Optional[mx.array]:
    token_ids = sorted(
        {
            int(token_id)
            for token_id in (suppress_token_ids or [])
            if 0 <= int(token_id) < vocab_size
        }
    )
    if not token_ids:
        return None
    vocab_indices = mx.arange(vocab_size, dtype=mx.int32)
    token_array = mx.array(token_ids, dtype=mx.int32)
    return mx.any(mx.equal(vocab_indices[:, None], token_array[None, :]), axis=1)

def greedy_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    floor = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(suppress_token_mask, floor, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)


def apply_repetition_penalty(
    logits: mx.array,
    generated_token_ids: list[int],
    penalty: float,
) -> mx.array:
    """Apply repetition penalty to logits in-place (lazy).

    Penalizes tokens that have already appeared in the full conversation
    (prompt + generated tokens). For each such token, divide positive logits
    or multiply negative logits by *penalty*.  A penalty of 1.0 is a no-op.
    """
    if penalty == 1.0 or not generated_token_ids:
        return logits
    unique_ids = sorted(set(generated_token_ids))
    if not unique_ids:
        return logits
    # Build a boolean mask for tokens that should be penalized
    vocab_size = int(logits.shape[-1])
    valid_ids = [tid for tid in unique_ids if 0 <= tid < vocab_size]
    if not valid_ids:
        return logits
    idx = mx.array(valid_ids, dtype=mx.int32)
    # Fetch penalized logits, apply penalty, scatter back
    penalized = mx.take(logits, idx, axis=-1)
    penalized = mx.where(
        penalized > 0,
        penalized / penalty,
        penalized * penalty,
    )
    scatter_idx = idx.reshape((1,) * (logits.ndim - 1) + (-1,))
    return mx.put_along_axis(logits, scatter_idx, penalized, axis=-1)

def compute_target_probs(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float = 1.0,
    suppress_token_mask: Optional[mx.array] = None,
    generated_token_ids: Optional[list[int]] = None,
    rep_penalty: float = 1.0,
) -> mx.array:
    """Compute target probability distribution at temperature T, with top-p truncation.

    Applies (in order): repetition penalty → suppress mask → divide by T → softmax →
    top-p truncation → renormalize. Output sums to 1 along the last axis.

    Args:
        logits: shape [..., vocab].
        temperature: > 0.
        top_p: in (0, 1]. 1.0 means no truncation.
        suppress_token_mask: optional [vocab] bool, True = suppress.
        generated_token_ids: history for repetition penalty.
        rep_penalty: 1.0 means no penalty.
    """
    if rep_penalty != 1.0 and generated_token_ids:
        logits = apply_repetition_penalty(logits, generated_token_ids, rep_penalty)
    if suppress_token_mask is not None:
        floor = mx.array(-1e9, dtype=logits.dtype)
        logits = mx.where(suppress_token_mask, floor, logits)
    scaled = logits / float(temperature)
    probs = mx.softmax(scaled, axis=-1)
    if top_p < 1.0:
        probs = _apply_top_p(probs, top_p)
    return probs


def _apply_top_p(probs: mx.array, top_p: float) -> mx.array:
    """Top-p (nucleus) truncation: keep smallest set of tokens whose mass ≥ top_p, renormalize."""
    # Sort descending along last axis, find cutoff
    sort_idx = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sort_idx, axis=-1)
    cumsum = mx.cumsum(sorted_probs, axis=-1)
    # Keep tokens where cumsum (before adding this token) < top_p.
    # Equivalent: keep if (cumsum - sorted_probs) < top_p.
    keep_sorted = (cumsum - sorted_probs) < float(top_p)
    # Scatter the keep mask back to original positions.
    keep = mx.zeros(probs.shape, dtype=mx.bool_)
    keep = mx.put_along_axis(keep, sort_idx, keep_sorted, axis=-1)
    truncated = mx.where(keep, probs, mx.zeros(probs.shape, dtype=probs.dtype))
    # Renormalize
    norm = mx.sum(truncated, axis=-1, keepdims=True)
    # If norm is zero (shouldn't happen because top-1 is always kept), fall back to original.
    safe_norm = mx.where(norm > 0, norm, mx.ones_like(norm))
    return truncated / safe_norm


def sample_with_temperature(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float = 1.0,
    suppress_token_mask: Optional[mx.array] = None,
    generated_token_ids: Optional[list[int]] = None,
    rep_penalty: float = 1.0,
) -> int:
    """Sample a single token from a single logit row using temp + top_p.

    Args:
        logits: shape [vocab] (1D).
    Returns:
        Sampled token id (int).
    """
    if logits.ndim != 1:
        logits = logits.reshape(-1)
    probs = compute_target_probs(
        logits,
        temperature=temperature,
        top_p=top_p,
        suppress_token_mask=suppress_token_mask,
        generated_token_ids=generated_token_ids,
        rep_penalty=rep_penalty,
    )
    log_probs = mx.log(probs + 1e-30)
    sampled = mx.random.categorical(log_probs[None, :])
    return int(sampled.item())


def _sample_categorical(p_row: mx.array) -> int:
    """Sample one token from a probability row [vocab] using mx.random.categorical.

    Falls back to argmax if probability sums to ~0 (pathological case).
    """
    total = float(mx.sum(p_row).item())
    if total < 1e-20:
        return int(mx.argmax(p_row).item())
    log_probs = mx.log(p_row + 1e-30)
    sampled = mx.random.categorical(log_probs[None, :])
    return int(sampled.item())


def stochastic_tree_walk(
    child_maps: list[dict],
    sampled_per_node: list[int],
) -> tuple[list[int], int]:
    """Walk a draft tree following tokens sampled from the target distribution.

    Each emitted token is drawn directly from the target softmax at the
    appropriate tree position; the walk descends a child whenever the sample
    matches a draft proposal, otherwise the sample becomes the bonus token and
    the walk terminates. Because every emitted token comes from p(·) at the
    correct position, the resulting sequence is distributed exactly as the
    target — this is the standard speculative-sampling formulation used by
    production engines (Medusa / EAGLE-2 / Specinfer fallback path).

    Args:
        child_maps: per-node dict {token_id: child_node_index}.
        sampled_per_node: pre-sampled target token at each tree node (one int per
            node, drawn from target_probs[node]). Caller pre-samples on GPU in
            one batch and passes the resulting list to avoid per-node sync.

    Returns:
        (accepted_indices, bonus_token_id) — accepted_indices always starts with [0].
    """
    accepted: list[int] = [0]
    current = 0
    while child_maps[current]:
        sampled_tok = int(sampled_per_node[current])
        if sampled_tok in child_maps[current]:
            current = child_maps[current][sampled_tok]
            accepted.append(current)
        else:
            return accepted, sampled_tok
    # Reached a leaf node.
    return accepted, int(sampled_per_node[current])


def stochastic_linear_walk(
    sampled_per_pos: list[int],
    draft_token_ids: list[int],
) -> tuple[int, int]:
    """Linear analogue of stochastic_tree_walk: accept while target sample matches draft.

    Args:
        sampled_per_pos: target samples at each position (length = block_len).
        draft_token_ids: draft's chosen tokens (length = block_len - 1 typically;
            matches existing acceptance semantics where position k's draft is
            compared to position k's target).
    Returns:
        (acceptance_length, bonus_token_id).
    """
    n_compare = min(len(sampled_per_pos), len(draft_token_ids))
    for i in range(n_compare):
        if int(sampled_per_pos[i]) != int(draft_token_ids[i]):
            return i, int(sampled_per_pos[i])
    bonus_idx = min(n_compare, len(sampled_per_pos) - 1)
    return n_compare, int(sampled_per_pos[bonus_idx])


def batched_sample_from_probs(target_probs: mx.array) -> mx.array:
    """Sample one token per row of a probability tensor in a single GPU call.

    Args:
        target_probs: [N, vocab] probabilities (each row sums to 1).
    Returns:
        [N] sampled token ids as mx.uint32.
    """
    # mx.random.categorical takes logits. log(p + eps) keeps numerical stability.
    log_probs = mx.log(target_probs + 1e-30)
    return mx.random.categorical(log_probs, axis=-1).astype(mx.uint32)


def _eval_logits_and_captured(
    logits: mx.array,
    captured: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)

def _ns_to_us(ns: int | float) -> float:
    return float(ns) / 1_000.0

@dataclass(frozen=True)
class VerifyConfig:
    mode: str = "auto"
    enable_qmm: bool = True

    @classmethod
    def from_mode(cls, mode: str | None) -> "VerifyConfig":
        resolved = (mode or "auto").strip().lower()
        if resolved not in ("auto", "off"):
            raise ValueError("verify mode must be auto or off")
        return cls(mode=resolved)

@dataclass(frozen=True)
class DraftQuantSpec:
    weight_bits: int
    group_size: int
    act_bits: int

_DRAFT_QUANT_RE = re.compile(
    r"^w(?P<wb>2|4|8)"
    r"(?:a(?P<ab>16|32))?"
    r"(?::gs(?P<gs>32|64|128))?$",
    re.IGNORECASE,
)

def parse_draft_quant_spec(spec: str) -> DraftQuantSpec:
    m = _DRAFT_QUANT_RE.match(spec.strip())
    if not m:
        raise ValueError(
            f"Invalid draft quant spec {spec!r}. "
            "Expected format: w4, w8a16, w4a32:gs128, etc. "
            "Weight bits: 2, 4, 8. Activation bits: 16 (bfloat16) or 32 (float32). "
            "Group size: 32, 64, 128."
        )
    wb = int(m.group("wb"))
    ab = int(m.group("ab") or 16)
    gs = int(m.group("gs") or 64)
    return DraftQuantSpec(weight_bits=wb, group_size=gs, act_bits=ab)

def _resolve_draft_quant(draft_quant: str | None) -> DraftQuantSpec | None:
    spec = (draft_quant or "").strip()
    if not spec:
        return None
    return parse_draft_quant_spec(spec)


def _draft_lm_head_weight_names(model_path: Path) -> list[str]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text())
            weight_map = payload.get("weight_map", {})
            return sorted(
                str(name)
                for name in weight_map
                if str(name).startswith("lm_head.")
            )
        except Exception:
            return []
    try:
        from safetensors import safe_open
    except Exception:
        return []
    names: list[str] = []
    for path in sorted(model_path.glob("model*.safetensors")):
        with safe_open(str(path), framework="mlx") as handle:
            names.extend(
                str(name)
                for name in handle.keys()
                if str(name).startswith("lm_head.")
            )
    return sorted(names)


def _raise_if_unsupported_draft_lm_head(model_path: Path, model_ref: str) -> None:
    lm_head_names = _draft_lm_head_weight_names(model_path)
    if not lm_head_names:
        return
    sample = ", ".join(lm_head_names[:3])
    if len(lm_head_names) > 3:
        sample += ", ..."
    raise ValueError(
        f"DFlash draft checkpoint '{model_ref}' contains draft-owned lm_head weights "
        f"({sample}), but this runtime projects draft hidden states through the target "
        "TargetOps.logits_from_hidden(...) path and does not load a draft lm_head yet."
    )

def pack_target_model_weights_selective(
    target_model: Any,
    *,
    validate: bool = True,
    pack_mlp: bool = True,
    pack_attention: bool = False,
) -> dict[str, Any]:
    text_model = resolve_target_ops(target_model).text_model(target_model)
    if getattr(text_model, "_dflash_pack_info", None) is not None:
        return text_model._dflash_pack_info

    pack_info = {
        "enabled": True,
        "validated": validate,
        "pack_mlp": pack_mlp,
        "pack_attention": pack_attention,
        "packed_mlp_layers": [],
        "packed_attention_layers": [],
    }
    text_model._dflash_pack_info = pack_info
    return pack_info

def load_target_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    pack_target_weights: bool = False,
    pack_attention_weights: bool = False,
    validate_packing: bool = True,
    split_full_attention_sdpa: Optional[bool] = None,
    split_full_attention_chunk_size: int = 8,
    quantize_kv_cache: bool = False,
    verify_config: VerifyConfig | None = None,
):
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    split_enabled = (
        default_split_sdpa_enabled(resolved_ref)
        if split_full_attention_sdpa is None
        else bool(split_full_attention_sdpa)
    )
    model, tokenizer, config = load(resolved_ref, lazy=lazy, return_config=True)
    target_ops = resolve_target_ops(model)
    target_family = target_ops.family(model)
    target_capabilities = target_ops.capabilities_for(model)
    if target_family == "hybrid_gdn":
        target_ops.install_speculative_hooks(model)
        target_ops.configure_full_attention_split(
            model,
            enabled=split_enabled and not quantize_kv_cache,
            chunk_size=split_full_attention_chunk_size,
        )
    meta = {
        "resolved_model_ref": resolved_ref,
        "config": config,
        "quantize_kv_cache": bool(quantize_kv_cache),
        "target_family": target_family,
        "split_full_attention_sdpa": bool(split_enabled and not quantize_kv_cache),
        "split_full_attention_sdpa_requested": split_full_attention_sdpa,
    }
    if pack_target_weights:
        meta["packing"] = pack_target_model_weights_selective(
            model,
            validate=validate_packing,
            pack_mlp=True,
            pack_attention=pack_attention_weights,
        )
    verify_linear_enabled = (
        bool(getattr(target_capabilities, "supports_verify_linear", True))
        and _verify_enabled_for(config, verify_config=verify_config)
    )
    meta["verify_linear_enabled"] = bool(verify_linear_enabled)
    meta["verify_mode"] = (verify_config.mode if verify_config is not None else "env")
    if verify_linear_enabled:
        from dflash_mlx.verify_linear import install_verify_linears
        n_swapped = install_verify_linears(
            model,
            enable_qmm=_verify_qmm_enabled(verify_config),
        )
        meta["verify_linear_swapped"] = n_swapped
    return model, tokenizer, meta


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _text_config(config: Any) -> Any:
    return _config_value(config, "text_config", config)


def _verify_enabled_for(
    config: Any,
    *,
    verify_config: VerifyConfig | None = None,
) -> bool:
    text_cfg = _text_config(config)
    if verify_config is not None:
        if verify_config.mode == "off":
            return False
    else:
        override = _debug_verify_linear_override()
        if override is not None:
            return override
    try:
        num_experts = int(_config_value(text_cfg, "num_experts", 0) or 0)
        num_layers = int(_config_value(text_cfg, "num_hidden_layers", 0) or 0)
        hidden_size = int(_config_value(text_cfg, "hidden_size", 0) or 0)
        num_heads = int(_config_value(text_cfg, "num_attention_heads", 0) or 0)
        num_kv_heads = int(_config_value(text_cfg, "num_key_value_heads", 0) or 0)
    except Exception:
        return False
    if num_experts > 0:
        if (
            num_layers == 40
            and hidden_size == 2048
            and num_heads == 16
            and num_kv_heads == 2
        ):
            return False
        return True
    return num_layers >= 40

def _verify_qmm_enabled(verify_config: VerifyConfig | None) -> bool:
    if verify_config is not None:
        return bool(verify_config.enable_qmm)
    return _debug_verify_qmm_enabled()

def load_draft_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    draft_quant: str | None = None,
):
    resolved_ref = resolve_model_ref(model_ref, kind="draft")
    model_path = _resolve_local_model_path(resolved_ref)
    _raise_if_unsupported_draft_lm_head(model_path, str(resolved_ref))
    model, config = load_model(
        model_path,
        lazy=lazy,
        get_model_classes=_get_dflash_model_classes,
    )
    quant_spec = _resolve_draft_quant(draft_quant)
    if quant_spec is not None:
        nn.quantize(model, bits=quant_spec.weight_bits, group_size=quant_spec.group_size)
        if quant_spec.weight_bits in (4, 8):
            from dflash_mlx.verify_linear import install_verify_linears, prewarm_verify_kernels
            install_verify_linears(model, enable_qmm=True)
            prewarm_verify_kernels(model)
        if quant_spec.act_bits == 32:
            def _cast_to_f32(_, x: mx.array) -> mx.array:
                if x.dtype not in (mx.uint32, mx.int32):
                    return x.astype(mx.float32)
                return x
            model.apply(_cast_to_f32)
    return model, {
        "resolved_model_ref": str(model_ref) if model_ref is not None else str(resolved_ref),
        "config": config,
        "draft_quant": (
            {
                "weight_bits": quant_spec.weight_bits,
                "group_size": quant_spec.group_size,
                "act_bits": quant_spec.act_bits,
            }
            if quant_spec is not None
            else None
        ),
    }

def stream_dflash_generate(**kwargs: Any) -> Iterator[dict[str, Any]]:
    if kwargs.get("runtime_context") is None:
        raise ValueError("runtime_context is required")
    gen_stream = mx.default_stream(mx.default_device())
    with mx.stream(gen_stream):
        yield from _stream_dflash_generate_impl(**kwargs)

from dflash_mlx.engine.fallback import stream_baseline_generate
from dflash_mlx.engine.spec_epoch import (
    stream_dflash_generate_impl as _stream_dflash_generate_impl,
)
