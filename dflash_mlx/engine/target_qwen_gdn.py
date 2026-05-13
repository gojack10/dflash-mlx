# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import time
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import gated_delta as gated_delta_mod
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)

from dflash_mlx.engine.target_ops import TargetCapabilities
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

_EXACT_SMALL_PROJ_PAD_M = 16
_HYBRID_SDPA_EXACT_KV_THRESHOLD = 1024

class _ExactSmallProjPad(nn.Module):
    def __init__(self, linear: nn.Module, *, pad_m: int = _EXACT_SMALL_PROJ_PAD_M):
        super().__init__()
        self.linear = linear
        self.pad_m = int(pad_m)
        self._dflash_exact_small_proj_wrapped = True

    @property
    def weight(self) -> mx.array:
        return self.linear.weight

    @weight.setter
    def weight(self, value: mx.array) -> None:
        self.linear.weight = value

    @property
    def bias(self) -> Optional[mx.array]:
        return getattr(self.linear, "bias", None)

    @bias.setter
    def bias(self, value: Optional[mx.array]) -> None:
        self.linear.bias = value

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3 and x.shape[1] < self.pad_m:
            batch_size, seq_len, hidden_dim = x.shape
            pad = mx.zeros((batch_size, self.pad_m - seq_len, hidden_dim), dtype=x.dtype)
            out = self.linear(mx.concatenate([x, pad], axis=1))
            return out[:, :seq_len, :]
        return self.linear(x)

def _install_fused_gdn_projection(linear_attn: Any) -> None:
    """Install fused GDN projection weight. Single matmul replaces 4 dispatches.

    Concatenates in_proj_qkv, in_proj_z, in_proj_b, in_proj_a weights into one
    [16480, 5120] tensor. Stores transposed weight on linear_attn for matmul.
    Bit-exact with 4 separate calls, 1.9-2.1× faster.
    """
    if getattr(linear_attn, "_dflash_fused_proj_installed", False):
        return

    def _get_weight(proj):
        if isinstance(proj, _ExactSmallProjPad):
            return proj.linear.weight
        return proj.weight

    w_qkv = _get_weight(linear_attn.in_proj_qkv)
    w_z   = _get_weight(linear_attn.in_proj_z)
    w_b   = _get_weight(linear_attn.in_proj_b)
    w_a   = _get_weight(linear_attn.in_proj_a)

    n_qkv = int(w_qkv.shape[0])
    n_z   = int(w_z.shape[0])
    n_b   = int(w_b.shape[0])
    n_a   = int(w_a.shape[0])

    fused = mx.concatenate([w_qkv, w_z, w_b, w_a], axis=0)

    linear_attn._dflash_fused_weight_T = fused.swapaxes(-1, -2)
    linear_attn._dflash_fused_split_qkv = n_qkv
    linear_attn._dflash_fused_split_z   = n_qkv + n_z
    linear_attn._dflash_fused_split_b   = n_qkv + n_z + n_b
    linear_attn._dflash_fused_proj_installed = True


def _install_exact_small_proj_hooks(
    linear_attn: Any,
    *,
    pad_m: int = _EXACT_SMALL_PROJ_PAD_M,
) -> None:
    for attr_name in ("in_proj_b", "in_proj_a"):
        proj = getattr(linear_attn, attr_name, None)
        if proj is None or getattr(proj, "_dflash_exact_small_proj_wrapped", False):
            continue
        setattr(linear_attn, attr_name, _ExactSmallProjPad(proj, pad_m=pad_m))

def _attention_num_heads(attn: Any) -> int:
    for attr in ("num_attention_heads", "n_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing attention head count attribute")

def _attention_num_kv_heads(attn: Any) -> int:
    for attr in ("num_key_value_heads", "n_kv_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing KV head count attribute")

def _attention_has_gated_q_proj(attn: Any) -> bool:
    q_proj = getattr(attn, "q_proj", None)
    q_norm = getattr(attn, "q_norm", None)
    q_proj_weight = getattr(q_proj, "weight", None)
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_proj_weight is None or q_norm_weight is None:
        return False
    try:
        num_attention_heads = _attention_num_heads(attn)
    except AttributeError:
        return False
    expected_out_dim = 2 * num_attention_heads * int(q_norm_weight.shape[0])
    return int(q_proj_weight.shape[0]) == expected_out_dim

def _split_sdpa_mask(
    mask: Optional[Any],
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> Optional[Any]:
    if mask is None or mask == "causal":
        return mask
    return mask[..., query_start:query_end, :key_end]

def _split_sdpa_output(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any],
    chunk_size: int,
    cached_prefix_len: int,
) -> mx.array:
    q_len = int(queries.shape[2])
    if q_len <= chunk_size:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    outputs: list[mx.array] = []
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        key_end = cached_prefix_len + end
        chunk_mask = _split_sdpa_mask(mask, query_start=start, query_end=end, key_end=key_end)
        outputs.append(
            scaled_dot_product_attention(
                queries[:, :, start:end, :],
                keys[:, :, :key_end, :],
                values[:, :, :key_end, :],
                cache=cache,
                scale=scale,
                mask=chunk_mask,
            )
        )
    return mx.concatenate(outputs, axis=2)

def _install_speculative_linear_cache_hook(linear_attn: Any) -> None:
    cls = type(linear_attn)

    # Per-instance: build fused projection weight for this layer
    _install_fused_gdn_projection(linear_attn)

    if getattr(cls, "_dflash_speculative_call_installed", False):
        return

    original_call = cls.__call__

    def speculative_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        """Unified forward pass — always uses fused GDN projection matmul.

        Single (16480, 5120) matmul replaces 4 separate dispatches.
        Branches only on cache type (RecurrentRollbackCache vs other)
        and recurrence path (kernel vs ops, tape vs plain).
        """
        from mlx.nn.layers.distributed import sum_gradients

        B, S, _ = inputs.shape
        sharding_group = getattr(self, "sharding_group", None)
        is_rrb_cache = isinstance(cache, RecurrentRollbackCache)
        is_armed = is_rrb_cache and getattr(cache, "_armed", False)

        if sharding_group is not None:
            inputs = sum_gradients(sharding_group)(inputs)

        # --- Fused projection (one matmul replaces 4) ---
        fused_weight_T = getattr(self, "_dflash_fused_weight_T", None)
        if fused_weight_T is None:
            # Fallback for instances without fused weights (e.g., second model load)
            return original_call(self, inputs, mask=mask, cache=cache)

        fused = inputs @ fused_weight_T
        qkv = fused[..., :self._dflash_fused_split_qkv]
        z = fused[..., self._dflash_fused_split_qkv:self._dflash_fused_split_z].reshape(
            B, S, self.num_v_heads, self.head_v_dim)
        b = fused[..., self._dflash_fused_split_z:self._dflash_fused_split_b]
        a = fused[..., self._dflash_fused_split_b:]

        # --- Conv cache ---
        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)

        if is_rrb_cache:
            cache[0] = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1) :])
        elif cache is not None:
            n_keep = self.conv_kernel_size - 1
            if getattr(cache, "lengths", None) is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])

        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(B, S, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]

        # --- Recurrence ---
        state = cache[1] if cache is not None else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        g = gated_delta_mod.compute_g(self.A_log, a, self.dt_bias)
        beta = mx.sigmoid(b)

        if state is None:
            _, _, h_k, d_k = q.shape
            h_v, d_v = v.shape[-2:]
            state = mx.zeros((B, h_v, d_v, d_k), dtype=mx.float32)
        state_in = state

        if (
            mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and not self.training
        ):
            if is_armed:
                from dflash_mlx.kernels import gated_delta_kernel_with_tape

                out, state, innovation_tape = gated_delta_kernel_with_tape(
                    q, k, v, g, beta, state, mask
                )
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )
            else:
                out, state = gated_delta_mod.gated_delta_kernel(q, k, v, g, beta, state, mask)
        else:
            out, state = gated_delta_mod.gated_delta_ops(q, k, v, g, beta, state, mask)
            if is_armed:
                decay = g[..., None, :] if g.ndim == 4 else g[..., None, None]
                decayed_state = state_in[:, None, ...] * decay
                kv_mem = (decayed_state * k[..., None, :]).sum(axis=-1)
                innovation_tape = (v - kv_mem) * beta[..., None]
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )

        # --- Output ---
        if cache is not None:
            cache[1] = mx.contiguous(state)
            cache.advance(S)
        out = self.norm(out, z)
        out_flat = out.reshape(B, S, -1)
        out = self.out_proj(out_flat)

        if sharding_group is not None:
            out = mx.distributed.all_sum(out, group=sharding_group)

        return out

    cls.__call__ = speculative_call
    cls._dflash_speculative_call_installed = True

def _install_split_full_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_split_full_attention_installed", False):
        return

    original_call = cls.__call__

    def split_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not getattr(self, "_dflash_split_sdpa_enabled", False):
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        num_attention_heads = _attention_num_heads(self)
        num_key_value_heads = _attention_num_kv_heads(self)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        cached_prefix_len = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0
        if cache is not None:
            queries = self.rope(queries, offset=cached_prefix_len)
            keys = self.rope(keys, offset=cached_prefix_len)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        exact_prefix_threshold = int(
            getattr(
                self,
                "_dflash_split_sdpa_exact_kv_threshold",
                _HYBRID_SDPA_EXACT_KV_THRESHOLD,
            )
        )
        should_split = (
            cache is not None
            and cached_prefix_len >= exact_prefix_threshold
            and (mask is None or mask == "causal" or isinstance(mask, mx.array))
        )
        should_use_batched_2pass = (
            should_split
            and int(queries.shape[2]) == 16
            and queries.dtype in (mx.bfloat16, mx.float16)
            and int(queries.shape[-1]) in (128, 256)
            and int(values.shape[-1]) in (128, 256)
        )
        if should_use_batched_2pass:
            from dflash_mlx.kernels import batched_sdpa_2pass_exact

            output = batched_sdpa_2pass_exact(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask if isinstance(mask, mx.array) else None,
            )
            if output is None:
                output = _split_sdpa_output(
                    queries=queries,
                    keys=keys,
                    values=values,
                    scale=self.scale,
                    mask=mask,
                    cache=cache,
                    chunk_size=1,
                    cached_prefix_len=cached_prefix_len,
                )
        elif should_split:
            output = _split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask,
                cache=cache,
                chunk_size=1,
                cached_prefix_len=cached_prefix_len,
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gated_output = output * mx.sigmoid(gate)
        return self.o_proj(gated_output)

    cls.__call__ = split_call
    cls._dflash_split_full_attention_installed = True

class QwenGdnTargetOps:
    backend_name = "qwen_gdn"

    def model_type(self, target_model: Any) -> str:
        args = getattr(target_model, "args", None)
        if args is None and hasattr(target_model, "language_model"):
            args = getattr(target_model.language_model, "args", None)
        value = getattr(args, "model_type", None)
        if value is not None:
            return str(value).lower()
        config = getattr(target_model, "config", None)
        if isinstance(config, dict):
            return str(config.get("model_type", "")).lower()
        return ""

    def supports_model(self, target_model: Any) -> bool:
        model_type = self.model_type(target_model)
        if "qwen" in model_type:
            return self._has_qwen_text_shape(target_model)
        return False

    def _has_qwen_text_shape(self, target_model: Any) -> bool:
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        return hasattr(inner, "layers") and hasattr(inner, "embed_tokens")

    def text_wrapper(self, target_model: Any) -> Any:
        if hasattr(target_model, "model"):
            return target_model
        if hasattr(target_model, "language_model"):
            return target_model.language_model
        raise AttributeError(f"Unsupported target model wrapper: {type(target_model)!r}")

    def text_model(self, target_model: Any) -> Any:
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "model"):
            return wrapper.model
        raise AttributeError(f"Unsupported target text model: {type(wrapper)!r}")

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(self, target_model: Any, hidden_states: mx.array) -> mx.array:
        wrapper = self.text_wrapper(target_model)
        if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True):
            return wrapper.model.embed_tokens.as_linear(hidden_states)
        return wrapper.lm_head(hidden_states)

    def family(self, target_model: Any) -> str:
        inner = self.text_model(target_model)
        has_linear = any(
            hasattr(layer, "linear_attn") or getattr(layer, "is_linear", False)
            for layer in inner.layers
        )
        return "hybrid_gdn" if has_linear else "pure_attention"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        has_recurrent = self.family(target_model) == "hybrid_gdn"
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=has_recurrent,
            supports_kv_trim=True,
            supports_prefix_snapshot=True,
            supports_rotating_cache_snapshot=True,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
        )

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array],
        target_layer_ids: list[int],
    ) -> mx.array:
        selected = [captured_dict[layer_id + 1] for layer_id in target_layer_ids]
        return mx.concatenate(selected, axis=-1)

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        inner = self.text_model(target_model)
        hidden_states = input_embeddings if input_embeddings is not None else inner.embed_tokens(input_ids)
        if cache is None:
            cache = [None] * len(inner.layers)
        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [hidden_states]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: hidden_states} if 0 in capture_layer_ids else {}
        h = hidden_states

        if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
            fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
            ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
            for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
                mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
                h = layer(h, mask=mask, cache=layer_cache)
                capture_key = layer_index + 1
                if capture_all:
                    captured.append(h)
                elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                    captured[capture_key] = h
        else:
            mask = create_attention_mask(hidden_states, cache[0])
            for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
                h = layer(h, mask, layer_cache)
                capture_key = layer_index + 1
                if capture_all:
                    captured.append(h)
                elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                    captured[capture_key] = h
        normalized = inner.norm(h)
        logits = self.logits_from_hidden(target_model, normalized)
        return logits, captured

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        if int(verify_ids.shape[1]) <= 0:
            raise ValueError("verify block must contain at least one token")
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def install_speculative_hooks(self, target_model: Any) -> None:
        text_model = self.text_model(target_model)
        if getattr(text_model, "_dflash_speculative_hooks_installed", False):
            return
        if self.family(target_model) == "pure_attention":
            text_model._dflash_speculative_hooks_installed = True
            return
        for layer in text_model.layers:
            if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
                _install_exact_small_proj_hooks(layer.linear_attn)
                _install_speculative_linear_cache_hook(layer.linear_attn)
            elif not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
                _install_split_full_attention_hook(layer.self_attn)
        text_model._dflash_speculative_hooks_installed = True

    def configure_full_attention_split(
        self,
        target_model: Any,
        *,
        enabled: bool,
        chunk_size: int = 8,
    ) -> None:
        text_model = self.text_model(target_model)
        if self.family(target_model) == "pure_attention":
            return
        self.install_speculative_hooks(target_model)
        for layer in text_model.layers:
            if not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
                layer.self_attn._dflash_split_sdpa_enabled = enabled
                layer.self_attn._dflash_split_sdpa_chunk_size = int(chunk_size)
                layer.self_attn._dflash_split_sdpa_exact_kv_threshold = (
                    _HYBRID_SDPA_EXACT_KV_THRESHOLD
                )

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
    ) -> list[Any]:
        fa_window = 0 if target_fa_window is None else int(target_fa_window)
        if fa_window < 0:
            raise ValueError("target_fa_window must be >= 0")
        if fa_window > 0 and quantize_kv_cache:
            raise ValueError(
                "target_fa_window does not support quantized target KV cache"
            )
        text_model = self.text_model(target_model)
        caches: list[Any] = []
        for layer in text_model.layers:
            if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
                if enable_speculative_linear_cache:
                    self.install_speculative_hooks(target_model)
                    conv_kernel_size = int(getattr(layer.linear_attn, "conv_kernel_size", 4))
                    caches.append(
                        RecurrentRollbackCache(size=2, conv_kernel_size=conv_kernel_size)
                    )
                else:
                    caches.append(cache_mod.ArraysCache(size=2))
            else:
                if fa_window > 0:
                    caches.append(cache_mod.RotatingKVCache(max_size=fa_window))
                elif quantize_kv_cache:
                    caches.append(cache_mod.QuantizedKVCache(group_size=64, bits=8))
                else:
                    caches.append(cache_mod.KVCache())
        return caches

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        for cache_entry in cache_entries:
            if hasattr(cache_entry, "arm_rollback"):
                cache_entry.arm_rollback(prefix_len=int(prefix_len))

    def clear_rollback_state(self, cache_entry: Any) -> None:
        if hasattr(cache_entry, "clear_transients"):
            cache_entry.clear_transients()
            return
        if hasattr(cache_entry, "_armed"):
            cache_entry._armed = False
        if hasattr(cache_entry, "_tape"):
            cache_entry._tape = None
        if hasattr(cache_entry, "_tape_k"):
            cache_entry._tape_k = None
        if hasattr(cache_entry, "_tape_g"):
            cache_entry._tape_g = None
        if hasattr(cache_entry, "_tape_qkv"):
            cache_entry._tape_qkv = None
        if hasattr(cache_entry, "_snapshot"):
            cache_entry._snapshot = None

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        replay_ns_total = 0
        fully_accepted = acceptance_length == drafted_tokens
        for cache_entry in cache_entries:
            if hasattr(cache_entry, "rollback"):
                if fully_accepted:
                    self.clear_rollback_state(cache_entry)
                    continue
                replay_start_ns = time.perf_counter_ns()
                cache_entry.rollback(acceptance_length)
                replay_ns_total += time.perf_counter_ns() - replay_start_ns
            elif hasattr(cache_entry, "trim"):
                offset = int(getattr(cache_entry, "offset", 0) or 0)
                if offset > target_len:
                    replay_start_ns = time.perf_counter_ns()
                    cache_entry.trim(offset - target_len)
                    replay_ns_total += time.perf_counter_ns() - replay_start_ns
            elif hasattr(cache_entry, "offset"):
                offset = int(getattr(cache_entry, "offset", 0) or 0)
                if offset > target_len:
                    cache_entry.offset = target_len
            elif hasattr(cache_entry, "crop"):
                cache_entry.crop(target_len)
        return replay_ns_total

    def cleanup_generation_caches(
        self,
        target_cache: list[Any],
        draft_cache: list[Any],
    ) -> None:
        for cache_entry in target_cache:
            if hasattr(cache_entry, "clear_transients"):
                cache_entry.clear_transients()
        draft_cache.clear()
        target_cache.clear()
