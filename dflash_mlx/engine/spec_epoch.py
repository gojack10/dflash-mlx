# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.cache.codecs import hydrate_target_cache
from dflash_mlx.cache.snapshot import (
    DFlashPrefixSnapshot,
    validate_prefix_snapshot as _validate_prefix_snapshot,
)
from dflash_mlx.draft_backend import make_draft_backend
from dflash_mlx.engine.acceptance import match_acceptance_length as _match_acceptance_length
from dflash_mlx.engine.fallback import stream_baseline_generate
from dflash_mlx.engine.prefill import (
    compute_snapshot_boundary,
    init_target_hidden_from_snapshot,
)
from dflash_mlx.engine.config import (
    _profile_dflash_cycles_enabled,
    resolve_draft_window,
    resolve_verify_len_cap,
    verify_token_count_for_block,
)
from dflash_mlx.engine.target_ops import bind_draft_to_target, resolve_target_ops
from dflash_mlx.model import DFlashDraftModel
from dflash_mlx.engine.memory_waterfall import (
    collect_memory_waterfall as _collect_memory_waterfall,
    memory_waterfall_enabled as _memory_waterfall_enabled,
    should_sample_cycle as _should_sample_memory_cycle,
)


def _clear_cache_transients(cache_entry: Any) -> None:
    clear = getattr(cache_entry, "clear_transients", None)
    if clear is not None:
        clear()
        return
    for attr in ("_armed", "_tape", "_tape_k", "_tape_g", "_tape_qkv", "_snapshot"):
        if hasattr(cache_entry, attr):
            setattr(cache_entry, attr, False if attr == "_armed" else None)


def _auto_draft_block_tokens(draft_model: DFlashDraftModel, fallback: int) -> int:
    """Choose the measured best speculative block size for known DFlash drafts."""
    args = getattr(draft_model, "args", None)
    hidden_size = int(getattr(args, "hidden_size", 0) or 0)
    target_layers = int(getattr(args, "num_target_layers", 0) or 0)
    draft_layers = int(getattr(args, "num_hidden_layers", 0) or 0)

    # Qwen3.6-27B dense DFlash: 32 was the best measured code-like prompt
    # setting. Longer 48/64 blocks only won on synthetic/repetitive text and
    # regressed mixed prompts because failed long drafts add useless work.
    if hidden_size >= 4096 and target_layers >= 60:
        return 32

    # Qwen3.6-35B-A3B MoE DFlash: the drafter's native block size (16 today)
    # was fastest in local DDTree measurements; 32/48/64 reduced tok/s.
    if hidden_size <= 3072 and target_layers == 40 and draft_layers >= 8:
        return int(fallback)

    return int(fallback)


def _commit_ddtree_target_cache(
    target_cache: list[Any],
    *,
    start: int,
    target_len: int,
    accepted_indices: list[int],
    inv_dfs_order: Any,
    tree_cache_state: dict[str, Any],
) -> int:
    """Commit only the DDTree accepted path to target caches.

    Tree verification appends FA KV in DFS order and computes GDN states for
    every tree node without mutating the recurrent cache.  To preserve exact
    vanilla-DFlash semantics, gather the accepted path into FA KV caches and set
    each GDN cache to the state of the final accepted node.
    """
    commit_start_ns = time.perf_counter_ns()
    accepted_dfs_positions = [int(inv_dfs_order[int(idx)]) for idx in accepted_indices]
    accepted_dfs = mx.array(accepted_dfs_positions, dtype=mx.int32)
    source_positions = int(start) + accepted_dfs
    final_tree_idx = int(accepted_indices[-1])
    gdn_layers = tree_cache_state.get("gdn_layers", {}) if tree_cache_state else {}
    gdn_recompute_layers = tree_cache_state.get("gdn_recompute_layers", {}) if tree_cache_state else {}
    accepted_array = mx.array([int(idx) for idx in accepted_indices], dtype=mx.int32)

    for layer_idx, cache_entry in enumerate(target_cache):
        if layer_idx in gdn_recompute_layers:
            layer_state = gdn_recompute_layers[layer_idx]
            linear_attn = layer_state["linear_attn"]
            path_inputs = mx.take(layer_state["inputs"], accepted_array, axis=1)
            try:
                from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

                scratch = RecurrentRollbackCache(
                    size=2,
                    conv_kernel_size=int(getattr(cache_entry, "conv_kernel_size", 4)),
                )
                scratch[0] = cache_entry[0]
                scratch[1] = cache_entry[1]
                linear_attn(path_inputs, None, scratch)
                cache_entry[0] = mx.contiguous(scratch[0]) if scratch[0] is not None else None
                cache_entry[1] = mx.contiguous(scratch[1]) if scratch[1] is not None else None
            finally:
                _clear_cache_transients(cache_entry)
            continue

        if layer_idx in gdn_layers:
            layer_state = gdn_layers[layer_idx]
            conv_states = layer_state["conv_states"]
            recurrent_states = layer_state["states"]
            if isinstance(conv_states, dict) and "all" in conv_states:
                conv_state = mx.take(conv_states["all"], mx.array([final_tree_idx]), axis=0)
            else:
                conv_state = conv_states[final_tree_idx]
            if isinstance(recurrent_states, dict) and "all" in recurrent_states:
                recurrent_state = mx.take(recurrent_states["all"], mx.array([final_tree_idx]), axis=0)
            else:
                recurrent_state = recurrent_states[final_tree_idx]
            cache_entry[0] = mx.contiguous(conv_state)
            cache_entry[1] = mx.contiguous(recurrent_state)
            _clear_cache_transients(cache_entry)
            continue

        keys = getattr(cache_entry, "keys", None)
        values = getattr(cache_entry, "values", None)
        if keys is not None and values is not None and hasattr(cache_entry, "offset"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > int(start):
                if hasattr(cache_entry, "max_size") and hasattr(cache_entry, "_idx"):
                    # RotatingKVCache stores only a physical suffix of the
                    # logical context.  After tree verification its tensor is:
                    #   [physical prefix in temporal order][DFS tree nodes]
                    # so commit by compacting the physical tree suffix, while
                    # preserving the logical offset for RoPE positions.
                    tree_size = max(0, int(offset) - int(start))
                    physical_len = int(keys.shape[2])
                    physical_prefix = max(0, physical_len - tree_size)
                    physical_sources = physical_prefix + accepted_dfs
                    prefix_keys = keys[..., :physical_prefix, :]
                    prefix_values = values[..., :physical_prefix, :]
                    selected_keys = mx.take(keys, physical_sources, axis=2)
                    selected_values = mx.take(values, physical_sources, axis=2)
                    cache_entry.keys = mx.concatenate([prefix_keys, selected_keys], axis=2)
                    cache_entry.values = mx.concatenate([prefix_values, selected_values], axis=2)
                    cache_entry.offset = int(target_len)
                    cache_entry._idx = int(cache_entry.keys.shape[2])
                else:
                    selected_keys = mx.take(keys, source_positions, axis=2)
                    selected_values = mx.take(values, source_positions, axis=2)
                    cache_entry.keys[..., int(start):int(target_len), :] = selected_keys
                    cache_entry.values[..., int(start):int(target_len), :] = selected_values
                    cache_entry.offset = int(target_len)
            continue

        if hasattr(cache_entry, "trim"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > int(target_len):
                cache_entry.trim(offset - int(target_len))
        elif hasattr(cache_entry, "offset"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > int(target_len):
                cache_entry.offset = int(target_len)
        elif hasattr(cache_entry, "crop"):
            cache_entry.crop(int(target_len))

    return time.perf_counter_ns() - commit_start_ns


def stream_dflash_generate_impl(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    prefix_snapshot: Optional[DFlashPrefixSnapshot] = None,
    stable_prefix_len: Optional[int] = None,
    prefix_cache: Optional[Any] = None,
    runtime_context: Any,
) -> Iterator[dict[str, Any]]:
    from dflash_mlx.runtime import (
        _eval_logits_and_captured,
        _ns_to_us,
        _prepare_prompt_tokens,
        apply_repetition_penalty,
        build_suppress_token_mask,
        greedy_tokens_with_mask,
    )
    target_ops = resolve_target_ops(target_model)
    bind_draft_to_target(draft_model, target_model, target_ops=target_ops)

    target_capabilities = target_ops.capabilities_for(target_model)
    supports_prefix_snapshot = bool(
        getattr(target_capabilities, "supports_prefix_snapshot", True)
    )
    if quantize_kv_cache:
        target_ops.configure_full_attention_split(target_model, enabled=False)

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    fallback_reason: Optional[str] = None

    prompt_len = len(prompt_tokens)
    if runtime_context is None:
        raise ValueError("runtime_context is required")
    runtime_config = runtime_context.runtime
    configured_max_ctx = int(runtime_config.dflash_max_ctx)
    dflash_max_ctx = configured_max_ctx if configured_max_ctx > 0 else sys.maxsize
    target_fa_window = int(runtime_config.target_fa_window)
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        yield from stream_baseline_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            fallback_reason=fallback_reason,
        )
        return
    draft_sink_size, draft_window_size = resolve_draft_window(
        runtime_config,
        draft_model,
        context_len=prompt_len + max(0, int(max_new_tokens)),
    )
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    draft_backend = make_draft_backend()

    snap_prefix_len = _validate_prefix_snapshot(prefix_snapshot, prompt_tokens)
    if not supports_prefix_snapshot:
        snap_prefix_len = 0
    if snap_prefix_len > 0 and (quantize_kv_cache or target_fa_window > 0):
        snap_prefix_len = 0
    if snap_prefix_len > 0:
        template_cache = target_ops.make_cache(
            target_model,
            enable_speculative_linear_cache=True,
            quantize_kv_cache=quantize_kv_cache,
            target_fa_window=target_fa_window,
        )
        try:
            assert prefix_snapshot is not None
            target_cache = hydrate_target_cache(prefix_snapshot, template_cache)
        except (ValueError, TypeError):
            snap_prefix_len = 0
            target_cache = target_ops.make_cache(
                target_model,
                enable_speculative_linear_cache=True,
                quantize_kv_cache=quantize_kv_cache,
                target_fa_window=target_fa_window,
            )
        finally:
            del template_cache
    else:
        target_cache = target_ops.make_cache(
            target_model,
            enable_speculative_linear_cache=True,
            quantize_kv_cache=quantize_kv_cache,
            target_fa_window=target_fa_window,
        )
    draft_cache = draft_backend.make_cache(
        draft_model=draft_model,
        sink_size=draft_sink_size,
        window_size=draft_window_size,
    )
    target_layer_id_list = list(draft_model.target_layer_ids)
    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}
    diagnostics = runtime_context.diagnostics
    profile_cycles = _profile_dflash_cycles_enabled(diagnostics)
    memory_waterfall = _memory_waterfall_enabled(diagnostics)
    clear_cache_boundaries = bool(runtime_config.clear_cache_boundaries)

    def _clear_cache_boundary() -> None:
        if clear_cache_boundaries and hasattr(mx, "clear_cache"):
            mx.clear_cache()

    def _waterfall_event(
        phase: str,
        *,
        target_hidden_value: Any = None,
        gen_hidden_chunks_value: Any = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        if not memory_waterfall:
            return None
        return {
            "event": "memory_waterfall",
            **_collect_memory_waterfall(
                phase=phase,
                target_cache=target_cache,
                draft_cache=draft_cache,
                target_hidden=target_hidden_value,
                gen_hidden_chunks=gen_hidden_chunks_value,
                prefix_cache=prefix_cache,
                extra=extra,
            ),
        }

    _yield_pause_ns = 0
    # Always track time spent outside the generator while the server handles
    # yielded events (streaming, prefix-cache snapshot insertion, diagnostics).
    # Summary elapsed_us is core runtime; request logs also report wall time.
    track_yield_pause = True

    def _yield_start() -> int:
        return time.perf_counter_ns() if track_yield_pause else 0

    def _yield_done(mark: int) -> None:
        nonlocal _yield_pause_ns
        if track_yield_pause:
            _yield_pause_ns += time.perf_counter_ns() - mark

    try:
        start_ns = time.perf_counter_ns()
        evt = _waterfall_event("after_target_cache_create")
        if evt is not None:
            _pre_yield = _yield_start()
            yield evt
            _yield_done(_pre_yield)
        prefill_start_ns = time.perf_counter_ns()
        prefill_step_size = int(runtime_config.prefill_step_size)
        prefill_logits = None
        target_hidden: Optional[mx.array] = None

        _phase_rebuild_ns = 0
        _phase_cold_ns = 0
        _phase_seam_ns = 0
        _phase_tail_ns = 0

        if snap_prefix_len > 0:
            assert prefix_snapshot is not None
            if profile_cycles:
                _t = time.perf_counter_ns()
            target_hidden = init_target_hidden_from_snapshot(
                prefix_snapshot,
                snap_prefix_len=snap_prefix_len,
                prompt_len=prompt_len,
            )
            if profile_cycles:
                _phase_rebuild_ns += time.perf_counter_ns() - _t
            evt = _waterfall_event(
                "after_prefix_hydrate",
                target_hidden_value=target_hidden,
                extra={"snap_prefix_len": int(snap_prefix_len)},
            )
            if evt is not None:
                _pre_yield = _yield_start()
                yield evt
                _yield_done(_pre_yield)

        snapshot_boundary = compute_snapshot_boundary(prompt_len, stable_prefix_len)
        prefill_context_len = max(0, snapshot_boundary - 1)
        chunked_start = min(snap_prefix_len, prefill_context_len)
        for chunk_start in range(chunked_start, prefill_context_len, prefill_step_size):
            if profile_cycles:
                _t = time.perf_counter_ns()
            chunk_end = min(chunk_start + prefill_step_size, prefill_context_len)
            chunk_ids = prompt_array[:, chunk_start:chunk_end]
            prefill_logits, prefill_hidden_states = target_ops.forward_with_hidden_capture(
                target_model,
                input_ids=chunk_ids,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(prefill_logits, prefill_hidden_states)
            feat = target_ops.extract_context_feature(
                prefill_hidden_states,
                target_layer_id_list,
            )
            if target_hidden is None:
                target_hidden = mx.zeros(
                    (feat.shape[0], prompt_len, feat.shape[-1]),
                    dtype=feat.dtype,
                )
            target_hidden[:, chunk_start:chunk_end, :] = feat
            mx.eval(target_hidden)
            del feat, prefill_hidden_states
            if profile_cycles:
                _phase_cold_ns += time.perf_counter_ns() - _t
            _clear_cache_boundary()
            _pre_yield = _yield_start()
            yield {
                "event": "prefill_progress",
                "tokens_processed": chunk_end,
                "tokens_total": prompt_len,
            }
            _yield_done(_pre_yield)
            evt = _waterfall_event(
                "after_prefill_chunk",
                target_hidden_value=target_hidden,
                extra={
                    "chunk_start": int(chunk_start),
                    "chunk_end": int(chunk_end),
                },
            )
            if evt is not None:
                _pre_yield = _yield_start()
                yield evt
                _yield_done(_pre_yield)

            # Agentic prompts often diverge in the middle while still sharing a
            # large leading prefix. A single snapshot at the final stable
            # boundary makes prefix reuse all-or-nothing. Emit safe checkpoint
            # snapshots after completed prefill chunks so future requests can
            # hydrate the longest byte-identical prefix before a divergence.
            # This preserves correctness for hybrid FA+GDN models because the
            # checkpoint is captured from a real forward pass boundary rather
            # than by cropping recurrent state.
            if supports_prefix_snapshot and chunk_end > snap_prefix_len:
                _pre_yield = _yield_start()
                yield {
                    "event": "prefill_snapshot_ready",
                    "token_ids": list(prompt_tokens[:chunk_end]),
                    "target_cache": target_cache,
                    "target_hidden": target_hidden[:, :chunk_end, :] if target_hidden is not None else None,
                    "last_logits": prefill_logits[:, -1, :] if prefill_logits is not None else None,
                    "from_snapshot": bool(snap_prefix_len > 0),
                    "snap_prefix_len": snap_prefix_len,
                    "snapshot_boundary": int(chunk_end),
                    "checkpoint": True,
                }
                _yield_done(_pre_yield)

        if (
            snap_prefix_len > 0
            and snap_prefix_len == snapshot_boundary
            and prefix_snapshot is not None
            and prefix_snapshot.last_logits is not None
        ):
            if profile_cycles:
                _t = time.perf_counter_ns()
            last_logits_2d = prefix_snapshot.last_logits
            prefill_logits = mx.expand_dims(last_logits_2d, axis=1)
            mx.eval(prefill_logits)
            if profile_cycles:
                _phase_seam_ns += time.perf_counter_ns() - _t
        elif snapshot_boundary > 0 and snap_prefix_len < snapshot_boundary:
            if profile_cycles:
                _t = time.perf_counter_ns()
            final_prompt_start = snapshot_boundary - 1
            prefill_logits, prefill_hidden_states = target_ops.forward_with_hidden_capture(
                target_model,
                input_ids=prompt_array[:, final_prompt_start:snapshot_boundary],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(prefill_logits, prefill_hidden_states)
            feat = target_ops.extract_context_feature(
                prefill_hidden_states,
                target_layer_id_list,
            )
            if target_hidden is None:
                target_hidden = mx.zeros(
                    (feat.shape[0], prompt_len, feat.shape[-1]),
                    dtype=feat.dtype,
                )
            target_hidden[:, final_prompt_start:snapshot_boundary, :] = feat
            mx.eval(target_hidden)
            del feat, prefill_hidden_states
            if profile_cycles:
                _phase_seam_ns += time.perf_counter_ns() - _t
        _pre_yield = _yield_start()
        yield {
            "event": "prefill_progress",
            "tokens_processed": snapshot_boundary,
            "tokens_total": prompt_len,
        }
        _yield_done(_pre_yield)
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()

        if supports_prefix_snapshot:
            _pre_yield = _yield_start()
            yield {
                "event": "prefill_snapshot_ready",
                "token_ids": list(prompt_tokens[:snapshot_boundary]),
                "target_cache": target_cache,
                "target_hidden": target_hidden[:, :snapshot_boundary, :] if target_hidden is not None else None,
                "last_logits": prefill_logits[:, -1, :] if prefill_logits is not None else None,
                "from_snapshot": bool(snap_prefix_len > 0),
                "snap_prefix_len": snap_prefix_len,
                "snapshot_boundary": snapshot_boundary,
            }
            _yield_done(_pre_yield)
            evt = _waterfall_event(
                "after_prefill_snapshot_ready",
                target_hidden_value=target_hidden,
                extra={"snapshot_boundary": int(snapshot_boundary)},
            )
            if evt is not None:
                _pre_yield = _yield_start()
                yield evt
                _yield_done(_pre_yield)

        if snapshot_boundary < prompt_len:
            if profile_cycles:
                _t = time.perf_counter_ns()
            tail_logits, tail_hidden_states = target_ops.forward_with_hidden_capture(
                target_model,
                input_ids=prompt_array[:, snapshot_boundary:prompt_len],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(tail_logits, tail_hidden_states)
            tail_feat = target_ops.extract_context_feature(
                tail_hidden_states,
                target_layer_id_list,
            )
            if target_hidden is None:
                target_hidden = mx.zeros(
                    (tail_feat.shape[0], prompt_len, tail_feat.shape[-1]),
                    dtype=tail_feat.dtype,
                )
            target_hidden[:, snapshot_boundary:prompt_len, :] = tail_feat
            mx.eval(target_hidden)
            prefill_logits = tail_logits
            del tail_feat, tail_hidden_states
            if profile_cycles:
                _phase_tail_ns += time.perf_counter_ns() - _t
            _clear_cache_boundary()
            _pre_yield = _yield_start()
            yield {
                "event": "prefill_progress",
                "tokens_processed": prompt_len,
                "tokens_total": prompt_len,
            }
            _yield_done(_pre_yield)
        evt = _waterfall_event(
            "after_tail_prefill",
            target_hidden_value=target_hidden,
            extra={"prompt_len": int(prompt_len)},
        )
        if evt is not None:
            _pre_yield = _yield_start()
            yield evt
            _yield_done(_pre_yield)

        prefill_ns = time.perf_counter_ns() - prefill_start_ns

        prefill_target_hidden_for_snapshot = (
            target_hidden if supports_prefix_snapshot else None
        )
        gen_hidden_chunks: list[mx.array] = []
        last_cycle_logits: Optional[mx.array] = None

        suppress_token_mask = build_suppress_token_mask(int(prefill_logits.shape[-1]), suppress_token_ids)
        rep_penalty = float(getattr(runtime_config, "repetition_penalty", 1.0) or 1.0)
        _sampled_ids: list[int] = list(prompt_tokens)  # seed with full prompt for conversation-level penalty
        staged_first = greedy_tokens_with_mask(
            apply_repetition_penalty(prefill_logits[:, -1:, :], _sampled_ids, rep_penalty).squeeze(0),
            suppress_token_mask,
        ).reshape(-1)
        prefill_tokens_restored = max(0, min(int(snap_prefix_len), int(prompt_len)))
        prefill_tokens_computed = max(0, int(prompt_len) - prefill_tokens_restored)

        prefill_event = {
            "event": "prefill",
            "prefill_us": prefill_ns / 1_000.0,
            "prompt_token_count": prompt_len,
            "snap_prefix_len": int(snap_prefix_len),
            "snapshot_boundary": int(snapshot_boundary),
            "logical_ctx_tokens": int(prompt_len),
            "physical_prefill_tokens": int(prefill_tokens_computed),
            "prefill_tokens_restored": int(prefill_tokens_restored),
            "prefill_tokens_computed": int(prefill_tokens_computed),
        }
        if profile_cycles:
            prefill_event.update(
                {
                    "phase_rebuild_us": _phase_rebuild_ns / 1_000.0,
                    "phase_cold_us": _phase_cold_ns / 1_000.0,
                    "phase_seam_us": _phase_seam_ns / 1_000.0,
                    "phase_tail_us": _phase_tail_ns / 1_000.0,
                }
            )
        _pre_yield = _yield_start()
        yield prefill_event
        _yield_done(_pre_yield)

        # Decode/read-speed accounting starts once prefill is complete and the
        # prefill event has been handed to the caller.  Keep both wall time and
        # core time (wall minus generator yield/backpressure) so request logs
        # can separate model decode from SSE/client pacing.
        decode_start_ns = time.perf_counter_ns()
        decode_yield_pause_start_ns = _yield_pause_ns

        first_token_yielded = False
        if max_new_tokens > 0:
            first_token_yielded = True
            _pre_yield = _yield_start()
            yield {
                "event": "token",
                "token_id": int(staged_first.item()),
                "generated_tokens": 1,
                "acceptance_ratio": 0.0,
                "cycles_completed": 0,
            }
            _yield_done(_pre_yield)

        draft_block_size = int(draft_model.block_size)
        configured_block_tokens = int(getattr(runtime_config, "draft_block_tokens", 0) or 0)
        if block_tokens is not None:
            if isinstance(block_tokens, str) and block_tokens.strip().lower() == "auto":
                requested_block_tokens = _auto_draft_block_tokens(draft_model, draft_block_size)
            else:
                requested_block_tokens = int(block_tokens)
        elif configured_block_tokens < 0:
            requested_block_tokens = _auto_draft_block_tokens(draft_model, draft_block_size)
        else:
            requested_block_tokens = configured_block_tokens if configured_block_tokens > 0 else draft_block_size
        # DFlash draft models are trained/configured with a nominal block size,
        # but the architecture is fully causal and can run longer speculative
        # blocks.  Keep the default at the model's block_size, while honoring an
        # explicit block_tokens override for controlled throughput experiments.
        # Target verification remains exact, so a too-long draft can only lower
        # acceptance/performance, not corrupt output.
        effective_block_tokens = max(1, requested_block_tokens)
        block_token_buffer = mx.full(
            (effective_block_tokens,),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        mask_token_tail = mx.full(
            (max(0, effective_block_tokens - 1),),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        )
        generated_token_ids: list[int] = []
        accepted_from_draft = 0
        cycles_completed = 0
        verify_len_cap = resolve_verify_len_cap(runtime_config, effective_block_tokens)
        start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        seen_draft_cycle = False
        acceptance_history: list[int] = []
        cycle_profiles: list[dict[str, Any]] = []
        generation_snapshot_ns = 0
        ddtree_debug = bool(profile_cycles)
        generation_snapshot_enabled = bool(getattr(runtime_config, "generation_snapshot", True))
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
        }
        ddtree_mode = str(getattr(runtime_config, "speculative_mode", "dflash")).lower() == "ddtree"
        ddtree_budget = int(getattr(runtime_config, "ddtree_budget", 64) or 64)
        ddtree_topk = int(getattr(runtime_config, "ddtree_topk", 64) or 64)
        ddtree_profile_totals_ns = {
            "tree_build": 0,
            "compile": 0,
            "verify_setup": 0,
            "verify_fa_layers": 0,
            "verify_gdn_layers": 0,
            "verify_final_norm": 0,
            "verify_lm_head": 0,
            "posterior_argmax": 0,
            "tree_walk": 0,
            "accepted_gather": 0,
            "cache_commit": 0,
            "next_draft_launch": 0,
        }
        # Low-overhead DDTree timing counters are always collected.  Unlike
        # diagnostics/full they do not add per-layer synchronization; they use
        # sync points already required by DDTree (top-k CPU transfer, target
        # verify, cache commit) so production request logs can explain where
        # the missing speedup went.
        ddtree_timing_totals_ns = {
            "draft_launch": 0,
            "tree_build": 0,
            "topk_cast": 0,
            "topk_sync": 0,
            "topk_transfer": 0,
            "heap_build": 0,
            "compile": 0,
            "target_verify": 0,
            "posterior_argmax": 0,
            "tree_walk": 0,
            "accepted_gather": 0,
            "cache_commit": 0,
            "next_draft_launch": 0,
            "cycle_total": 0,
        }
        ddtree_cycle_count = 0
        ddtree_tree_size_total = 0
        ddtree_node_count_total = 0
        ddtree_tree_size_max = 0
        ddtree_acceptance_len_total = 0
        ddtree_commit_count_total = 0
        ddtree_prefetch_hits = 0
        ddtree_prefetch_misses = 0
        draft_prefetch_hits = 0
        draft_prefetch_misses = 0
        token_loop_end_ns = 0
        token_loop_yield_pause_end_ns = 0
        cycle_total_ns_total = 0
        prefetched_draft: Optional[dict[str, Any]] = None

        while len(generated_token_ids) < max_new_tokens:
            _ddtree_cycle = False
            cycle_start_ns = time.perf_counter_ns()
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            commit_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            ddtree_tree_build_ns = 0
            ddtree_bonus_token_id: int | None = None
            ddtree_committed_ids: list[int] | None = None
            ddtree_compile_ns = 0
            ddtree_tree_profile: dict[str, Any] = {}
            ddtree_verify_profile: dict[str, Any] = {}
            ddtree_posterior_ns = 0
            ddtree_walk_ns = 0
            ddtree_gather_ns = 0
            ddtree_cache_commit_ns = 0
            ddtree_next_draft_launch_ns = 0
            remaining = max_new_tokens - len(generated_token_ids)
            block_len = max(1, min(effective_block_tokens, remaining))
            block_token_buffer[:block_len] = int(draft_model.mask_token_id)
            block_token_buffer[:1] = staged_first
            block_token_ids = block_token_buffer[:block_len]
            current_staged_first = staged_first
            drafted = None

            draft_tree_top_ids: Any = None
            draft_tree_top_log_probs: Any = None
            if block_len > 1:
                if ddtree_mode:
                    # DDTree mode: compute greedy draft tokens plus compact
                    # top-k distributions in one draft forward pass. Full-vocab
                    # logits stay scoped inside the backend.  In production we
                    # pre-launch the next cycle's draft_topk after commit, then
                    # consume it here so draft work overlaps token yielding.
                    if (
                        prefetched_draft is not None
                        and prefetched_draft.get("mode") == "ddtree"
                        and int(prefetched_draft["block_len"]) == block_len
                        and int(prefetched_draft.get("topk", 0)) == min(ddtree_topk, ddtree_budget)
                    ):
                        ddtree_prefetch_hits += 1
                        drafted = prefetched_draft["drafted"]
                        draft_tree_top_ids = prefetched_draft["top_ids"]
                        draft_tree_top_log_probs = prefetched_draft["top_log_probs"]
                        current_staged_first = prefetched_draft["staged_first"]
                        prefetched_draft = None
                    else:
                        ddtree_prefetch_misses += 1
                        draft_start_ns = time.perf_counter_ns()
                        drafted, draft_tree_top_ids, draft_tree_top_log_probs = draft_backend.draft_topk(
                            target_model=target_model,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=current_staged_first,
                            target_hidden=target_hidden,
                            block_len=block_len,
                            mask_token_tail=mask_token_tail,
                            topk=min(ddtree_topk, ddtree_budget),
                            suppress_token_mask=suppress_token_mask,
                        )
                        if profile_cycles:
                            mx.eval(drafted, draft_tree_top_ids, draft_tree_top_log_probs)
                        else:
                            mx.async_eval(drafted, draft_tree_top_ids, draft_tree_top_log_probs)
                        draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    block_token_ids[1:block_len] = drafted
                else:
                    if (
                        prefetched_draft is not None
                        and int(prefetched_draft["block_len"]) == block_len
                    ):
                        draft_prefetch_hits += 1
                        drafted = prefetched_draft["drafted"]
                        current_staged_first = prefetched_draft["staged_first"]
                    else:
                        draft_prefetch_misses += 1
                        draft_start_ns = time.perf_counter_ns()
                        drafted = draft_backend.draft_greedy(
                            target_model=target_model,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=current_staged_first,
                            target_hidden=target_hidden,
                            block_len=block_len,
                            mask_token_tail=mask_token_tail,
                            suppress_token_mask=suppress_token_mask,
                            async_launch=True,
                            previous_token_ids=_sampled_ids,
                            repetition_penalty=rep_penalty,
                        )
                        draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                    prefetched_draft = None
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            verify_token_count = verify_token_count_for_block(block_len, verify_len_cap)

            if ddtree_mode and draft_tree_top_ids is not None and block_len > 1:
                # ── DDTree verification path ──
                from vllm_mlx.engine.ddtree import (
                    build_ddtree_tree_from_mlx_topk,
                    compile_tree,
                    follow_verified_tree as _follow_verified_tree,
                    tree_verify_forward,
                )

                _tree_build_start_ns = time.perf_counter_ns()
                ddtree_tree_profile: dict[str, Any] = {}
                tree = build_ddtree_tree_from_mlx_topk(
                    draft_tree_top_ids,
                    draft_tree_top_log_probs,
                    budget=ddtree_budget,
                    profile=ddtree_tree_profile,
                )
                mx.eval()  # sync: tree build complete, CPU data ready
                ddtree_tree_build_ns = time.perf_counter_ns() - _tree_build_start_ns

                # Compile tree
                _compile_start_ns = time.perf_counter_ns()
                root_token = int(staged_first.item())
                ct = compile_tree(tree, root_token_id=root_token, prefix_len=start)
                if profile_cycles:
                    mx.eval(ct.input_ids, ct.position_ids, ct.attention_mask, ct.dfs_order, ct.inv_dfs_order)
                ddtree_compile_ns = time.perf_counter_ns() - _compile_start_ns

                if ddtree_debug:
                    sys.stderr.write(
                        f"[ddtree] tree_size={ct.tree_size} root={root_token} "
                        f"prefix_len={start}\n"
                    )
                    sys.stderr.flush()

                # Arm rollback, verify tree forward, walk
                target_ops.arm_rollback(target_cache, prefix_len=start)
                verify_start_ns = time.perf_counter_ns()
                tree_cache_state: dict[str, Any] = {}
                ddtree_verify_profile = {} if profile_cycles else {}
                tree_logits, tree_hidden_states = tree_verify_forward(
                    target_model,
                    compiled_tree=ct,
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                    tree_cache_state=tree_cache_state,
                    profile=ddtree_verify_profile if profile_cycles else None,
                )
                if profile_cycles:
                    mx.eval(tree_logits)
                else:
                    mx.eval()  # sync: verify forward complete
                verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
                verify_ns_total += verify_cycle_ns
                if ddtree_debug:
                    sys.stderr.write(
                        f"[ddtree] verify complete: {verify_cycle_ns/1e6:.1f}ms\n"
                    )
                    sys.stderr.flush()

                # Walk tree. This includes the lazy full-tree LM head when
                # diagnostics/full is off; request logs expose that cost as
                # ddtree_timing_avg_us.posterior_argmax.
                _posterior_start_ns = time.perf_counter_ns()
                posterior_tokens = greedy_tokens_with_mask(
                    apply_repetition_penalty(tree_logits[0], _sampled_ids, rep_penalty),
                    suppress_token_mask,
                )
                posterior_list = posterior_tokens.tolist()
                ddtree_posterior_ns = time.perf_counter_ns() - _posterior_start_ns
                _walk_start_ns = time.perf_counter_ns()
                accepted_indices, bonus_token = _follow_verified_tree(
                    tree.child_maps, posterior_list
                )
                ddtree_walk_ns = time.perf_counter_ns() - _walk_start_ns
                acceptance_len = len(accepted_indices) - 1  # minus root
                acceptance_history.append(acceptance_len)

                # ── Use tree logits directly, skip re-verify ──
                # Extract accepted node logits and hidden states from tree results.
                _gather_start_ns = time.perf_counter_ns()
                accepted_array = mx.array(accepted_indices, dtype=mx.int32)
                final_accepted_array = mx.array([accepted_indices[-1]], dtype=mx.int32)
                # Only the final accepted node's logits are needed after the
                # tree walk (for generation snapshots / next-token state).
                # Hidden states still need the full accepted path for the
                # draft model context feature.
                verify_logits = mx.take(tree_logits, final_accepted_array, axis=1)
                # Reconstruct hidden states for accepted nodes
                captured_dict: dict[int, mx.array] = {}
                for lid, hid in tree_hidden_states.items():
                    captured_dict[lid] = mx.take(hid, accepted_array, axis=1)
                if profile_cycles:
                    mx.eval(verify_logits, *captured_dict.values())
                ddtree_gather_ns = time.perf_counter_ns() - _gather_start_ns

                # Commit exactly the accepted path. FA KV was appended in DFS
                # order, while GDN states were computed per tree node; gather
                # only accepted nodes so the target cache matches vanilla DFlash.
                target_len = start + 1 + acceptance_len
                _cache_commit_start_ns = time.perf_counter_ns()
                replay_cycle_ns = _commit_ddtree_target_cache(
                    target_cache,
                    start=start,
                    target_len=target_len,
                    accepted_indices=accepted_indices,
                    inv_dfs_order=ct.inv_dfs_order.tolist(),
                    tree_cache_state=tree_cache_state,
                )
                mx.eval()  # sync: cache compaction complete
                if profile_cycles:
                    replay_cycle_ns = time.perf_counter_ns() - _cache_commit_start_ns
                ddtree_cache_commit_ns = time.perf_counter_ns() - _cache_commit_start_ns
                replay_ns_total += replay_cycle_ns

                # Build accepted token sequence (for token yielding + committed_segment)
                accepted_path_ids = [int(staged_first.item())]
                for idx in accepted_indices[1:]:
                    accepted_path_ids.append(int(tree.node_token_ids[idx - 1]))
                verify_token_ids = mx.array(accepted_path_ids, dtype=mx.uint32)
                verify_hidden_states = captured_dict
                ddtree_bonus_token_id = int(bonus_token)
                ddtree_committed_ids = accepted_path_ids

                # Verbose: log accepted tokens every 50 cycles
                if ddtree_debug and (cycles_completed % 50 == 0 or cycles_completed < 5):
                    sys.stderr.write(
                        f"[ddtree] cyc={cycles_completed} acc={accepted_path_ids[:6]}"
                        f" bonus={bonus_token} nodes={tree.node_count}\n"
                    )
                    sys.stderr.flush()

                sample_memory_cycle = False

                # Flag: shared code should skip its own acceptance computation
                _ddtree_cycle = True

            else:
                _ddtree_cycle = False
                # ── Vanilla DFlash verify path ──
                if profile_cycles or block_len <= 1:
                    verify_token_ids = block_token_ids[:verify_token_count]
                elif verify_token_count <= 1:
                    verify_token_ids = current_staged_first[:1]
                else:
                    verify_token_ids = mx.concatenate(
                        [current_staged_first[:1], drafted[: verify_token_count - 1]],
                        axis=0,
                    )
                verify_ids = verify_token_ids[None]
                target_ops.arm_rollback(target_cache, prefix_len=start)
                sample_memory_cycle = memory_waterfall and _should_sample_memory_cycle(
                    cycles_completed + 1
                )
                if sample_memory_cycle:
                    evt = _waterfall_event(
                        "before_verify_cycle",
                        target_hidden_value=target_hidden,
                        gen_hidden_chunks_value=gen_hidden_chunks,
                        extra={"cycle": int(cycles_completed + 1), "start": int(start)},
                    )
                    if evt is not None:
                        _pre_yield = _yield_start()
                        yield evt
                        _yield_done(_pre_yield)
                verify_start_ns = time.perf_counter_ns()
                verify_logits, verify_hidden_states = target_ops.verify_block(
                    target_model=target_model,
                    verify_ids=verify_ids,
                    target_cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                )
                if profile_cycles:
                    _eval_logits_and_captured(verify_logits, verify_hidden_states)
                verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
                verify_ns_total += verify_cycle_ns
                if sample_memory_cycle:
                    evt = _waterfall_event(
                        "after_verify_cycle",
                        target_hidden_value=target_hidden,
                        gen_hidden_chunks_value=gen_hidden_chunks,
                        extra={"cycle": int(cycles_completed + 1), "start": int(start)},
                    )
                    if evt is not None:
                        _pre_yield = _yield_start()
                        yield evt
                        _yield_done(_pre_yield)

            if not _ddtree_cycle:
                acceptance_start_ns = time.perf_counter_ns() if profile_cycles else 0
                posterior = greedy_tokens_with_mask(
                    apply_repetition_penalty(verify_logits[0], _sampled_ids, rep_penalty),
                    suppress_token_mask,
                )
                if not profile_cycles:
                    mx.async_eval(posterior)
                acceptance_len = int(
                    _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
                )
                acceptance_history.append(acceptance_len)
                if profile_cycles:
                    acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            else:
                # DDTree already computed the verified posterior for all tree
                # nodes during tree walk.  Do not run a second full-vocab
                # argmax over the accepted path; the unmatched posterior token
                # from the walk is the next staged token.
                posterior = None
                acceptance_cycle_ns = 0
            hidden_extract_start_ns = time.perf_counter_ns() if profile_cycles else 0
            committed_hidden = target_ops.extract_context_feature(
                verify_hidden_states,
                target_layer_id_list,
            )[:, : (1 + acceptance_len), :]
            if profile_cycles:
                if posterior is None:
                    mx.eval(committed_hidden)
                else:
                    mx.eval(committed_hidden, posterior)
            else:
                mx.async_eval(committed_hidden)
            if profile_cycles:
                hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            commit_start_ns = time.perf_counter_ns()
            start += commit_count
            target_hidden = committed_hidden
            if supports_prefix_snapshot:
                gen_hidden_chunks.append(committed_hidden)
            if _ddtree_cycle:
                last_cycle_logits = verify_logits[:, -1, :]
            else:
                last_cycle_logits = verify_logits[:, acceptance_len, :]
            if not _ddtree_cycle:
                replay_cycle_ns = target_ops.restore_after_acceptance(
                    target_cache,
                    target_len=start,
                    acceptance_length=acceptance_len,
                    drafted_tokens=max(0, verify_token_count - 1),
                )
            else:
                replay_cycle_ns = 0  # already restored in DDTree path
            if sample_memory_cycle:
                evt = _waterfall_event(
                    "after_rollback",
                    target_hidden_value=target_hidden,
                    gen_hidden_chunks_value=gen_hidden_chunks,
                    extra={
                        "cycle": int(cycles_completed + 1),
                        "start": int(start),
                        "commit_count": int(commit_count),
                    },
                )
                if evt is not None:
                    _pre_yield = _yield_start()
                    yield evt
                    _yield_done(_pre_yield)
            replay_ns_total += replay_cycle_ns
            cycles_completed += 1
            commit_wall_ns = time.perf_counter_ns() - commit_start_ns
            commit_ns_total += commit_wall_ns
            commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

            accepted_from_draft += acceptance_len
            if _ddtree_cycle:
                if ddtree_bonus_token_id is None:
                    raise RuntimeError("DDTree cycle missing bonus token")
                staged_first_next = mx.array([ddtree_bonus_token_id], dtype=mx.uint32)
            else:
                staged_first_next = posterior[acceptance_len : acceptance_len + 1]
            if not profile_cycles and not ddtree_mode:
                next_remaining = max_new_tokens - len(generated_token_ids) - commit_count
                next_block_len = max(1, min(effective_block_tokens, next_remaining))
                if next_remaining > 0 and next_block_len > 1:
                    draft_start_ns = time.perf_counter_ns()
                    next_drafted = draft_backend.draft_greedy(
                        target_model=target_model,
                        draft_model=draft_model,
                        draft_cache=draft_cache,
                        staged_first=staged_first_next,
                        target_hidden=committed_hidden,
                        block_len=next_block_len,
                        mask_token_tail=mask_token_tail,
                        suppress_token_mask=suppress_token_mask,
                        async_launch=True,
                        previous_token_ids=_sampled_ids,
                        repetition_penalty=rep_penalty,
                    )
                    launch_ns = time.perf_counter_ns() - draft_start_ns
                    draft_ns_total += launch_ns
                    draft_incremental_ns += launch_ns
                    prefetched_draft = {
                        "block_len": next_block_len,
                        "staged_first": staged_first_next,
                        "drafted": next_drafted,
                    }
                else:
                    prefetched_draft = None
            elif ddtree_mode:
                if not profile_cycles:
                    next_remaining = max_new_tokens - len(generated_token_ids) - commit_count
                    next_block_len = max(1, min(effective_block_tokens, next_remaining))
                    if next_remaining > 0 and next_block_len > 1:
                        draft_start_ns = time.perf_counter_ns()
                        next_topk = min(ddtree_topk, ddtree_budget)
                        next_drafted, next_top_ids, next_top_log_probs = draft_backend.draft_topk(
                            target_model=target_model,
                            draft_model=draft_model,
                            draft_cache=draft_cache,
                            staged_first=staged_first_next,
                            target_hidden=committed_hidden,
                            block_len=next_block_len,
                            mask_token_tail=mask_token_tail,
                            topk=next_topk,
                            suppress_token_mask=suppress_token_mask,
                        )
                        mx.async_eval(next_drafted, next_top_ids, next_top_log_probs)
                        launch_ns = time.perf_counter_ns() - draft_start_ns
                        ddtree_next_draft_launch_ns = launch_ns
                        draft_ns_total += launch_ns
                        draft_incremental_ns += launch_ns
                        prefetched_draft = {
                            "mode": "ddtree",
                            "block_len": next_block_len,
                            "topk": next_topk,
                            "staged_first": staged_first_next,
                            "drafted": next_drafted,
                            "top_ids": next_top_ids,
                            "top_log_probs": next_top_log_probs,
                        }
                    else:
                        prefetched_draft = None
                else:
                    prefetched_draft = None
            if _ddtree_cycle and ddtree_committed_ids is not None:
                committed_ids = ddtree_committed_ids[:commit_count]
            else:
                committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
            for token_id in committed_ids:
                if len(generated_token_ids) >= max_new_tokens:
                    break
                generated_token_ids.append(token_id)
                _sampled_ids.append(token_id)
                if first_token_yielded:
                    first_token_yielded = False
                    continue
                _pre_yield = _yield_start()
                yield {
                    "event": "token",
                    "token_id": token_id,
                    "generated_tokens": len(generated_token_ids),
                    "acceptance_ratio": (
                        accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
                    ),
                    "cycles_completed": cycles_completed,
                }
                _yield_done(_pre_yield)

            stop_hit = False
            if stop_token_ids:
                if _ddtree_cycle:
                    stop_hit = any(int(token_id) in stop_token_ids for token_id in committed_ids)
                elif stop_token_array is not None:
                    stop_hit = bool(
                        mx.any(
                            mx.equal(
                                committed_segment[:, None],
                                stop_token_array[None, :],
                            )
                        ).item()
                    )
            staged_first = staged_first_next

            cycle_total_ns = time.perf_counter_ns() - cycle_start_ns
            cycle_total_ns_total += cycle_total_ns
            if _ddtree_cycle:
                ddtree_cycle_count += 1
                _tree_size = int(getattr(tree, "tree_size", 0))
                _node_count = int(getattr(tree, "node_count", 0))
                ddtree_tree_size_total += _tree_size
                ddtree_node_count_total += _node_count
                ddtree_tree_size_max = max(ddtree_tree_size_max, _tree_size)
                ddtree_acceptance_len_total += int(acceptance_len)
                ddtree_commit_count_total += int(commit_count)
                ddtree_timing_totals_ns["draft_launch"] += draft_cycle_ns
                ddtree_timing_totals_ns["tree_build"] += ddtree_tree_build_ns
                ddtree_timing_totals_ns["topk_cast"] += int(ddtree_tree_profile.get("topk_cast_ns", 0))
                ddtree_timing_totals_ns["topk_sync"] += int(ddtree_tree_profile.get("topk_sync_ns", 0))
                ddtree_timing_totals_ns["topk_transfer"] += int(ddtree_tree_profile.get("topk_transfer_ns", 0))
                ddtree_timing_totals_ns["heap_build"] += int(ddtree_tree_profile.get("heap_build_ns", 0))
                ddtree_timing_totals_ns["compile"] += ddtree_compile_ns
                ddtree_timing_totals_ns["target_verify"] += verify_cycle_ns
                ddtree_timing_totals_ns["posterior_argmax"] += ddtree_posterior_ns
                ddtree_timing_totals_ns["tree_walk"] += ddtree_walk_ns
                ddtree_timing_totals_ns["accepted_gather"] += ddtree_gather_ns
                ddtree_timing_totals_ns["cache_commit"] += ddtree_cache_commit_ns
                ddtree_timing_totals_ns["next_draft_launch"] += ddtree_next_draft_launch_ns
                ddtree_timing_totals_ns["cycle_total"] += cycle_total_ns

            if profile_cycles:
                ddtree_named_ns = (
                    ddtree_tree_build_ns
                    + ddtree_compile_ns
                    + ddtree_posterior_ns
                    + ddtree_walk_ns
                    + ddtree_gather_ns
                ) if _ddtree_cycle else 0
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + replay_cycle_ns
                    + ddtree_named_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profile_entry = {
                    "cycle": cycles_completed,
                    "block_len": int(block_len),
                    "commit_count": int(commit_count),
                    "acceptance_len": int(acceptance_len),
                    "draft_us": _ns_to_us(draft_cycle_ns),
                    "verify_us": _ns_to_us(verify_cycle_ns),
                    "acceptance_us": _ns_to_us(acceptance_cycle_ns),
                    "hidden_extraction_us": _ns_to_us(hidden_extract_cycle_ns),
                    "rollback_us": _ns_to_us(replay_cycle_ns),
                    "other_us": _ns_to_us(other_cycle_ns),
                    "cycle_total_us": _ns_to_us(cycle_total_ns),
                }
                if _ddtree_cycle:
                    slow_layers = sorted(
                        list(ddtree_verify_profile.get("layers", [])),
                        key=lambda item: float(item.get("us", 0.0)),
                        reverse=True,
                    )[:6]
                    cycle_profile_entry.update(
                        {
                            "ddtree_tree_size": int(getattr(tree, "tree_size", 0)),
                            "ddtree_node_count": int(getattr(tree, "node_count", 0)),
                            "ddtree_tree_build_us": _ns_to_us(ddtree_tree_build_ns),
                            "ddtree_topk_cast_us": _ns_to_us(int(ddtree_tree_profile.get("topk_cast_ns", 0))),
                            "ddtree_topk_sync_us": _ns_to_us(int(ddtree_tree_profile.get("topk_sync_ns", 0))),
                            "ddtree_topk_transfer_us": _ns_to_us(int(ddtree_tree_profile.get("topk_transfer_ns", 0))),
                            "ddtree_heap_build_us": _ns_to_us(int(ddtree_tree_profile.get("heap_build_ns", 0))),
                            "ddtree_compile_us": _ns_to_us(ddtree_compile_ns),
                            "ddtree_verify_setup_us": _ns_to_us(int(ddtree_verify_profile.get("setup_ns", 0))),
                            "ddtree_verify_fa_layers_us": _ns_to_us(int(ddtree_verify_profile.get("fa_layers_ns", 0))),
                            "ddtree_verify_gdn_layers_us": _ns_to_us(int(ddtree_verify_profile.get("gdn_layers_ns", 0))),
                            "ddtree_verify_final_norm_us": _ns_to_us(int(ddtree_verify_profile.get("final_norm_ns", 0))),
                            "ddtree_verify_lm_head_us": _ns_to_us(int(ddtree_verify_profile.get("lm_head_ns", 0))),
                            "ddtree_posterior_argmax_us": _ns_to_us(ddtree_posterior_ns),
                            "ddtree_tree_walk_us": _ns_to_us(ddtree_walk_ns),
                            "ddtree_accepted_gather_us": _ns_to_us(ddtree_gather_ns),
                            "ddtree_cache_commit_us": _ns_to_us(ddtree_cache_commit_ns),
                            "ddtree_next_draft_launch_us": _ns_to_us(ddtree_next_draft_launch_ns),
                            "ddtree_slowest_layers": slow_layers,
                        }
                    )
                cycle_profiles.append(cycle_profile_entry)
                _pre_yield = _yield_start()
                yield {"event": "cycle_complete", **cycle_profile_entry}
                _yield_done(_pre_yield)
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns
                if _ddtree_cycle:
                    ddtree_profile_totals_ns["tree_build"] += ddtree_tree_build_ns
                    ddtree_profile_totals_ns["compile"] += ddtree_compile_ns
                    ddtree_profile_totals_ns["verify_setup"] += int(ddtree_verify_profile.get("setup_ns", 0))
                    ddtree_profile_totals_ns["verify_fa_layers"] += int(ddtree_verify_profile.get("fa_layers_ns", 0))
                    ddtree_profile_totals_ns["verify_gdn_layers"] += int(ddtree_verify_profile.get("gdn_layers_ns", 0))
                    ddtree_profile_totals_ns["verify_final_norm"] += int(ddtree_verify_profile.get("final_norm_ns", 0))
                    ddtree_profile_totals_ns["verify_lm_head"] += int(ddtree_verify_profile.get("lm_head_ns", 0))
                    ddtree_profile_totals_ns["posterior_argmax"] += ddtree_posterior_ns
                    ddtree_profile_totals_ns["tree_walk"] += ddtree_walk_ns
                    ddtree_profile_totals_ns["accepted_gather"] += ddtree_gather_ns
                    ddtree_profile_totals_ns["cache_commit"] += ddtree_cache_commit_ns
                    ddtree_profile_totals_ns["next_draft_launch"] += ddtree_next_draft_launch_ns

            if stop_hit:
                break

        token_loop_end_ns = time.perf_counter_ns()
        token_loop_yield_pause_end_ns = _yield_pause_ns

        if (
            generation_snapshot_enabled
            and supports_prefix_snapshot
            and generated_token_ids
            and prefill_target_hidden_for_snapshot is not None
            and gen_hidden_chunks
        ):
            generation_snapshot_start_ns = time.perf_counter_ns()
            try:
                gen_hidden = (
                    gen_hidden_chunks[0]
                    if len(gen_hidden_chunks) == 1
                    else mx.concatenate(gen_hidden_chunks, axis=1)
                )
                # Keep this lazy; Rapid handles generation_snapshot_ready on a
                # background executor and build_snapshot() will evaluate/clamp
                # the arrays there. The shallow cache list copy survives normal
                # target_cache list cleanup while retaining the cache objects.
                end_target_hidden = mx.concatenate(
                    [prefill_target_hidden_for_snapshot, gen_hidden], axis=1
                )
                _clear_cache_boundary()
                end_total_len = prompt_len + len(generated_token_ids)
                _pre_yield = _yield_start()
                yield {
                    "event": "generation_snapshot_ready",
                    "token_ids": list(prompt_tokens) + list(generated_token_ids),
                    "target_cache": list(target_cache),
                    "target_hidden": end_target_hidden,
                    "last_logits": last_cycle_logits,
                    "snapshot_boundary": end_total_len,
                }
                _yield_done(_pre_yield)
                evt = _waterfall_event(
                    "after_generation_snapshot_build",
                    target_hidden_value=end_target_hidden,
                    gen_hidden_chunks_value=gen_hidden_chunks,
                    extra={"snapshot_boundary": int(end_total_len)},
                )
                if evt is not None:
                    _pre_yield = _yield_start()
                    yield evt
                    _yield_done(_pre_yield)
            except Exception as _gen_snap_err:
                sys.stderr.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"[dflash] generation_snapshot_ready build failed: {_gen_snap_err}\n"
                )
                sys.stderr.flush()
            finally:
                generation_snapshot_ns += time.perf_counter_ns() - generation_snapshot_start_ns

        summary_end_ns = time.perf_counter_ns()
        if decode_start_ns <= 0:
            decode_start_ns = summary_end_ns
        if token_loop_end_ns <= 0:
            token_loop_end_ns = summary_end_ns
            token_loop_yield_pause_end_ns = _yield_pause_ns
        total_wall_ns = summary_end_ns - start_ns
        elapsed_us = (total_wall_ns - _yield_pause_ns) / 1_000.0
        decode_wall_ns = max(0, summary_end_ns - decode_start_ns)
        decode_yield_pause_ns = max(0, _yield_pause_ns - decode_yield_pause_start_ns)
        decode_core_ns = max(0, decode_wall_ns - decode_yield_pause_ns)
        token_loop_wall_ns = max(0, token_loop_end_ns - decode_start_ns)
        token_loop_yield_pause_ns = max(0, token_loop_yield_pause_end_ns - decode_yield_pause_start_ns)
        token_loop_core_ns = max(0, token_loop_wall_ns - token_loop_yield_pause_ns)

        def _tps(token_count: int, elapsed_ns: int) -> float:
            return (float(token_count) / (elapsed_ns / 1_000_000_000.0)) if elapsed_ns > 0 else 0.0

        first_20 = acceptance_history[:20]
        last_20 = acceptance_history[-20:]
        summary = {
            "event": "summary",
            "elapsed_us": elapsed_us,
            "prompt_token_count": prompt_len,
            "generated_token_ids": generated_token_ids,
            "generation_tokens": len(generated_token_ids),
            "accepted_from_draft": accepted_from_draft,
            "acceptance_ratio": (
                accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
            ),
            "block_tokens": effective_block_tokens,
            "cycles_completed": cycles_completed,
            "phase_timings_us": {
                "prefill": prefill_ns / 1_000.0,
                "draft": draft_ns_total / 1_000.0,
                "draft_prefill": draft_prefill_ns / 1_000.0,
                "draft_incremental": draft_incremental_ns / 1_000.0,
                "verify": verify_ns_total / 1_000.0,
                "replay": replay_ns_total / 1_000.0,
                "commit": commit_ns_total / 1_000.0,
                "generation_snapshot": generation_snapshot_ns / 1_000.0,
                "yield_pause": _yield_pause_ns / 1_000.0,
                "decode_yield_pause": decode_yield_pause_ns / 1_000.0,
                "token_loop_yield_pause": token_loop_yield_pause_ns / 1_000.0,
            },
            "decode_timings_us": {
                "post_prefill_wall": decode_wall_ns / 1_000.0,
                "post_prefill_core": decode_core_ns / 1_000.0,
                "to_last_token_wall": token_loop_wall_ns / 1_000.0,
                "to_last_token_core": token_loop_core_ns / 1_000.0,
                "generation_snapshot": generation_snapshot_ns / 1_000.0,
            },
            "post_prefill_wall_tps": _tps(len(generated_token_ids), decode_wall_ns),
            "post_prefill_core_tps": _tps(len(generated_token_ids), decode_core_ns),
            "decode_to_last_token_wall_tps": _tps(len(generated_token_ids), token_loop_wall_ns),
            "decode_to_last_token_core_tps": _tps(len(generated_token_ids), token_loop_core_ns),
            "cycle_wall_ms": (token_loop_wall_ns / cycles_completed / 1_000_000.0) if cycles_completed > 0 else 0.0,
            "cycle_core_ms": (token_loop_core_ns / cycles_completed / 1_000_000.0) if cycles_completed > 0 else 0.0,
            "cycle_measured_avg_ms": (cycle_total_ns_total / cycles_completed / 1_000_000.0) if cycles_completed > 0 else 0.0,
            "verify_len_cap": int(verify_len_cap),
            "quantize_kv_cache": bool(quantize_kv_cache),
            "target_fa_window": int(target_fa_window),
            "draft_sink_size": int(draft_sink_size),
            "draft_window_size": int(draft_window_size),
            "clear_cache_boundaries": bool(clear_cache_boundaries),
            "generation_snapshot": bool(generation_snapshot_enabled),
            "prefetch": {
                "draft_hits": int(draft_prefetch_hits),
                "draft_misses": int(draft_prefetch_misses),
                "ddtree_hits": int(ddtree_prefetch_hits),
                "ddtree_misses": int(ddtree_prefetch_misses),
            },
            "tokens_per_cycle": (len(generated_token_ids) / cycles_completed) if cycles_completed > 0 else 0.0,
            "acceptance_history": list(acceptance_history),
            "acceptance_first_20_avg": (sum(first_20) / len(first_20)) if first_20 else 0.0,
            "acceptance_last_20_avg": (sum(last_20) / len(last_20)) if last_20 else 0.0,
            "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
        }
        if ddtree_cycle_count > 0:
            summary["ddtree"] = {
                "cycles": int(ddtree_cycle_count),
                "avg_tree_size": ddtree_tree_size_total / ddtree_cycle_count,
                "avg_node_count": ddtree_node_count_total / ddtree_cycle_count,
                "max_tree_size": int(ddtree_tree_size_max),
                "avg_acceptance_len": ddtree_acceptance_len_total / ddtree_cycle_count,
                "avg_commit_count": ddtree_commit_count_total / ddtree_cycle_count,
                "prefetch_hit_rate": (
                    ddtree_prefetch_hits / max(1, ddtree_prefetch_hits + ddtree_prefetch_misses)
                ),
            }
            summary["ddtree_timing_totals_us"] = {
                key: _ns_to_us(value) for key, value in ddtree_timing_totals_ns.items()
            }
            summary["ddtree_timing_avg_us"] = {
                key: _ns_to_us(value) / ddtree_cycle_count
                for key, value in ddtree_timing_totals_ns.items()
            }
        if profile_cycles:
            summary["cycle_profile_us"] = cycle_profiles
            summary["cycle_profile_totals_us"] = {
                key: _ns_to_us(value) for key, value in profile_totals_ns.items()
            }
            summary["ddtree_profile_totals_us"] = {
                key: _ns_to_us(value) for key, value in ddtree_profile_totals_ns.items()
            }
        yield summary
    finally:
        target_ops.cleanup_generation_caches(target_cache, draft_cache)
        del draft_cache
        del target_cache
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
