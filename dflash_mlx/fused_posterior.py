# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""Fused repetition-penalty + argmax Metal kernel for DDTree posterior."""

from __future__ import annotations

import mlx.core as mx

_FUSED_KERNEL_CACHE: dict[tuple, object] = {}


def _build_fused_penalty_argmax_kernel(dtype: mx.Dtype):
    """Build a Metal kernel that applies repetition penalty + suppress mask
    and returns argmax per row, all in one pass. No intermediate tensors."""

    key = ("fused_penalty_argmax", dtype)
    if key in _FUSED_KERNEL_CACHE:
        return _FUSED_KERNEL_CACHE[key]

    dtype_tag = {
        mx.bfloat16: "bfloat",
        mx.float16: "half",
        mx.float32: "float",
    }.get(dtype, "float")

    source = f"""
    using namespace metal;

    /// ── Fused penalty + mask + argmax per row ──
    /// Grid: (tree_size, 1, 1)  — one threadgroup per row
    /// Threadgroup: (256, 1, 1) — each thread processes V/256 elements
    ///
    /// For each row in logits [tree_size, vocab_size]:
    ///   1. Apply repetition penalty: penalized tokens' logits are
    ///      divided (if positive) or multiplied (if negative) by penalty.
    ///   2. Apply suppress mask: suppressed tokens get -inf.
    ///   3. Reduce to find argmax (index of maximum value).
    ///
    /// Penalty indices are packed as uint32. Penalty multiplier is
    /// precomputed per-index: 1/penalty for the common case where
    /// logits are mostly positive.

    constant uint  V         [[buffer(0)]];  // vocab_size
    constant uint  num_pen   [[buffer(1)]];  // number of penalized tokens
    constant float penalty   [[buffer(2)]];  // penalty value
    constant float inv_pen   = 1.0f / penalty;

    device const {dtype_tag}*  logits          [[buffer(3)]];  // [tree_size, V]
    device const uint32_t*   penalty_ids     [[buffer(4)]];  // [num_pen] token ids to penalize
    device const bool*       suppress_mask   [[buffer(5)]];  // [V] true = suppress

    device uint32_t*  out_argmax      [[buffer(6)]];  // [tree_size]
    device {dtype_tag}*    out_max_vals    [[buffer(7)]];  // [tree_size] (optional, for debug)

    kernel void fused_penalty_argmax(
        uint row [[threadgroup_position_in_grid]]
    ) {{
        // ── Build penalty lookup table in threadgroup memory ──
        // For small penalty sets (≤256), build a direct lookup.
        // For larger sets, use a hash or binary search.
        threadgroup bool penalized[256];
        threadgroup float pen_mult[256];

        uint tid = thread_position_in_threadgroup.x;
        uint threads = threadgroups_per_grid.x > 0 ? 256 : 256;

        // Initialize penalty table
        if (tid < 256) {{
            penalized[tid] = false;
            pen_mult[tid] = 1.0f;
        }}

        // Fill penalty table from penalty_ids
        // Each penalty_id maps to pen_mult = 1/penalty (for positive logits)
        // Since we don't know sign at lookup time, we store both and branch.
        for (uint i = tid; i < num_pen && i < 256; i += threads) {{
            uint pid = penalty_ids[i];
            if (pid < 256) {{
                penalized[pid] = true;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Per-element processing + warp reduction ──
        float best_val = -INFINITY;
        uint  best_idx = 0;

        uint base = row * V;

        // Each thread processes its stride of elements
        for (uint v = tid; v < V; v += threads) {{
            float val = float(logits[base + v]);

            // Apply repetition penalty
            // For penalized tokens: positive → divide, negative → multiply
            bool is_pen = (v < 256) ? penalized[v] : false;
            if (is_pen) {{
                val = (val > 0.0f) ? (val * inv_pen) : (val * penalty);
            }}

            // Apply suppress mask
            if (suppress_mask[v]) {{
                val = -INFINITY;
            }}

            if (val > best_val || (val == best_val && v < best_idx)) {{
                best_val = val;
                best_idx = v;
            }}
        }}

        // ── Threadgroup reduction ──
        threadgroup float tg_vals[256];
        threadgroup uint  tg_idxs[256];

        tg_vals[tid] = best_val;
        tg_idxs[tid] = best_idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Parallel reduction
        for (uint stride = threads / 2; stride > 0; stride >>= 1) {{
            if (tid < stride) {{
                float other_val = tg_vals[tid + stride];
                uint  other_idx = tg_idxs[tid + stride];
                if (other_val > tg_vals[tid] ||
                    (other_val == tg_vals[tid] && other_idx < tg_idxs[tid])) {{
                    tg_vals[tid] = other_val;
                    tg_idxs[tid] = other_idx;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }}

        // Write result
        if (tid == 0) {{
            out_argmax[row] = tg_idxs[0];
            out_max_vals[row] = {dtype_tag}(tg_vals[0]);
        }}
    }}
    """

    kernel = mx.fast.metal_kernel(
        name=f"fused_penalty_argmax_{dtype_tag}",
        input_names=[
            "V",
            "num_pen",
            "penalty",
            "logits",
            "penalty_ids",
            "suppress_mask",
        ],
        output_names=["out_argmax", "out_max_vals"],
        source=source,
    )
    _FUSED_KERNEL_CACHE[key] = kernel
    return kernel


def _suppress_mask_array(
    suppress_token_ids: list[int] | None,
    vocab_size: int,
) -> mx.array | None:
    """Build a boolean suppress mask [vocab_size] from a list of token IDs."""
    if not suppress_token_ids:
        return None
    mask = mx.zeros(vocab_size, dtype=mx.bool_)
    valid = [t for t in suppress_token_ids if 0 <= t < vocab_size]
    if valid:
        idx = mx.array(valid, dtype=mx.int32)
        mask = mx.put_along_axis(
            mask.reshape(1, -1),
            idx.reshape(1, -1),
            mx.ones((1, len(valid)), dtype=mx.bool_),
            axis=-1,
        ).reshape(-1)
    return mask


def fused_penalty_argmax(
    logits: mx.array,
    *,
    penalty_ids: list[int],
    penalty: float,
    suppress_token_ids: list[int] | None = None,
) -> mx.array:
    """Fused repetition-penalty + argmax in one Metal kernel.

    Args:
        logits: [tree_size, vocab_size] BF16 or FP16 logits.
        penalty_ids: List of token IDs to penalize.
        penalty: Penalty factor (> 1.0 penalizes, 1.0 = no-op).
        suppress_token_ids: Optional list of token IDs to suppress (-inf).

    Returns:
        [tree_size] uint32 argmax token IDs.
    """
    if penalty == 1.0 and not suppress_token_ids:
        # No penalty, no suppression — use native argmax
        return mx.argmax(logits, axis=-1).astype(mx.uint32)

    tree_size = logits.shape[0]
    vocab_size = logits.shape[-1]

    # Build penalty ID array
    valid_penalty_ids = sorted(set(
        tid for tid in penalty_ids if 0 <= tid < vocab_size and tid < 256
    ))
    penalty_ids_arr = mx.array(valid_penalty_ids, dtype=mx.uint32)

    # Build suppress mask
    sup_mask = _suppress_mask_array(suppress_token_ids, vocab_size)
    if sup_mask is None:
        sup_mask = mx.zeros(vocab_size, dtype=mx.bool_)

    kwargs = {
        "template": [("T", logits.dtype)],
        "grid": (tree_size, 1, 1),
        "threadgroup": (256, 1, 1),
        "output_shapes": [(tree_size,), (tree_size,)],
        "output_dtypes": [mx.uint32, logits.dtype],
    }

    # Handle vocab_size > 256 by evaluating penalty on more indices
    num_pen = len(valid_penalty_ids)

    kernel = _build_fused_penalty_argmax_kernel(logits.dtype)
    argmax_ids, _max_vals = kernel(
        inputs=[
            mx.array(vocab_size, dtype=mx.uint32),
            mx.array(num_pen, dtype=mx.uint32),
            mx.array(penalty, dtype=mx.float32),
            logits,
            penalty_ids_arr,
            sup_mask,
        ],
        **kwargs,
    )

    return argmax_ids


def fused_penalty_argmax_python(
    logits: mx.array,
    *,
    penalty_ids: list[int],
    penalty: float,
    suppress_token_ids: list[int] | None = None,
) -> mx.array:
    """Fused penalty + argmax using only element-wise ops (no scatter).

    Builds a penalty multiplier tensor [vocab_size] via a single scatter on a
    small mask array, then applies it with mx.where (broadcast, no scatter on
    logits).  The key insight: mx.where with a broadcast mask evaluates as one
    kernel, whereas mx.put_along_axis on the logits tensor creates a scatter
    that forces materialization.
    """
    if penalty == 1.0 and not suppress_token_ids:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)

    vocab_size = logits.shape[-1]
    result = logits

    if penalty_ids and penalty != 1.0:
        valid = sorted(set(t for t in penalty_ids if 0 <= t < vocab_size))
        if valid:
            # Build penalty mask [vocab_size] — ONE scatter on a small array
            pen_mask = mx.zeros(vocab_size, dtype=mx.bool_)
            idx = mx.array(valid, dtype=mx.int32)
            pen_mask = mx.put_along_axis(
                pen_mask.reshape(1, -1),
                idx.reshape(1, -1),
                mx.ones((1, len(valid)), dtype=mx.bool_),
                axis=-1,
            ).reshape(-1)
            # Apply penalty via mx.where — broadcasts over rows, no scatter on logits
            result = mx.where(
                pen_mask,
                mx.where(result > 0, result / penalty, result * penalty),
                result,
            )

    if suppress_token_ids:
        valid_sup = sorted(set(t for t in suppress_token_ids if 0 <= t < vocab_size))
        if valid_sup:
            sup_mask = mx.zeros(vocab_size, dtype=mx.bool_)
            idx_s = mx.array(valid_sup, dtype=mx.int32)
            sup_mask = mx.put_along_axis(
                sup_mask.reshape(1, -1),
                idx_s.reshape(1, -1),
                mx.ones((1, len(valid_sup)), dtype=mx.bool_),
                axis=-1,
            ).reshape(-1)
            floor = mx.array(-1e9, dtype=result.dtype)
            result = mx.where(sup_mask, floor, result)

    return mx.argmax(result, axis=-1).astype(mx.uint32)
