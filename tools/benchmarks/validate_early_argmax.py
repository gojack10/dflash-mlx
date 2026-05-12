# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""Validate A2 approach: does early-layer argmax predict final-layer argmax?

Theory: if the model's token preference stabilizes early, we can run a cheap
shallow forward pass (layers 0-7), compute argmax, prune unpromising tree
nodes, then run the expensive deep pass (layers 8-63) only for survivors.

This script validates that theory by running the full forward pass once,
extracting the argmax token at every layer, and comparing each layer's
argmax to the final layer's argmax.

Usage:
    python -m tools.benchmarks.validate_early_argmax \
        --model /path/to/model \
        --draft /path/to/draft \
        --num-samples 50 \
        --draft-len 12
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from typing import Any

import mlx.core as mx
import numpy as np

from dflash_mlx.generate import load_runtime_components
from dflash_mlx.engine.target_ops import resolve_target_ops


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    print("Loading models...")
    target_model, tokenizer, draft_model, ref = load_runtime_components(
        model_ref=args.model,
        draft_ref=args.draft,
    )
    ops = resolve_target_ops(target_model)
    inner = ops.text_model(target_model)
    num_layers = len(inner.layers)
    print(f"Model: {num_layers} layers")

    # ── Get draft tokens for the prompt ──
    from dflash_mlx.draft_backend import make_draft_backend

    prompt_tokens = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(tokenizer, "apply_chat_template")
        else tokenizer.encode(args.prompt)
    )
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    prompt_len = len(prompt_tokens)

    # Prefill: run the prompt through the model to get KV cache + hidden states
    cache = ops.make_cache(
        target_model,
        enable_speculative_linear_cache=False,
    )

    # Capture all layers — draft model needs specific intermediate layers
    capture_all_prefill = set(range(1, num_layers + 1))
    print(f"Prefilling {prompt_len} tokens...")
    _, prefill_hidden = ops.forward_with_hidden_capture(
        target_model,
        input_ids=prompt_array,
        cache=cache,
        capture_layer_ids=capture_all_prefill,
    )
    mx.eval(*prefill_hidden.values())
    print("Prefill done.")

    # ── Generate draft tokens for the next few positions ──
    draft_backend = make_draft_backend()
    draft_cache = draft_backend.make_cache(
        draft_model=draft_model,
        sink_size=64,
        window_size=1024,
    )

    # Get draft greedy tokens for args.draft_len positions
    staged_first = mx.array([prompt_tokens[-1]], dtype=mx.uint32)
    target_hidden = ops.extract_context_feature(
        prefill_hidden,
        list(draft_model.target_layer_ids),
    )

    # Run draft model to get block of tokens
    draft_tokens = draft_backend.draft_greedy(
        target_model=target_model,
        draft_model=draft_model,
        draft_cache=draft_cache,
        staged_first=staged_first,
        target_hidden=target_hidden,
        block_len=args.draft_len,
        mask_token_tail=mx.full(
            (args.draft_len - 1,),
            int(draft_model.mask_token_id),
            dtype=mx.uint32,
        ),
        suppress_token_mask=None,
        async_launch=False,
    )
    mx.eval(draft_tokens)
    draft_token_list = draft_tokens.tolist()
    print(f"Draft tokens: {draft_token_list}")

    # ── Build a set of draft tokens to test against ──
    # We'll test: for each draft token, does the target's early-layer argmax
    # agree with the final-layer argmax?
    # We run the draft tokens through the target model, capturing hidden
    # states at EVERY layer, then compute LM head at each layer.

    draft_ids = mx.array(draft_token_list, dtype=mx.uint32)[None]  # (1, draft_len)
    print(f"Running target forward on {args.draft_len} tokens, capturing all {num_layers} layers...")

    # Capture at every layer (indices 1..num_layers)
    capture_all = set(range(1, num_layers + 1))
    start_ns = time.perf_counter_ns()
    logits, captured = ops.forward_with_hidden_capture(
        target_model,
        input_ids=draft_ids,
        cache=cache,
        capture_layer_ids=capture_all,
    )
    mx.eval(logits, *captured.values())
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
    print(f"Forward pass: {elapsed_ms:.1f}ms")

    # ── Compute argmax at each layer for each position ──
    # captured: {layer_idx: hidden_states} where layer_idx is 1-indexed
    # hidden_states shape: (1, draft_len, hidden_dim)

    # Final layer argmax (ground truth)
    final_logits = logits[0]  # (draft_len, vocab_size)
    final_argmax = mx.argmax(final_logits, axis=-1).tolist()

    # Compute LM head for each intermediate layer
    print(f"\n{'Layer':>6s} {'Match%':>8s} {'Kendall τ':>10s} {'Top-1 agree':>12s} {'Top-3 agree':>12s}")
    print("-" * 55)

    results: list[dict[str, Any]] = []

    for layer_idx in range(1, num_layers + 1):
        if layer_idx not in captured:
            continue
        hidden = captured[layer_idx]  # (1, draft_len, hidden_dim)
        layer_logits = ops.logits_from_hidden(target_model, hidden)
        if layer_logits is None:
            continue
        # layer_logits shape: (1, draft_len, vocab_size)
        layer_logits_2d = layer_logits[0]  # (draft_len, vocab_size)
        mx.eval(layer_logits_2d)

        layer_argmax = mx.argmax(layer_logits_2d, axis=-1).tolist()

        # Match rate: how often layer argmax == final argmax?
        matches = sum(1 for a, b in zip(layer_argmax, final_argmax) if a == b)
        match_pct = matches / len(final_argmax) * 100

        # Top-k agreement: is final argmax in layer's top-3?
        top3 = mx.topk(layer_logits_2d, k=3, axis=-1)
        mx.eval(top3)
        top3_ids = top3.tolist()  # list of lists
        top3_hits = sum(1 for i, t3 in enumerate(top3_ids) if final_argmax[i] in t3)
        top3_pct = top3_hits / len(final_argmax) * 100

        # Top-1 agreement: is layer argmax in final's top-3?
        final_top3 = mx.topk(final_logits, k=3, axis=-1)
        mx.eval(final_top3)
        final_top3_ids = final_top3.tolist()
        top1_in_final_top3 = sum(
            1 for i, la in enumerate(layer_argmax) if la in final_top3_ids[i]
        )
        top1_in_final_pct = top1_in_final_top3 / len(final_argmax) * 100

        # Kendall tau (rank correlation) - simplified: compare ordering
        # We'll just compute overlap of top-1 across all positions
        # For a proper metric, we'd need rank correlation, but match rate suffices

        results.append({
            "layer": layer_idx,
            "match_pct": match_pct,
            "top3_pct": top3_pct,
            "top1_in_final_top3_pct": top1_in_final_pct,
            "layer_argmax": layer_argmax,
            "final_argmax": final_argmax,
        })

        marker = ""
        if layer_idx <= 8:
            marker = " ◀── shallow cutoff"
        elif layer_idx <= 16:
            marker = ""
        print(
            f"{layer_idx:>6d} {match_pct:>7.1f}% {'—':>10s} "
            f"{match_pct:>7.1f}%{marker} {top3_pct:>7.1f}%"
        )

    # ── Summary: is layer 8 predictive enough? ──
    print(f"\n{'='*55}")
    print("SUMMARY FOR A2 FEASIBILITY")
    print(f"{'='*55}")
    layer8 = next((r for r in results if r["layer"] == 8), None)
    if layer8:
        print(f"Layer 8 → 64 argmax match:  {layer8['match_pct']:.1f}%")
        print(f"Layer 8 top-3 contains final: {layer8['top3_pct']:.1f}%")
        print(f"Final top-3 contains layer 8: {layer8['top1_in_final_top3_pct']:.1f}%")
        if layer8["match_pct"] >= 80:
            print("✓ A2 is VIABLE: early layers strongly predict final argmax")
        elif layer8["match_pct"] >= 60:
            print("△ A2 is MARGINAL: may work with careful threshold tuning")
        else:
            print("✗ A2 is RISKY: early layers don't predict final argmax well")

    # ── Detailed per-position breakdown for layer 8 ──
    if layer8 and args.verbose:
        print(f"\nPer-position breakdown (layer 8 vs layer {num_layers}):")
        for i in range(args.draft_len):
            l8 = layer8["layer_argmax"][i]
            f64 = layer8["final_argmax"][i]
            match = "✓" if l8 == f64 else "✗"
            print(f"  pos {i}: layer8={l8} final={f64} {match}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate early-layer argmax prediction for A2 approach"
    )
    p.add_argument("--model", required=True, help="Target model path")
    p.add_argument("--draft", required=True, help="Draft model path")
    p.add_argument(
        "--prompt",
        default="Explain how a CPU cache hierarchy works and why L1 is faster than L3.",
        help="Prompt to use for validation",
    )
    p.add_argument(
        "--draft-len",
        type=int,
        default=12,
        help="Number of draft tokens to test (default: 12)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-position breakdown",
    )
    return p.parse_args(list(argv) if argv is not None else None)


if __name__ == "__main__":
    main()
