# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
"""DDTree vs DFlash benchmark — matches server config exactly.

Usage:
    # Match your server config (DDTree mode):
    python -m tools.benchmarks.ddtree_profile \
        --model mlx-community/Qwen3.5-27B-4bit \
        --draft z-lab/Qwen3.5-27B-DFlash \
        --prompt "Write a Python function that..." \
        --max-tokens 512 \
        --profile long-session \
        --speculative-mode ddtree \
        --ddtree-budget 12 \
        --ddtree-topk 12 \
        --repetition-penalty 1.1 \
        --prefill-step-size 8192

    # Compare with vanilla DFlash:
    python -m tools.benchmarks.ddtree_profile \
        --model ... --draft ... --prompt "..." \
        --profile long-session \
        --speculative-mode dflash
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections.abc import Sequence
from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.generate import load_runtime_components
from dflash_mlx.runtime import stream_dflash_generate
from dflash_mlx.runtime_context import (
    build_runtime_context,
    runtime_config_from_profile,
)

def _ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0


def run_benchmark(
    *,
    model_ref: str,
    draft_ref: str,
    prompt: str,
    max_tokens: int,
    profile: str = "long-session",
    speculative_mode: str = "dflash",
    ddtree_budget: int = 12,
    ddtree_topk: int = 12,
    repetition_penalty: float = 1.0,
    prefill_step_size: int | None = None,
    draft_block_tokens: int | None = None,
) -> dict[str, Any]:
    # ── Load models ──
    target_model, tokenizer, draft_model, resolved_draft = load_runtime_components(
        model_ref=model_ref,
        draft_ref=draft_ref,
    )

    # ── Build runtime context matching server config ──
    runtime_config = runtime_config_from_profile(
        profile=profile,
        speculative_mode=speculative_mode,
        ddtree_budget=ddtree_budget,
        ddtree_topk=ddtree_topk,
        repetition_penalty=repetition_penalty,
        prefill_step_size=prefill_step_size,
        draft_block_tokens=draft_block_tokens,
        prefix_cache=False,
        prefix_cache_l2=False,
        memory_waterfall=False,
        clear_cache_boundaries=False,
        generation_snapshot=False,
    )
    runtime_context = build_runtime_context(runtime_config)

    # ── Tokenize prompt ──
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        prompt_tokens = tokenizer.encode(prompt)

    # ── Run generation ──
    summary = None
    token_count = 0
    wall_start_ns = time.perf_counter_ns()

    stream = stream_dflash_generate(
        target_model=target_model,
        tokenizer=tokenizer,
        draft_model=draft_model,
        prompt=prompt,
        max_new_tokens=max_tokens,
        use_chat_template=True,
        prompt_tokens_override=list(prompt_tokens),
        runtime_context=runtime_context,
    )

    try:
        for event in stream:
            if event.get("event") == "token":
                token_count += 1
            elif event.get("event") == "summary":
                summary = dict(event)
    finally:
        stream.close()

    wall_ns = time.perf_counter_ns() - wall_start_ns
    del target_model, tokenizer, draft_model
    gc.collect()
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()

    if summary is None:
        raise RuntimeError("No summary event received")

    # ── Extract key metrics ──
    phase_us = summary.get("phase_timings_us", summary.get("phase_timings", {}))
    ddtree_us = summary.get("ddtree_timing_avg_us", {})
    ddtree_totals_us = summary.get("ddtree_timing_totals_us", {})
    ddtree_meta = summary.get("ddtree", {})

    result = {
        "config": {
            "model": model_ref,
            "draft": resolved_draft,
            "prompt_tokens": len(prompt_tokens),
            "max_tokens": max_tokens,
            "speculative_mode": speculative_mode,
            "ddtree_budget": ddtree_budget,
            "ddtree_topk": ddtree_topk,
            "repetition_penalty": repetition_penalty,
            "profile": profile,
        },
        "results": {
            "generated_tokens": token_count,
            "wall_s": wall_ns / 1e9,
            "wall_tok_s": token_count / (wall_ns / 1e9) if wall_ns > 0 else 0,
            "core_us": summary.get("elapsed_us", 0),
            "core_tok_s": (
                token_count / (summary.get("elapsed_us", 1) / 1e6)
                if summary.get("elapsed_us", 0) > 0
                else 0
            ),
            "acceptance_pct": float(summary.get("acceptance_ratio", 0)) * 100,
            "tokens_per_cycle": summary.get("tokens_per_cycle", 0),
            "cycles": summary.get("cycles_completed", 0),
            "phase_timings_ms": {
                k: float(v) / 1000
                for k, v in phase_us.items()
                if v is not None
            },
        },
    }

    if ddtree_us:
        result["ddtree"] = {
            "meta": ddtree_meta,
            "avg_ms": {
                k: float(v) / 1000 for k, v in ddtree_us.items()
            },
            "totals_ms": {
                k: float(v) / 1000 for k, v in ddtree_totals_us.items()
            },
        }

    return result


def print_result(result: dict[str, Any]) -> None:
    cfg = result["config"]
    res = result["results"]
    print(f"\n{'='*70}")
    print(f"DDTree Profile Benchmark")
    print(f"{'='*70}")
    print(f"Mode:        {cfg['speculative_mode']}")
    print(f"Model:       {cfg['model']}")
    print(f"Draft:       {cfg['draft']}")
    print(f"Prompt len:  {cfg['prompt_tokens']} tokens")
    print(f"Max tokens:  {cfg['max_tokens']}")
    if cfg["speculative_mode"] == "ddtree":
        print(f"DDTree:      budget={cfg['ddtree_budget']} topk={cfg['ddtree_topk']}")
    print(f"Rep penalty: {cfg['repetition_penalty']}")
    print()
    print(f"Generated:   {res['generated_tokens']} tokens")
    print(f"Wall time:   {res['wall_s']:.1f}s")
    print(f"Wall tok/s:  {res['wall_tok_s']:.1f}")
    print(f"Core tok/s:  {res['core_tok_s']:.1f}")
    print(f"Acceptance:  {res['acceptance_pct']:.1f}%")
    print(f"Tokens/cyc:  {res['tokens_per_cycle']:.2f}")
    print(f"Cycles:      {res['cycles']}")

    # Phase timing breakdown
    phases = res.get("phase_timings_ms", {})
    if phases:
        print(f"\n── Phase timings (ms total) ──")
        for name, ms in sorted(phases.items(), key=lambda x: -x[1]):
            pct = (ms / max(1, sum(phases.values()))) * 100
            bar = "█" * int(pct / 2)
            print(f"  {name:<20s} {ms:>8.1f}ms  {bar} {pct:.1f}%")

    # DDTree timing breakdown
    ddtree = result.get("ddtree", {})
    if ddtree:
        avg = ddtree.get("avg_ms", {})
        cycle_ms = avg.get("cycle_total", 1)
        print(f"\n── DDTree per-cycle breakdown (ms) ──")
        ordered = [
            "posterior_argmax", "tree_build", "topk_sync", "target_verify",
            "cache_commit", "next_draft_launch", "draft_launch",
            "topk_transfer", "compile", "accepted_gather", "tree_walk",
        ]
        for name in ordered:
            if name in avg:
                ms = avg[name]
                pct = (ms / max(0.001, cycle_ms)) * 100
                bar = "█" * int(pct / 2)
                print(f"  {name:<22s} {ms:>8.2f}ms  {bar} {pct:.1f}%")
        # Print any remaining keys
        for name, ms in sorted(avg.items(), key=lambda x: -x[1]):
            if name not in ordered and name != "cycle_total":
                pct = (ms / max(0.001, cycle_ms)) * 100
                bar = "█" * int(pct / 2)
                print(f"  {name:<22s} {ms:>8.2f}ms  {bar} {pct:.1f}%")
        print(f"  {'cycle_total':<22s} {cycle_ms:>8.2f}ms")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DDTree vs DFlash profile benchmark — matches server config"
    )
    p.add_argument("--model", required=True, help="Target model reference")
    p.add_argument("--draft", required=True, help="Draft model reference")
    p.add_argument("--prompt", required=True, help="Prompt text")
    p.add_argument("--max-tokens", type=int, default=512, help="Tokens to generate")
    p.add_argument(
        "--profile", default="long-session",
        choices=("balanced", "fast", "low-memory", "long-session"),
        help="Runtime profile (default: long-session)",
    )
    p.add_argument(
        "--speculative-mode", default="dflash",
        choices=("dflash", "ddtree"),
        help="Speculative mode (default: dflash)",
    )
    p.add_argument("--ddtree-budget", type=int, default=12)
    p.add_argument("--ddtree-topk", type=int, default=12)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--prefill-step-size", type=int, default=None)
    p.add_argument("--draft-block-tokens", type=int, default=None)
    p.add_argument("--json", action="store_true", help="Output JSON")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    result = run_benchmark(
        model_ref=args.model,
        draft_ref=args.draft,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        profile=args.profile,
        speculative_mode=args.speculative_mode,
        ddtree_budget=args.ddtree_budget,
        ddtree_topk=args.ddtree_topk,
        repetition_penalty=args.repetition_penalty,
        prefill_step_size=args.prefill_step_size,
        draft_block_tokens=args.draft_block_tokens,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
