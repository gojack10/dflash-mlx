# Copyright 2026 bstnxbt
# Licensed under the Apache License, Version 2.0 - see LICENSE file
# Based on DFlash (arXiv:2602.06036)

from __future__ import annotations

from typing import Any, Optional

import mlx.core as mx

from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
    FullContextDraftKVCache,
)
from dflash_mlx.engine.target_ops import resolve_target_ops

class EagerDraftBackend:
    def make_cache(
        self,
        *,
        draft_model: DFlashDraftModel,
        sink_size: int,
        window_size: int,
    ) -> list[Any]:
        caches: list[Any] = []
        layer_types = tuple(getattr(draft_model.args, "layer_types", ()) or ())
        for index in range(len(draft_model.layers)):
            layer_type = str(layer_types[index] if index < len(layer_types) else "")
            if layer_type == "full_attention":
                caches.append(FullContextDraftKVCache())
            else:
                caches.append(
                    ContextOnlyDraftKVCache(
                        sink_size=sink_size,
                        window_size=window_size,
                    )
                )
        return caches

    def draft_greedy(
        self,
        *,
        target_model: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        target_hidden: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        suppress_token_mask: Optional[mx.array],
        async_launch: bool,
        previous_token_ids: Optional[list[int]] = None,
        repetition_penalty: float = 1.0,
    ) -> mx.array:
        if int(block_len) <= 1:
            raise ValueError("draft_greedy requires block_len > 1")

        block_token_ids = mx.concatenate(
            [staged_first[:1], mask_token_tail[: int(block_len) - 1]],
            axis=0,
        )
        target_ops = resolve_target_ops(target_model)
        noise_embedding = target_ops.embed_tokens(target_model)(block_token_ids[None])
        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = target_ops.logits_from_hidden(target_model, draft_hidden[:, 1:, :])
        from dflash_mlx import runtime as runtime_mod

        draft_logits_squeezed = draft_logits
        if repetition_penalty != 1.0 and previous_token_ids:
            draft_logits_squeezed = runtime_mod.apply_repetition_penalty(
                draft_logits_squeezed, previous_token_ids, repetition_penalty
            )
        drafted = runtime_mod.greedy_tokens_with_mask(
            draft_logits_squeezed,
            suppress_token_mask,
        ).squeeze(0)
        if async_launch:
            mx.async_eval(drafted)
        else:
            mx.eval(draft_logits)
        return drafted

    def _draft_logits_impl(
        self,
        *,
        target_model: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        target_hidden: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
    ) -> mx.array:
        if int(block_len) <= 1:
            raise ValueError("draft logits require block_len > 1")

        block_token_ids = mx.concatenate(
            [staged_first[:1], mask_token_tail[: int(block_len) - 1]],
            axis=0,
        )
        target_ops = resolve_target_ops(target_model)
        noise_embedding = target_ops.embed_tokens(target_model)(block_token_ids[None])
        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        return target_ops.logits_from_hidden(
            target_model, draft_hidden[:, 1:, :]
        )

    def draft_logits(
        self,
        *,
        target_model: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        target_hidden: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
    ) -> mx.array:
        """Return raw per-position logits for DDTree construction.

        Same forward pass as draft_greedy but returns the [B-1, vocab]
        logits tensor instead of argmax tokens. Caller is responsible for
        evaluating / transferring data.
        """
        return self._draft_logits_impl(
            target_model=target_model,
            draft_model=draft_model,
            draft_cache=draft_cache,
            staged_first=staged_first,
            target_hidden=target_hidden,
            block_len=block_len,
            mask_token_tail=mask_token_tail,
        )

    def draft_topk(
        self,
        *,
        target_model: Any,
        draft_model: DFlashDraftModel,
        draft_cache: list[Any],
        staged_first: mx.array,
        target_hidden: mx.array,
        block_len: int,
        mask_token_tail: mx.array,
        topk: int,
        suppress_token_mask: Optional[mx.array],
        use_log_softmax: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Return greedy draft tokens plus top-k log-probs for DDTree.

        Keeps full-vocab logits on device and transfers only top-k IDs/log-probs
        to the tree builder. This avoids a second top-k/argmax pass in the
        DDTree loop and keeps the full logits tensor scoped to this method.

        When ``use_log_softmax`` is True, returns actual log-probabilities
        (log-softmax) instead of centered logit differences.  This adds one
        logsumexp reduction per position but restores the original cumulative-
        log-probability semantics that the DDTree threshold was designed for.
        """
        logits = self._draft_logits_impl(
            target_model=target_model,
            draft_model=draft_model,
            draft_cache=draft_cache,
            staged_first=staged_first,
            target_hidden=target_hidden,
            block_len=block_len,
            mask_token_tail=mask_token_tail,
        )
        masked_logits = logits
        if suppress_token_mask is not None:
            floor = mx.array(-1e9, dtype=logits.dtype)
            masked_logits = mx.where(suppress_token_mask, floor, logits)
        greedy = mx.argmax(masked_logits, axis=-1).astype(mx.uint32).squeeze(0)
        k = max(1, min(int(topk), int(masked_logits.shape[-1])))
        dlogits = masked_logits.astype(mx.float32)
        top_indices = mx.argpartition(-dlogits, kth=k - 1, axis=-1)[:, :, :k]
        top_logits = mx.take_along_axis(dlogits, top_indices, axis=-1)
        sort_order = mx.argsort(-top_logits, axis=-1)
        top_token_ids = mx.take_along_axis(top_indices, sort_order, axis=-1).astype(mx.uint32)
        top_logits = mx.take_along_axis(top_logits, sort_order, axis=-1)
        # DDTree only needs proposal scores to rank draft branches; target
        # verification remains exact.  When use_log_softmax=False (default),
        # center scores on the local best top-k logit: this preserves
        # within-position ordering while avoiding the full softmax normalization.
        # When use_log_softmax=True, compute actual log-probabilities via
        # logsumexp — adds one vocabulary-sized reduction per position but
        # makes cumulative log-probability semantics correct for threshold pruning.
        if use_log_softmax:
            top_scores = (top_logits - mx.logsumexp(dlogits, axis=-1, keepdims=True)).astype(mx.float32)
        else:
            top_scores = (top_logits - top_logits[:, :, :1]).astype(mx.float32)
        return greedy, top_token_ids.squeeze(0), top_scores.squeeze(0)


def make_draft_backend() -> EagerDraftBackend:
    return EagerDraftBackend()
