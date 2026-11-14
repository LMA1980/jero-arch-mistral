---
name: jero-arch-mistral
description: "Implement and validate Mistral, Mistral3, and Mixtral ArchRecipe execution, Sliding Window Attention (SWA), zero-allocation ScratchWorkspace decode, and row-parallel multi-row GEMV in jero-arch-mistral."
---

# `jero-arch-mistral` Execution, Caching & Row-Parallel Decode Invariants

- Implement `jero_core::ArchRecipe` supporting dense `mistral`/`mistral3` (Sliding Window Attention `sliding_window` and decoupled GQA head dimensions) and `mixtral` Sparse MoE (`expert_count`, `expert_used_count` top-k softmax gating via allocation-free `route_topk_into` over expert FFN weights).
- Eliminate per-token and per-layer heap allocations in `MistralRecipe::decode_step` / `decode_step_with_cache` by allocating `128-byte`-aligned temporary slices and GEMV staging pools from `ExecutionContext::with_scratch` (or `MistralLayerCache`'s reusable internal `ScratchWorkspace`), using `gqa_causal_attention_with_scratch_f32`, pre-built `LayerTensorKeys`, and `fill_rope_inv_freq_compat`.
- Partition output rows of dense `Q`/`K`/`V`/`O` and `Gate`/`Up`/`Down`, active top-k `Mixtral` expert `Gate`/`Up`/`Down`, and `output.weight` (`131,072`-row `Tekken` LM head) on `8`-row tile boundaries across `WorkerPool` / `rayon` worker threads (`with_worker_partitions`) for bit-identical `M = 1` multi-row GEMV determinism without `TILE_M = 8` zero-row padding waste.
- Execute batched `[M, K]` dense prefill (`Q`/`K`/`V`/`O` and dense `SwiGLU`) via `gemm_repacked_rows`, batched `Norm` RoPE across `start_position..end_position`, causal + `SWA` `gqa_causal_attention_prefill_f32`, and final-row-only (`M - 1`) parallel LM-head projection gated on `ExecutionContext::compute_logits()`.
- Execute `mixtral` prefill via token-bucketed active-expert `[M_e, hidden_dim]` GEMMs (`repack_transient_projection` within memlock headroom or zero-copy fallback), loading each active expert at most once per layer batch without mutating resident expert cache bytes.
- Cache `128-byte`-aligned dense projections and once-decoded `f32` norm vectors in `MistralLayerCache` with content-verified anti-ABA invalidation and bounded `RLIMIT_MEMLOCK` headroom, while slicing `3D` `_exps` `[n_embd, ffn_dim, n_expert]` tensors zero-copy per active expert without duplicating expert RAM.
- Validate `SentencePiece` (`32,000` / `32,768`) and `Tekken` (`131,072`) tokenizer metadata, `[INST]` chat framing, special/stop token bounds, LM-head logit buffer sizing, and `weights + KV + scratch <= RLIMIT_MEMLOCK` admission before page locking.
- Preserve deterministic expert selection and delegate to `jero-kernels-rvv` when the `with-kernels` feature is enabled.
