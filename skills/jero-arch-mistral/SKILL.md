---
name: jero-arch-mistral
description: "Implement and validate Mistral and Mixtral ArchRecipe execution, Sliding Window Attention (SWA), and Sparse Mixture-of-Experts (MoE) top-k routing in jero-arch-mistral."
---

# `jero-arch-mistral` Sliding Window Attention & Sparse MoE Invariants

- Implement `jero_core::ArchRecipe` supporting both dense Mistral (Sliding Window Attention `sliding_window`) and Mixtral Sparse MoE (`expert_count`, `expert_used_count` top-k softmax gating over expert FFN weights).
- Preserve deterministic expert selection and delegates to `jero-kernels-rvv` when the `with-kernels` feature is enabled.
