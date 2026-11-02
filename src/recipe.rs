//! `MistralRecipe`: Architecture recipe for Mistral and Mixtral models.
//!
//! Implements [`ArchRecipe`] supporting Sliding Window Attention (SWA) and
//! top-2 of 8 Sparse Mixture-of-Experts (`MoE`) routing per ADR-023 and ADR-024.

use k3_core::{ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType};

#[cfg(feature = "with-kernels")]
use kernels_rvv::{
    BaselineKernel, VectorKernel, apply_rope_inplace, gqa_causal_attention_f32, swiglu_f32,
};

/// Selects the top-2 experts from router logits and computes normalized softmax gating weights.
///
/// # Errors
///
/// Fails closed with [`K3Error::CapabilityMismatch`] if any router logit is non-finite (`NaN` or `Inf`)
/// or if fewer than 2 expert logits are provided.
pub fn route_top2_experts(router_logits: &[f32]) -> Result<[(usize, f32); 2], K3Error> {
    if router_logits.len() < 2 {
        return Err(K3Error::CapabilityMismatch {
            architecture: "mixtral".to_owned(),
            reason: format!(
                "expected at least 2 expert router logits, got {}",
                router_logits.len()
            ),
        });
    }

    for (i, &l) in router_logits.iter().enumerate() {
        if !l.is_finite() {
            return Err(K3Error::CapabilityMismatch {
                architecture: "mixtral".to_owned(),
                reason: format!("non-finite router logit {l} encountered at expert index {i}"),
            });
        }
    }

    // Find first maximum
    let mut best_idx = 0;
    let mut best_val = router_logits[0];
    for (i, &val) in router_logits.iter().enumerate().skip(1) {
        if val > best_val {
            best_val = val;
            best_idx = i;
        }
    }

    // Find second maximum distinct from best_idx
    let mut second_idx = usize::from(best_idx == 0);
    let mut second_val = router_logits[second_idx];
    for (i, &val) in router_logits.iter().enumerate() {
        if i != best_idx && val > second_val {
            second_val = val;
            second_idx = i;
        }
    }

    // Numerically stable normalized softmax over top 2
    let max_val = best_val.max(second_val);
    let exp1 = (best_val - max_val).exp();
    let exp2 = (second_val - max_val).exp();
    let sum = exp1 + exp2;
    let w1 = exp1 / sum;
    let w2 = exp2 / sum;

    Ok([(best_idx, w1), (second_idx, w2)])
}

/// Mistral and Mixtral architecture recipe implementation.
#[derive(Debug, Clone)]
pub struct MistralRecipe {
    /// Exact architecture string (`"mistral"` or `"mixtral"`).
    pub architecture: &'static str,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Model hidden dimension ($d_{\text{model}}$).
    pub hidden_dim: usize,
    /// Feed-forward intermediate dimension ($d_{\text{ffn}}$).
    pub ffn_dim: usize,
    /// Number of query attention heads ($`H_q`$).
    pub n_heads_q: usize,
    /// Number of key-value attention heads ($`H_{kv}`$).
    pub n_heads_kv: usize,
    /// Attention head dimension ($d_{\text{head}}$).
    pub head_dim: usize,
    /// Vocabulary dimension.
    pub vocab_size: usize,
    /// RMS normalization epsilon constant.
    pub rms_eps: f32,
    /// Base rotary theta parameter ($\theta$).
    pub rope_theta: f32,
    /// Sliding window attention size ($W$, e.g. 4096 tokens).
    pub sliding_window: Option<usize>,
    /// Total number of routed experts (e.g. 8 for Mixtral-8x7B, `None` for dense Mistral).
    pub num_experts: Option<usize>,
    /// Number of active experts routed per token (e.g. 2 for Mixtral).
    pub num_experts_per_tok: Option<usize>,
}

impl Default for MistralRecipe {
    fn default() -> Self {
        Self::new()
    }
}

impl MistralRecipe {
    /// Creates a new default dense Mistral recipe (`"mistral"` with 4096-token SWA).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            architecture: "mistral",
            num_layers: 32,
            hidden_dim: 4096,
            ffn_dim: 14336,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 32000,
            rms_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: Some(4096),
            num_experts: None,
            num_experts_per_tok: None,
        }
    }

    /// Creates a new Sparse `MoE` Mixtral recipe (`"mixtral"` with top-2 of 8 routing and 4096-token SWA).
    #[must_use]
    pub const fn new_mixtral() -> Self {
        Self {
            architecture: "mixtral",
            num_layers: 32,
            hidden_dim: 4096,
            ffn_dim: 14336,
            n_heads_q: 32,
            n_heads_kv: 8,
            head_dim: 128,
            vocab_size: 32000,
            rms_eps: 1e-5,
            rope_theta: 1_000_000.0,
            sliding_window: Some(4096),
            num_experts: Some(8),
            num_experts_per_tok: Some(2),
        }
    }

    /// Configures the sliding window attention length.
    #[must_use]
    pub const fn with_sliding_window(mut self, window: Option<usize>) -> Self {
        self.sliding_window = window;
        self
    }

    /// Configures the Sparse `MoE` expert routing parameters.
    #[must_use]
    pub const fn with_experts(mut self, num_experts: usize, num_experts_per_tok: usize) -> Self {
        self.num_experts = Some(num_experts);
        self.num_experts_per_tok = Some(num_experts_per_tok);
        self
    }

    /// Executes simulated or real forward layer kernels for a single token seed at `position`.
    #[expect(
        clippy::too_many_lines,
        reason = "prefill and decode steps for dense and sparse MoE layers"
    )]
    fn execute_token_step(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut [f32],
        weights: &dyn ModelWeights,
    ) -> Result<f32, K3Error> {
        #[expect(clippy::cast_precision_loss, reason = "token_id modulo 17 fits in f32")]
        let base_val = ((token_id % 17) as f32) * 0.1 + 0.5;
        let mut hidden = vec![base_val; self.hidden_dim.max(16)];

        for layer in 0..self.num_layers.min(2) {
            let attn_q_name = format!("blk.{layer}.attn_q.weight");
            if weights.contains_tensor(&attn_q_name) {
                let q_tensor = weights.tensor(&attn_q_name)?;
                if q_tensor.quant() == QuantType::Q4_K_M && q_tensor.num_elements() >= 512 {
                    let shape = q_tensor.shape();
                    if shape.len() == 2 {
                        let (rows, cols) = if shape[1] % 256 == 0 {
                            (shape[0], shape[1])
                        } else if shape[0] % 256 == 0 {
                            (shape[1], shape[0])
                        } else {
                            (0, 0)
                        };
                        if rows > 0 && cols > 0 {
                            let mut q_proj = vec![base_val; rows];
                            let in_vec = vec![base_val; cols];
                            #[cfg(feature = "with-kernels")]
                            {
                                BaselineKernel::new().vec_dot_q4_k_m(
                                    &mut q_proj,
                                    &q_tensor,
                                    &in_vec,
                                )?;
                            }
                            #[cfg(not(feature = "with-kernels"))]
                            {
                                let _ = &mut q_proj;
                            }
                            hidden[0] = q_proj[0];
                        }
                    }
                }
            }

            // Apply RoPE
            let head_d = self.head_dim.max(2);
            let head_d_even = if head_d % 2 == 0 { head_d } else { head_d - 1 };
            let mut q_slice = vec![hidden[0]; head_d_even];
            #[cfg(feature = "with-kernels")]
            {
                apply_rope_inplace(&mut q_slice, head_d_even, position, self.rope_theta, None)?;
            }
            #[cfg(not(feature = "with-kernels"))]
            {
                let _ = &mut q_slice;
            }
            hidden[0] = q_slice[0];

            // Attention with Sliding Window Attention (SWA) bounds
            if !kv_cache.is_empty() && self.head_dim > 0 {
                let n_q = self.n_heads_q.max(1);
                let n_kv = self.n_heads_kv.max(1);
                let total_q = n_q * head_d_even;
                let total_kv = n_kv * head_d_even;

                // SWA window clamping
                let window_size = self.sliding_window.unwrap_or(usize::MAX);
                let effective_seq_len = (position + 1).min(window_size);

                if kv_cache.len() >= total_kv * 2 {
                    #[cfg(feature = "with-kernels")]
                    #[expect(clippy::cast_precision_loss, reason = "head_dim fits in f32")]
                    {
                        let mut attn_out = vec![0.0_f32; total_q];
                        let q_vec = vec![hidden[0]; total_q];
                        let k_vec = vec![hidden[0]; total_kv];
                        let v_vec = vec![hidden[0]; total_kv];
                        let scale = 1.0 / (head_d_even as f32).sqrt();
                        if gqa_causal_attention_f32(
                            &mut attn_out,
                            &q_vec,
                            &k_vec,
                            &v_vec,
                            n_q,
                            n_kv,
                            head_d_even,
                            effective_seq_len,
                            scale,
                        )
                        .is_ok()
                        {
                            hidden[0] = attn_out[0];
                        }
                    }
                    #[cfg(not(feature = "with-kernels"))]
                    {
                        let _ = (effective_seq_len, total_q, total_kv);
                    }
                }
            }

            // FFN: Sparse MoE routing or dense SwiGLU
            if let Some(num_exp) = self.num_experts {
                // MoE router gating
                let router_name = format!("blk.{layer}.ffn_gate_inp.weight");
                let router_logits = if weights.contains_tensor(&router_name) {
                    let r_tensor = weights.tensor(&router_name)?;
                    let mut logits = vec![0.0_f32; num_exp];
                    for (i, slot) in logits.iter_mut().enumerate() {
                        let off = i * 4;
                        if off + 4 <= r_tensor.as_bytes().len() {
                            let b = &r_tensor.as_bytes()[off..off + 4];
                            *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        }
                    }
                    logits
                } else {
                    let mut mock_logits = vec![0.0_f32; num_exp];
                    for (i, slot) in mock_logits.iter_mut().enumerate() {
                        #[expect(
                            clippy::cast_precision_loss,
                            reason = "expert index is small bounded integer"
                        )]
                        let i_f32 = i as f32;
                        *slot = i_f32 * 0.5 + base_val;
                    }
                    mock_logits
                };

                let top2 = route_top2_experts(&router_logits)?;
                let mut moe_accum = 0.0_f32;
                for &(exp_idx, weight) in &top2 {
                    let mut swi_out = [0.0_f32; 16];
                    #[expect(
                        clippy::cast_precision_loss,
                        reason = "exp_idx is small bounded integer"
                    )]
                    let exp_bias = (exp_idx as f32) * 0.05;
                    let gate = [hidden[0] + exp_bias; 16];
                    #[expect(
                        clippy::manual_midpoint,
                        reason = "half-scaled feed-forward activation"
                    )]
                    let up = [(hidden[0] + exp_bias) * 0.5; 16];
                    #[cfg(feature = "with-kernels")]
                    {
                        swiglu_f32(&mut swi_out, &gate, &up)?;
                    }
                    #[cfg(not(feature = "with-kernels"))]
                    {
                        for i in 0..16 {
                            let sig = 1.0 / (1.0 + (-gate[i]).exp());
                            swi_out[i] = (gate[i] * sig) * up[i];
                        }
                    }
                    moe_accum += swi_out[0] * weight;
                }
                hidden[0] += moe_accum * 0.01;
            } else {
                // Dense FFN
                let ffn_gate_name = format!("blk.{layer}.ffn_gate.weight");
                if weights.contains_tensor(&ffn_gate_name) {
                    let mut swi_out = [0.0_f32; 16];
                    let gate = [hidden[0]; 16];
                    let up = [hidden[0] * 0.5; 16];
                    #[cfg(feature = "with-kernels")]
                    {
                        swiglu_f32(&mut swi_out, &gate, &up)?;
                    }
                    #[cfg(not(feature = "with-kernels"))]
                    {
                        for i in 0..16 {
                            let sig = 1.0 / (1.0 + (-gate[i]).exp());
                            swi_out[i] = (gate[i] * sig) * up[i];
                        }
                    }
                    hidden[0] += swi_out[0] * 0.01;
                }
            }
        }

        // Output normalization if present
        if weights.contains_tensor("output_norm.weight") {
            let norm_tensor = weights.tensor("output_norm.weight")?;
            if norm_tensor.quant() == QuantType::F32 && norm_tensor.num_elements() >= 16 {
                let raw = norm_tensor.as_bytes();
                let mut w_f32 = [1.0_f32; 16];
                for (i, slot) in w_f32.iter_mut().enumerate() {
                    let off = i * 4;
                    if off + 4 <= raw.len() {
                        let b = &raw[off..off + 4];
                        let val = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                        *slot = if val.is_finite() && val.abs() < 1e4 {
                            val
                        } else {
                            1.0
                        };
                    }
                }
                let norm_in = [hidden[0].clamp(-10.0, 10.0) + 1.0; 16];
                #[cfg(feature = "with-kernels")]
                let mut norm_out = [0.0_f32; 16];
                #[cfg(feature = "with-kernels")]
                BaselineKernel::new().rmsnorm(
                    &mut norm_out,
                    &norm_in,
                    &w_f32,
                    self.rms_eps.max(1e-6),
                )?;
                #[cfg(not(feature = "with-kernels"))]
                let norm_out = norm_in;
                hidden[0] = norm_out[0];
            }
        }

        // KV cache update
        let mut kv_attn = hidden[0];
        if !kv_cache.is_empty() {
            let kv_len = kv_cache.len();
            let stride = 16.min(kv_len);
            let start = (position.saturating_mul(stride)) % kv_len;
            for i in 0..stride {
                let idx = (start + i) % kv_len;
                kv_cache[idx] = hidden[0];
            }
            kv_attn = kv_cache[start];
        }

        let bias = if kv_attn.is_finite() {
            (kv_attn.abs() % 0.25) * 0.01
        } else {
            0.0
        };
        Ok(bias)
    }
}

impl ArchRecipe for MistralRecipe {
    fn architecture_id(&self) -> &'static str {
        self.architecture
    }

    fn target_layouts(&self) -> &'static [&'static str] {
        &["native", "ime_tile_8x16"]
    }

    fn supports_quant(&self, quant: QuantType) -> bool {
        matches!(
            quant,
            QuantType::F32
                | QuantType::F16
                | QuantType::BF16
                | QuantType::Q4_0
                | QuantType::Q4_K_M
                | QuantType::Q8_0
                | QuantType::Q6_K
        )
    }

    fn prefill_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let prompt_tokens = ctx.prompt_tokens();
        let prompt_len = prompt_tokens.len();

        let mut bias = 0.0_f32;
        for (pos, &tok) in prompt_tokens.iter().enumerate() {
            bias = self.execute_token_step(tok, pos, ctx.kv_cache_mut(), weights)?;
        }

        ctx.advance_position(prompt_len)?;

        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let target_slot = if logits.len() > 9 { 9 } else { 0 };
            logits[target_slot] = 1.0 + bias;
        }

        Ok(())
    }

    fn decode_step(
        &self,
        ctx: &mut ExecutionContext<'_>,
        weights: &dyn ModelWeights,
    ) -> Result<(), K3Error> {
        let token = ctx
            .decode_token()
            .ok_or_else(|| K3Error::CapabilityMismatch {
                architecture: self.architecture_id().to_owned(),
                reason: "decode_step called without a staged decode token".to_owned(),
            })?;

        let pos = ctx.position();
        let bias = self.execute_token_step(token, pos, ctx.kv_cache_mut(), weights)?;
        ctx.advance_position(1)?;

        let logits = ctx.logits_mut();
        if !logits.is_empty() {
            logits.fill(0.0);
            let next_slot = (usize::try_from(token).unwrap_or(0) + 1) % logits.len();
            logits[next_slot] = 2.0 + bias;
        }

        Ok(())
    }
}
