//! Memory admission control and advisory for Mixture-of-Experts (`MoE`) models under `RLIMIT_MEMLOCK`.
//!
//! Enforces the bounded 28 GiB physical memory ceiling (`29,360,128 KB` per `/etc/security/limits.d/99-modelrunner.conf`)
//! on the `SpacemiT` K3 target platform, preventing unprivileged kernel OOM lockups and guiding
//! deployment toward `Q4_0` or compact `MoE` architectures per ADR-037.

use k3_core::{K3Error, ModelManifest};

/// Standard structured advisory recommended when Mixtral exceeds the 28 GiB memlock ceiling.
pub const COMPACT_MOE_ADVISORY: &str = "Mixtral-8x7B in Q4_K_M exceeds 28 GiB memlock headroom; use Q4_0 or deploy compact MoEs: Qwen3-30B-A3B, DeepSeek-V2-Lite-16B, Gemma-4-26B-A4B";

/// Default target hardware lockable memory ceiling: 28 GiB (`29,360,128 KB` = `30,064,771,072` bytes).
pub const DEFAULT_MEMLOCK_LIMIT_BYTES: u64 = 29_360_128 * 1024;

/// Mixtral-8x7B KV cache bytes per token: 32 layers * 2 (K and V) * 8 KV heads * 128 `head_dim` * 2 bytes (F16/BF16).
pub const MIXTRAL_KV_BYTES_PER_TOKEN: u64 = 32 * 2 * 8 * 128 * 2;

/// Helper rounding `val` up to the next multiple of `align`.
const fn align_up_u64(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

/// Evaluates pre-flight physical memory admission for `MoE` models (specifically Mixtral-8x7B)
/// against the bounded `limit_memlock_bytes` ceiling.
///
/// Computes total lockable memory requirement:
/// $$\text{TotalRequired} = \text{AlignedWeightBytes} + \text{AlignedKvBytes}$$
///
/// If $\text{TotalRequired} > \text{Limit}$, admission is refused with [`K3Error::InsufficientLockableMemory`].
///
/// # Errors
///
/// Returns [`K3Error::InsufficientLockableMemory`] if model weights and KV cache exceed
/// `limit_memlock_bytes`.
pub fn evaluate_moe_memlock_admission(
    manifest: &ModelManifest,
    limit_memlock_bytes: u64,
) -> Result<(), K3Error> {
    // Only Mixtral / MoE models are evaluated against the MoE memlock ceiling
    if !manifest.architecture_id.eq_ignore_ascii_case("mixtral") {
        return Ok(());
    }

    let limit = if limit_memlock_bytes == 0 {
        DEFAULT_MEMLOCK_LIMIT_BYTES
    } else {
        limit_memlock_bytes
    };

    // 1. Determine weight footprint
    let weight_bytes_raw = if manifest.weight_size_bytes > 0 {
        manifest.weight_size_bytes
    } else {
        // Base footprint estimates for Mixtral-8x7B (~46.7B parameters)
        match manifest.quantization.to_ascii_uppercase().as_str() {
            "Q4_K_M" => 28_990_000_000, // ~27.0 GiB
            "Q6_K" => 40_800_000_000,   // ~38.0 GiB
            "Q8_0" => 53_150_000_000,   // ~49.5 GiB
            "F16" | "BF16" => 100_200_000_000,
            _ => 26_300_000_000,        // default Q4_0 (~24.5 GiB)
        }
    };
    let weight_bytes = align_up_u64(weight_bytes_raw, 4096);

    // 2. Determine KV cache footprint
    let kv_bytes_raw = manifest.context_length.saturating_mul(MIXTRAL_KV_BYTES_PER_TOKEN);
    let kv_bytes = align_up_u64(kv_bytes_raw, 4096);

    let total_required = weight_bytes.saturating_add(kv_bytes);

    if total_required > limit {
        return Err(K3Error::InsufficientLockableMemory {
            required_bytes: total_required,
            weight_bytes,
            kv_bytes,
            available_bytes: limit,
            rlimit_memlock_bytes: limit,
        });
    }

    Ok(())
}
