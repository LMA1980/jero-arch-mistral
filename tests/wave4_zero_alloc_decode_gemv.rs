//! Wave 4 integration tests verifying:
//! - Issue #34: Zero steady-state heap allocations in `MistralRecipe::decode_step` /
//!   `decode_step_with_cache` across dense `Mistral3` and sparse top-2 `Mixtral` models using
//!   `128-byte`-aligned `ScratchWorkspace` arenas and pre-built layer/expert tensor keys.
//! - Issue #35: Deterministic bit-identical row-parallel `M = 1` GEMV across `1`, `4`, and `8`
//!   worker partitions on `Q`/`K`/`V`/`O`, dense `Gate`/`Up`/`Down`, active top-2 `Mixtral`
//!   experts, and full `131,072`-row `Tekken` `output.weight` LM heads across all 8 admitted quants.

use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use jero_arch_mistral::{
    MistralLayerCache, MistralRecipe, TEKKEN_VOCAB_SIZE, clear_global_layer_cache,
};
use jero_core::{
    ArchRecipe, ExecutionContext, K3Error, ModelWeights, QuantType, ScratchWorkspace, TensorView,
};

struct CountingAllocator;

static TRACKING: AtomicBool = AtomicBool::new(false);
static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static ALLOC_TEST_LOCK: Mutex<()> = Mutex::new(());
static SUITE_SERIAL_LOCK: Mutex<()> = Mutex::new(());

// SAFETY: `CountingAllocator` delegates exclusively to `std::alloc::System` without violating layout or pointer invariants.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Acquire) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Caller satisfies `GlobalAlloc::alloc` layout invariants; delegated directly to `System`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if TRACKING.load(Ordering::Acquire) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: Caller satisfies `GlobalAlloc::alloc_zeroed` layout invariants; delegated directly to `System`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Pointer and layout originate from the paired `System` allocator above.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if TRACKING.load(Ordering::Acquire) {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        // SAFETY: Pointer, old layout, and new size satisfy `GlobalAlloc::realloc` invariants; delegated to `System`.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOC: CountingAllocator = CountingAllocator;

struct TrackResetGuard;

impl Drop for TrackResetGuard {
    fn drop(&mut self) {
        TRACKING.store(false, Ordering::Release);
    }
}

fn measure_heap_allocations<F: FnOnce() -> R, R>(f: F) -> (R, usize, usize) {
    let _lock = ALLOC_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
    TRACKING.store(true, Ordering::SeqCst);
    let result = {
        let _reset = TrackResetGuard;
        f()
    };
    let count = ALLOC_COUNT.load(Ordering::SeqCst);
    let bytes = ALLOC_BYTES.load(Ordering::SeqCst);
    (result, count, bytes)
}

fn run_in_worker_pool<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    static POOL_4: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    static POOL_8: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    let build_warmed = |n: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("rayon worker pool");
        pool.broadcast(|_| {});
        std::thread::sleep(std::time::Duration::from_millis(20));
        pool
    };
    let pool = if threads <= 4 {
        POOL_4.get_or_init(|| build_warmed(4))
    } else {
        POOL_8.get_or_init(|| build_warmed(8))
    };
    pool.install(f)
}

struct StoredTensor {
    name: String,
    quant: QuantType,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

#[derive(Default)]
struct Wave4TestWeights {
    tensors: HashMap<String, StoredTensor>,
    calls: Mutex<HashMap<String, usize>>,
}

impl Wave4TestWeights {
    fn insert(&mut self, name: &str, quant: QuantType, shape: Vec<usize>, bytes: Vec<u8>) {
        self.tensors.insert(
            name.to_owned(),
            StoredTensor {
                name: name.to_owned(),
                quant,
                shape,
                bytes,
            },
        );
    }

    fn insert_f32(&mut self, name: &str, shape: Vec<usize>, vals: &[f32]) {
        let mut bytes = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        self.insert(name, QuantType::F32, shape, bytes);
    }
}

impl ModelWeights for Wave4TestWeights {
    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn tensor(&self, name: &str) -> Result<TensorView<'_>, K3Error> {
        {
            let mut calls = self.calls.lock().unwrap();
            if let Some(count) = calls.get_mut(name) {
                *count += 1;
            } else {
                calls.insert(name.to_owned(), 1);
            }
        }
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| K3Error::TensorBoundsError {
                tensor_name: name.to_owned(),
                offset: 0,
                byte_len: 0,
                buffer_len: 0,
                reason: format!("missing tensor '{name}'"),
            })?;
        TensorView::from_bytes(&t.name, t.quant, &t.shape, 0, &t.bytes)
    }

    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    fn total_weight_bytes(&self) -> usize {
        self.tensors.values().map(|t| t.bytes.len()).sum()
    }
}

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
    let exp = i32::try_from((bits >> 23) & 0xFF).unwrap_or(0) - 127 + 15;
    let mant = u16::try_from((bits >> 13) & 0x03FF).unwrap_or(0);
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        sign | (u16::try_from(exp).unwrap_or(0) << 10) | mant
    }
}

fn f32_to_bf16_bits(val: f32) -> u16 {
    u16::try_from(val.to_bits() >> 16).unwrap_or(0)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "small synthetic indices fit cleanly"
)]
fn deterministic_scalar(idx: usize, seed: u32) -> f32 {
    let raw = i32::try_from((idx * 17 + seed as usize * 31) % 29).unwrap_or(0) - 14;
    (raw as f32) * 0.025
}

#[expect(
    clippy::cast_precision_loss,
    reason = "small synthetic indices fit cleanly"
)]
fn encode_block(quant: QuantType, block_bytes: usize, block_idx: usize, seed: u32) -> Vec<u8> {
    let d_bits = f32_to_f16_bits(0.02 + ((block_idx % 5) as f32) * 0.005);
    let dmin_bits = f32_to_f16_bits(0.004);
    let mut blk = vec![0_u8; block_bytes];
    match quant {
        QuantType::Q4_0 => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            for (idx, slot) in blk[2..18].iter_mut().enumerate() {
                let lo = u8::try_from((block_idx * 3 + idx + seed as usize) % 16).unwrap_or(0);
                let hi = u8::try_from((block_idx * 5 + idx + 3) % 16).unwrap_or(0);
                *slot = lo | (hi << 4);
            }
        }
        QuantType::Q8_0 => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            for (idx, slot) in blk[2..34].iter_mut().enumerate() {
                let q = i8::try_from(
                    i32::try_from((block_idx * 5 + idx + seed as usize) % 21).unwrap_or(0) - 10,
                )
                .unwrap_or(0);
                *slot = q.to_le_bytes()[0];
            }
        }
        QuantType::Q4_K_M => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            blk[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
            for (j, slot) in blk[4..16].iter_mut().enumerate() {
                *slot = u8::try_from((j + block_idx + seed as usize) % 31 + 1).unwrap_or(1);
            }
            for (j, slot) in blk[16..144].iter_mut().enumerate() {
                *slot =
                    u8::try_from(((j * 7 + block_idx + seed as usize) % 251) & 0x77).unwrap_or(0);
            }
        }
        QuantType::Q5_K => {
            blk[0..2].copy_from_slice(&d_bits.to_le_bytes());
            blk[2..4].copy_from_slice(&dmin_bits.to_le_bytes());
            for (j, slot) in blk[4..16].iter_mut().enumerate() {
                *slot = u8::try_from((j + block_idx + seed as usize) % 31 + 1).unwrap_or(1);
            }
            for (j, slot) in blk[16..48].iter_mut().enumerate() {
                *slot = u8::try_from((j + block_idx) % 255).unwrap_or(0);
            }
            for (j, slot) in blk[48..176].iter_mut().enumerate() {
                *slot = u8::try_from((j * 5 + block_idx + seed as usize) % 251).unwrap_or(0);
            }
        }
        QuantType::Q6_K => {
            blk[208..210].copy_from_slice(&d_bits.to_le_bytes());
            for (j, slot) in blk[0..128].iter_mut().enumerate() {
                *slot = u8::try_from((j * 3 + block_idx + seed as usize) % 251).unwrap_or(0);
            }
            for (j, slot) in blk[128..192].iter_mut().enumerate() {
                *slot = u8::try_from((j + block_idx) % 255).unwrap_or(0);
            }
            blk[192..208].fill(2_u8);
        }
        _ => unreachable!(),
    }
    blk
}

fn encode_matrix(quant: QuantType, rows: usize, cols: usize, seed: u32) -> Vec<u8> {
    match quant {
        QuantType::F32 | QuantType::F16 | QuantType::BF16 => {
            let mut out = Vec::new();
            for r in 0..rows {
                for c in 0..cols {
                    let val = deterministic_scalar(r * cols + c, seed);
                    match quant {
                        QuantType::F32 => out.extend_from_slice(&val.to_le_bytes()),
                        QuantType::F16 => {
                            out.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes());
                        }
                        QuantType::BF16 => {
                            out.extend_from_slice(&f32_to_bf16_bits(val).to_le_bytes());
                        }
                        _ => unreachable!(),
                    }
                }
            }
            out
        }
        q => {
            let block_elems = q.block_elements();
            let block_bytes = q.block_bytes();
            let total_blocks = (rows * cols) / block_elems;
            let mut out = Vec::with_capacity(total_blocks * block_bytes);
            for b in 0..total_blocks {
                out.extend_from_slice(&encode_block(q, block_bytes, b, seed));
            }
            out
        }
    }
}

fn build_dense_mistral3_2layer(
    quant: QuantType,
    vocab_size: usize,
) -> (MistralRecipe, Wave4TestWeights) {
    let mut recipe = MistralRecipe::new_mistral3();
    recipe.num_layers = 2;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 64;
    recipe.vocab_size = vocab_size;
    recipe.sliding_window = Some(4);

    let h = recipe.hidden_dim;
    let total_q = recipe.n_heads_q * recipe.head_dim;
    let total_kv = recipe.n_heads_kv * recipe.head_dim;
    let ffn = recipe.ffn_dim;

    let mut w = Wave4TestWeights::default();
    let emb_rows = vocab_size.min(64);
    w.insert(
        "token_embd.weight",
        QuantType::F32,
        vec![h, emb_rows],
        encode_matrix(QuantType::F32, emb_rows, h, 3),
    );
    w.insert(
        "output.weight",
        quant,
        vec![h, vocab_size],
        encode_matrix(quant, vocab_size, h, 7),
    );
    w.insert_f32("output_norm.weight", vec![h], &vec![1.0_f32; h]);

    for layer in 0..recipe.num_layers {
        let off = u32::try_from(layer * 20).unwrap_or(0);
        w.insert_f32(
            &format!("blk.{layer}.attn_norm.weight"),
            vec![h],
            &vec![1.0_f32; h],
        );
        w.insert_f32(
            &format!("blk.{layer}.ffn_norm.weight"),
            vec![h],
            &vec![1.0_f32; h],
        );
        w.insert(
            &format!("blk.{layer}.attn_q.weight"),
            quant,
            vec![h, total_q],
            encode_matrix(quant, total_q, h, 11 + off),
        );
        w.insert(
            &format!("blk.{layer}.attn_k.weight"),
            quant,
            vec![h, total_kv],
            encode_matrix(quant, total_kv, h, 13 + off),
        );
        w.insert(
            &format!("blk.{layer}.attn_v.weight"),
            quant,
            vec![h, total_kv],
            encode_matrix(quant, total_kv, h, 15 + off),
        );
        w.insert(
            &format!("blk.{layer}.attn_output.weight"),
            quant,
            vec![total_q, h],
            encode_matrix(quant, h, total_q, 17 + off),
        );
        w.insert(
            &format!("blk.{layer}.ffn_gate.weight"),
            quant,
            vec![h, ffn],
            encode_matrix(quant, ffn, h, 19 + off),
        );
        w.insert(
            &format!("blk.{layer}.ffn_up.weight"),
            quant,
            vec![h, ffn],
            encode_matrix(quant, ffn, h, 21 + off),
        );
        w.insert(
            &format!("blk.{layer}.ffn_down.weight"),
            quant,
            vec![ffn, h],
            encode_matrix(quant, h, ffn, 23 + off),
        );
    }

    // Override `token_embd.weight` shape to `[h, vocab_size]` when vocab_size == emb_rows so dims match.
    if vocab_size == emb_rows {
        let bytes = encode_matrix(QuantType::F32, vocab_size, h, 3);
        w.insert(
            "token_embd.weight",
            QuantType::F32,
            vec![h, vocab_size],
            bytes,
        );
    }

    (recipe, w)
}

fn build_dense_mistral3_with_tekken_head(quant: QuantType) -> (MistralRecipe, Wave4TestWeights) {
    let mut recipe = MistralRecipe::new_mistral3();
    recipe.num_layers = 1;
    recipe.hidden_dim = 256;
    recipe.ffn_dim = 256;
    recipe.n_heads_q = 4;
    recipe.n_heads_kv = 2;
    recipe.head_dim = 64;
    recipe.vocab_size = TEKKEN_VOCAB_SIZE;
    recipe.sliding_window = None;

    let h = recipe.hidden_dim;
    let total_q = recipe.n_heads_q * recipe.head_dim;
    let total_kv = recipe.n_heads_kv * recipe.head_dim;
    let ffn = recipe.ffn_dim;

    let mut w = Wave4TestWeights::default();
    w.insert(
        "token_embd.weight",
        QuantType::F32,
        vec![h, TEKKEN_VOCAB_SIZE],
        vec![0.02_f32.to_le_bytes(); h * TEKKEN_VOCAB_SIZE]
            .into_iter()
            .flatten()
            .collect(),
    );
    w.insert_f32("output_norm.weight", vec![h], &vec![1.0_f32; h]);
    w.insert_f32("blk.0.attn_norm.weight", vec![h], &vec![1.0_f32; h]);
    w.insert_f32("blk.0.ffn_norm.weight", vec![h], &vec![1.0_f32; h]);
    w.insert(
        "blk.0.attn_q.weight",
        quant,
        vec![h, total_q],
        encode_matrix(quant, total_q, h, 11),
    );
    w.insert(
        "blk.0.attn_k.weight",
        quant,
        vec![h, total_kv],
        encode_matrix(quant, total_kv, h, 13),
    );
    w.insert(
        "blk.0.attn_v.weight",
        quant,
        vec![h, total_kv],
        encode_matrix(quant, total_kv, h, 15),
    );
    w.insert(
        "blk.0.attn_output.weight",
        quant,
        vec![total_q, h],
        encode_matrix(quant, h, total_q, 17),
    );
    w.insert(
        "blk.0.ffn_gate.weight",
        quant,
        vec![h, ffn],
        encode_matrix(quant, ffn, h, 19),
    );
    w.insert(
        "blk.0.ffn_up.weight",
        quant,
        vec![h, ffn],
        encode_matrix(quant, ffn, h, 21),
    );
    w.insert(
        "blk.0.ffn_down.weight",
        quant,
        vec![ffn, h],
        encode_matrix(quant, h, ffn, 23),
    );

    // Repeat a 64-row quantized tile cleanly across all 131,072 Tekken rows.
    let tile_rows = 64_usize;
    let tile_bytes = encode_matrix(quant, tile_rows, h, 29);
    let mut lm_head = Vec::with_capacity(tile_bytes.len() * (TEKKEN_VOCAB_SIZE / tile_rows));
    for _ in 0..(TEKKEN_VOCAB_SIZE / tile_rows) {
        lm_head.extend_from_slice(&tile_bytes);
    }
    w.insert("output.weight", quant, vec![h, TEKKEN_VOCAB_SIZE], lm_head);

    (recipe, w)
}

fn build_mixtral_2layer_3d_exps(quant: QuantType) -> (MistralRecipe, Wave4TestWeights) {
    let (mut recipe, mut w) = build_dense_mistral3_2layer(quant, 64);
    recipe.architecture = "mixtral";
    recipe = recipe.with_experts(4, 2);

    let h = recipe.hidden_dim;
    let ffn = recipe.ffn_dim;
    let mut gate_inp = vec![0.0_f32; 4 * h];
    gate_inp[0] = 5.0;
    gate_inp[h] = 2.5;
    gate_inp[2 * h] = 4.0;
    for layer in 0..recipe.num_layers {
        let off = u32::try_from(layer * 30).unwrap_or(0);
        w.insert_f32(
            &format!("blk.{layer}.ffn_gate_inp.weight"),
            vec![h, 4],
            &gate_inp,
        );
        for (proj_name, rows, cols, seed_base) in [
            ("gate", ffn, h, 40),
            ("up", ffn, h, 50),
            ("down", h, ffn, 60),
        ] {
            let mut stacked = Vec::new();
            for e in 0..4_u32 {
                stacked.extend_from_slice(&encode_matrix(
                    quant,
                    rows,
                    cols,
                    seed_base + off + e * 3,
                ));
            }
            w.insert(
                &format!("blk.{layer}.ffn_{proj_name}_exps.weight"),
                quant,
                vec![cols, rows, 4],
                stacked,
            );
        }
    }
    (recipe, w)
}

fn verify_dense_mistral3_partition_parity(
    base_recipe: &MistralRecipe,
    weights: &Wave4TestWeights,
    quant: QuantType,
    n_ctx: usize,
) {
    let kv_len = 2 * base_recipe.num_layers * n_ctx * base_recipe.n_heads_kv * base_recipe.head_dim;
    let mut baseline_bits = Vec::new();
    for parts in [1_usize, 4, 8] {
        let r = base_recipe.clone().with_worker_partitions(parts);
        let scratch_cap = r
            .estimate_decode_scratch_bytes(n_ctx)
            .expect("scratch bytes");
        let mut ws = ScratchWorkspace::new(scratch_cap).expect("ws");
        let mut cache = MistralLayerCache::new();
        let mut kv = vec![0.0_f32; kv_len];
        let mut logits = vec![0.0_f32; r.vocab_size];

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(parts)
            .build()
            .expect("rayon pool");
        let bits = pool.install(|| {
            let mut ctx = ExecutionContext::new(&[0], n_ctx, &mut logits, &mut kv)
                .expect("ctx")
                .with_scratch(&mut ws);
            r.prefill_step_with_cache(&mut ctx, weights, &mut cache)
                .expect("prefill");
            for tok in [1_u32, 2, 3] {
                ctx.prepare_decode(tok).expect("prepare decode");
                r.decode_step_with_cache(&mut ctx, weights, &mut cache)
                    .expect("decode");
            }
            assert!(
                logits.iter().all(|v| v.is_finite()),
                "quant={quant:?} non-finite"
            );
            logits.iter().map(|v| v.to_bits()).collect::<Vec<u32>>()
        });
        if baseline_bits.is_empty() {
            baseline_bits = bits;
        } else {
            assert_eq!(
                bits, baseline_bits,
                "quant={quant:?} partition={parts} broke bit-identical decode logits"
            );
        }
    }
}

fn verify_dense_mistral3_zero_alloc(
    base_recipe: &MistralRecipe,
    weights: &Wave4TestWeights,
    quant: QuantType,
    n_ctx: usize,
) {
    let kv_len = 2 * base_recipe.num_layers * n_ctx * base_recipe.n_heads_kv * base_recipe.head_dim;
    let r4 = base_recipe.clone().with_worker_partitions(4);
    let scratch_cap = r4
        .estimate_decode_scratch_bytes(n_ctx)
        .expect("scratch bytes");
    let mut ws = ScratchWorkspace::new(scratch_cap).expect("ws");
    let mut cache = MistralLayerCache::new();
    let mut kv = vec![0.0_f32; kv_len];
    let mut logits = vec![0.0_f32; r4.vocab_size];
    r4.warm_layer_cache(weights, &mut cache)
        .expect("warm cache");
    let mut ctx = ExecutionContext::new(&[0], n_ctx, &mut logits, &mut kv)
        .expect("ctx")
        .with_scratch(&mut ws);
    r4.prefill_step_with_cache(&mut ctx, weights, &mut cache)
        .expect("prefill warmup");
    let (step_res, alloc_count, alloc_bytes) = run_in_worker_pool(4, || {
        ctx.prepare_decode(1).unwrap();
        r4.decode_step_with_cache(&mut ctx, weights, &mut cache)
            .expect("decode warmup");
        measure_heap_allocations(|| {
            for tok in 2_u32..6 {
                ctx.prepare_decode(tok).unwrap();
                r4.decode_step_with_cache(&mut ctx, weights, &mut cache)?;
            }
            Ok::<(), K3Error>(())
        })
    });
    step_res.expect("steady-state decode");
    assert_eq!(
        (alloc_count, alloc_bytes),
        (0, 0),
        "quant={quant:?} leaked heap allocations during steady-state decode"
    );
    assert!(ws.peak_used_bytes() > 0 && ws.peak_used_bytes() <= ws.capacity_bytes());

    let mut internal_ctx =
        ExecutionContext::new(&[0], n_ctx, &mut logits, &mut kv).expect("internal ctx");
    r4.prefill_step_with_cache(&mut internal_ctx, weights, &mut cache)
        .expect("internal prefill");
    let (internal_res, int_count, int_bytes) = run_in_worker_pool(4, || {
        internal_ctx.prepare_decode(1).unwrap();
        r4.decode_step_with_cache(&mut internal_ctx, weights, &mut cache)
            .expect("internal decode warmup");
        measure_heap_allocations(|| {
            for tok in 2_u32..6 {
                internal_ctx.prepare_decode(tok).unwrap();
                r4.decode_step_with_cache(&mut internal_ctx, weights, &mut cache)?;
            }
            Ok::<(), K3Error>(())
        })
    });
    internal_res.expect("internal scratch steady-state decode");
    assert_eq!(
        (int_count, int_bytes),
        (0, 0),
        "quant={quant:?} leaked heap allocations using cache internal decode scratch"
    );
}

#[test]
fn test_wave4_dense_mistral3_zero_alloc_decode_all_quants_and_partitions() {
    let _suite_guard = SUITE_SERIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let quants = [
        QuantType::F32,
        QuantType::F16,
        QuantType::BF16,
        QuantType::Q4_0,
        QuantType::Q4_K_M,
        QuantType::Q5_K,
        QuantType::Q6_K,
        QuantType::Q8_0,
    ];

    for quant in quants {
        let (base_recipe, weights) = build_dense_mistral3_2layer(quant, 64);
        verify_dense_mistral3_partition_parity(&base_recipe, &weights, quant, 16);
        verify_dense_mistral3_zero_alloc(&base_recipe, &weights, quant, 16);
    }
}

#[test]
fn test_wave4_mixtral_moe_zero_alloc_decode_and_partition_determinism() {
    let _suite_guard = SUITE_SERIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for quant in [
        QuantType::F32,
        QuantType::F16,
        QuantType::Q4_0,
        QuantType::Q4_K_M,
    ] {
        let (base_recipe, weights) = build_mixtral_2layer_3d_exps(quant);
        let n_ctx = 16_usize;
        let kv_len =
            2 * base_recipe.num_layers * n_ctx * base_recipe.n_heads_kv * base_recipe.head_dim;

        let mut baseline_bits = Vec::new();
        for parts in [1_usize, 4, 8] {
            let r = base_recipe.clone().with_worker_partitions(parts);
            let mut ws =
                ScratchWorkspace::new(r.estimate_decode_scratch_bytes(n_ctx).unwrap()).unwrap();
            let mut cache = MistralLayerCache::new();
            let mut kv = vec![0.0_f32; kv_len];
            let mut logits = vec![0.0_f32; r.vocab_size];
            let mut ctx = ExecutionContext::new(&[0], n_ctx, &mut logits, &mut kv)
                .unwrap()
                .with_scratch(&mut ws);
            r.prefill_step_with_cache(&mut ctx, &weights, &mut cache)
                .expect("mixtral prefill");
            ctx.prepare_decode(1).unwrap();
            r.decode_step_with_cache(&mut ctx, &weights, &mut cache)
                .expect("mixtral decode");
            assert!(cache.stats().expert_zero_copy_slices > 0);
            let bits: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect();
            assert!(logits.iter().all(|v| v.is_finite()));
            if baseline_bits.is_empty() {
                baseline_bits = bits;
            } else {
                assert_eq!(
                    bits, baseline_bits,
                    "Mixtral quant={quant:?} partition={parts} mismatch"
                );
            }
        }

        // Steady-state Mixtral decode zero-allocation verification.
        let r8 = base_recipe.clone().with_worker_partitions(8);
        let mut ws =
            ScratchWorkspace::new(r8.estimate_decode_scratch_bytes(n_ctx).unwrap()).unwrap();
        let mut cache = MistralLayerCache::new();
        r8.warm_layer_cache(&weights, &mut cache).unwrap();
        let resident_before = cache.resident_repacked_bytes();
        let mut kv = vec![0.0_f32; kv_len];
        let mut logits = vec![0.0_f32; r8.vocab_size];
        let mut ctx = ExecutionContext::new(&[0], n_ctx, &mut logits, &mut kv)
            .unwrap()
            .with_scratch(&mut ws);
        r8.prefill_step_with_cache(&mut ctx, &weights, &mut cache)
            .unwrap();
        let (res, count, bytes) = run_in_worker_pool(8, || {
            ctx.prepare_decode(1).unwrap();
            r8.decode_step_with_cache(&mut ctx, &weights, &mut cache)
                .unwrap();
            measure_heap_allocations(|| {
                for tok in 2_u32..5 {
                    ctx.prepare_decode(tok).unwrap();
                    r8.decode_step_with_cache(&mut ctx, &weights, &mut cache)?;
                }
                Ok::<(), K3Error>(())
            })
        });
        res.expect("steady-state mixtral decode");
        assert_eq!(cache.resident_repacked_bytes(), resident_before);
        assert_eq!((count, bytes), (0, 0), "quant={quant:?} mixtral leaked");
    }
}

#[test]
fn test_wave4_tekken_131072_vocab_lm_head_zero_alloc_row_parallel_decode() {
    let _suite_guard = SUITE_SERIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (base_recipe, weights) = build_dense_mistral3_with_tekken_head(QuantType::Q4_0);
    let n_ctx = 4_usize;
    let kv_len = 2 * base_recipe.num_layers * n_ctx * base_recipe.n_heads_kv * base_recipe.head_dim;

    let mut baseline_bits = Vec::new();
    for parts in [1_usize, 4, 8] {
        let r = base_recipe.clone().with_worker_partitions(parts);
        let mut ws =
            ScratchWorkspace::new(r.estimate_decode_scratch_bytes(n_ctx).unwrap()).unwrap();
        let mut cache = MistralLayerCache::new();
        let mut kv = vec![0.0_f32; kv_len];
        let mut logits = vec![0.0_f32; TEKKEN_VOCAB_SIZE];
        let mut ctx = ExecutionContext::new(&[0], n_ctx, &mut logits, &mut kv)
            .unwrap()
            .with_scratch(&mut ws);
        r.prefill_step_with_cache(&mut ctx, &weights, &mut cache)
            .expect("tekken prefill");
        ctx.prepare_decode(1).unwrap();
        r.decode_step_with_cache(&mut ctx, &weights, &mut cache)
            .expect("tekken decode");
        assert_eq!(logits.len(), TEKKEN_VOCAB_SIZE);
        assert!(logits.iter().all(|v| v.is_finite()));
        // LM head (`output.weight`) must not be expanded inside resident dense projection cache.
        assert!(cache.resident_repacked_bytes() < 8 * 1024 * 1024);

        let bits: Vec<u32> = logits.iter().map(|v| v.to_bits()).collect();
        if baseline_bits.is_empty() {
            baseline_bits = bits;
        } else {
            assert_eq!(
                bits, baseline_bits,
                "131k Tekken LM head differed at worker_partitions={parts}"
            );
        }
    }

    // Zero heap allocation verification over 131,072 Tekken rows.
    let r4 = base_recipe.with_worker_partitions(4);
    let mut ws = ScratchWorkspace::new(r4.estimate_decode_scratch_bytes(n_ctx).unwrap()).unwrap();
    let mut cache = MistralLayerCache::new();
    r4.warm_layer_cache(&weights, &mut cache).unwrap();
    let mut kv = vec![0.0_f32; kv_len];
    let mut logits = vec![0.0_f32; TEKKEN_VOCAB_SIZE];
    let mut ctx = ExecutionContext::new(&[0], n_ctx, &mut logits, &mut kv)
        .unwrap()
        .with_scratch(&mut ws);
    r4.prefill_step_with_cache(&mut ctx, &weights, &mut cache)
        .unwrap();
    let (res, count, bytes) = run_in_worker_pool(4, || {
        ctx.prepare_decode(1).unwrap();
        r4.decode_step_with_cache(&mut ctx, &weights, &mut cache)
            .unwrap();
        measure_heap_allocations(|| {
            ctx.prepare_decode(2).unwrap();
            r4.decode_step_with_cache(&mut ctx, &weights, &mut cache)
        })
    });
    res.expect("steady-state 131k decode");
    assert_eq!(
        (count, bytes),
        (0, 0),
        "131k Tekken decode leaked heap bytes"
    );
}

#[test]
fn test_wave4_scratch_capacity_rejection_direct_fallback_and_trait_decode_zero_alloc() {
    let _suite_guard = SUITE_SERIAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_global_layer_cache();
    let (base_recipe, weights) = build_dense_mistral3_2layer(QuantType::Q4_0, 64);
    let r = base_recipe.with_worker_partitions(4);
    let n_ctx = 8_usize;
    let kv_len = 2 * r.num_layers * n_ctx * r.n_heads_kv * r.head_dim;

    // 1. Undersized caller `ScratchWorkspace` must fail cleanly without falling back to heap Vecs.
    let mut tiny_ws = ScratchWorkspace::new(64).expect("tiny scratch");
    let mut tiny_cache = MistralLayerCache::new();
    let mut kv = vec![0.0_f32; kv_len];
    let mut logits = vec![0.0_f32; r.vocab_size];
    let mut tiny_ctx = ExecutionContext::new(&[0], n_ctx, &mut logits, &mut kv)
        .unwrap()
        .with_scratch(&mut tiny_ws);
    tiny_ctx.prepare_decode(1).unwrap();
    assert!(matches!(
        r.decode_step_with_cache(&mut tiny_ctx, &weights, &mut tiny_cache),
        Err(K3Error::TensorBoundsError { .. })
    ));

    // 2. DirectFallback (`max_repacked_bytes == 0`) achieves strict 0 heap allocations in both
    //    `with-kernels` and `--no-default-features` builds.
    let scratch_cap = r.estimate_decode_scratch_bytes(n_ctx).unwrap();
    let mut ws = ScratchWorkspace::new(scratch_cap).unwrap();
    let mut fb_cache = MistralLayerCache::new().with_max_repacked_bytes(0);
    r.warm_layer_cache(&weights, &mut fb_cache).unwrap();
    assert_eq!(fb_cache.resident_repacked_bytes(), 0);
    let mut fb_ctx = ExecutionContext::new(&[0], n_ctx, &mut logits, &mut kv)
        .unwrap()
        .with_scratch(&mut ws);
    r.prefill_step_with_cache(&mut fb_ctx, &weights, &mut fb_cache)
        .unwrap();
    let (fb_res, fb_count, fb_bytes) = run_in_worker_pool(4, || {
        fb_ctx.prepare_decode(1).unwrap();
        r.decode_step_with_cache(&mut fb_ctx, &weights, &mut fb_cache)
            .unwrap();
        measure_heap_allocations(|| {
            for tok in 2_u32..5 {
                fb_ctx.prepare_decode(tok).unwrap();
                r.decode_step_with_cache(&mut fb_ctx, &weights, &mut fb_cache)?;
            }
            Ok::<(), K3Error>(())
        })
    });
    fb_res.expect("direct fallback decode");
    assert_eq!(
        (fb_count, fb_bytes),
        (0, 0),
        "DirectFallback decode must allocate zero heap bytes under both feature modes"
    );

    // 3. Trait entrypoint `ArchRecipe::decode_step` using global shared cache + `with_scratch`.
    let mut trait_kv = vec![0.0_f32; kv_len];
    let mut trait_logits = vec![0.0_f32; r.vocab_size];
    let mut trait_ctx = ExecutionContext::new(&[0], n_ctx, &mut trait_logits, &mut trait_kv)
        .unwrap()
        .with_scratch(&mut ws);
    r.prefill_step(&mut trait_ctx, &weights).unwrap();

    let (trait_res, trait_count, trait_bytes) = run_in_worker_pool(4, || {
        trait_ctx.prepare_decode(1).unwrap();
        r.decode_step(&mut trait_ctx, &weights).unwrap();
        measure_heap_allocations(|| {
            for tok in 2_u32..5 {
                trait_ctx.prepare_decode(tok).unwrap();
                r.decode_step(&mut trait_ctx, &weights)?;
            }
            Ok::<(), K3Error>(())
        })
    });
    trait_res.expect("trait decode_step");
    assert_eq!(
        (trait_count, trait_bytes),
        (0, 0),
        "ArchRecipe::decode_step leaked heap bytes"
    );
}
