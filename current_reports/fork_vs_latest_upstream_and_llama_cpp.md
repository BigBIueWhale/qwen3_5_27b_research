# Qwen 3.5 Fork vs Latest Upstream: Comprehensive Research Report

**Date:** 2026-03-06
**Fork:** `BigBIueWhale/ollama` @ `57e3f80f` (8 commits atop `v0.17.4` merge base `cc90a035`)
**Upstream Ollama (previous):** `82848a78` (analyzed in `fork_vs_upstream_analysis.md`)
**Upstream Ollama (latest):** `9896e36` (4 commits newer than `82848a78`)
**llama.cpp (previous):** `d969e933`
**llama.cpp (latest):** `a0ed91a` (March 5–6, 2026)
**Clones:** `/tmp/ollama-latest/`, `/tmp/llama-cpp-latest/`, `/tmp/llama-cpp-master/` (d969e933), `/tmp/ollama-master/` (82848a78)

---

## Executive Summary

The fork's critical fixes (prefill bug, `mropeInterleaved` default, penalty sampler architecture, `repeat_last_n` wiring) remain **unfixed in upstream Ollama**. llama.cpp's sampler is also unchanged since `d969e933`. The most important new findings are:

1. **CUDA async copy / reduced synchronization** — llama.cpp gained a meaningful performance optimization that the fork's vendored GGML lacks
2. **M-RoPE `can_shift()` guard** — llama.cpp added a correctness fix that prevents corrupt K-shift on M-RoPE models; the fork's vendored llama.cpp lacks this but the Go runner gracefully degrades
3. **History `<think>` blocks bug** — both fork and upstream incorrectly gate history thinking blocks on `isThinking`; the official template always includes them for messages after `lastQueryIndex`, regardless of `enable_thinking`
4. **Speculative decoding** is confirmed impossible for hybrid recurrent architectures in both llama.cpp and Ollama — the recurrent state fundamentally cannot be rolled back
5. **Parallelism** is restricted at the scheduler level (`sched.go:450`) despite the runner supporting it — both the fork and upstream have this limitation
6. **Thinking level constraint** was loosened in upstream — the fork should incorporate this (but it's a hack, not a proper fix)
7. **KDA chunk size** change doesn't affect Qwen 3.5 (it uses GDA, not KDA)
8. **Vision prompt caching** was added to llama.cpp's server — Ollama has no equivalent

---

## Part 1: What Changed in llama.cpp Since d969e933

### 1.1 CRITICAL PERFORMANCE: CUDA Async Copy + Reduced Synchronization (commit `2cd20b7`)

**What changed:** Two files in `ggml/src/`:

**`ggml-cuda.cu`**: `ggml_backend_cuda_cpy_tensor_async()` now supports **CPU→CUDA async copies** in addition to CUDA→CUDA. The old version (in the fork at `ggml-cuda.cu:2923`) requires both source and destination to be CUDA backends. The new version detects `ggml_backend_buffer_is_host(buf_src) && ggml_backend_dev_type(backend_src->device) == GGML_BACKEND_DEVICE_TYPE_CPU` and issues `cudaMemcpyAsync(..., cudaMemcpyHostToDevice, ...)`. This eliminates a synchronous copy that previously blocked the CPU thread.

**`ggml-backend.cpp`**: The scheduler's `compute_splits()` was restructured. The old pattern (in the fork at `ggml-backend.cpp:1603`) performs per-input synchronization:
```cpp
// OLD (fork): sync happens per-input inside the copy loop
if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
    ggml_backend_synchronize(input_backend);  // blocks CPU per input tensor
    ggml_backend_tensor_copy(input, input_cpy);
}
```

The new pattern moves synchronization to happen **once** before the input copies begin and **once** after graph compute, using event-based signaling — a "sync-async-async-async-sync-graph" (saaasg) pattern. This reduces CPU-GPU sync points from O(inputs) to O(1) per compute split.

**Blackwell-specific?** No. Zero references to sm_100, Blackwell, or Ada in this commit. This is a general CUDA improvement benefiting all NVIDIA GPUs.

**Fork impact:** The fork vendors GGML at the `v0.17.4` level. Both the CUDA async copy and the scheduler sync pattern are missing. This is the single most impactful **performance** change llama.cpp has made since `d969e933` for CUDA users. The improvement is most noticeable with partial GPU offload (many input tensors copied per split) and multi-GPU setups.

**How to adopt:** The fork would need to update its vendored GGML (`ml/backend/ggml/ggml/src/ggml-backend.cpp` and `ggml-cuda/ggml-cuda.cu`). This is a focused change to two files in the GGML layer — it does not touch model code or the Go runner. However, the scheduler restructuring is a moderately complex diff because the surrounding code also changed (expert-level copying was reworked). Cherry-picking cleanly may require manual conflict resolution.

### 1.2 CORRECTNESS: M-RoPE `can_shift()` Guard (commit `99bd67c`)

**What changed:** `llama-kv-cache.cpp` — `get_can_shift()` now returns `false` for models with `n_pos_per_embd() > 1`:

```cpp
// NEW (llama.cpp latest, line 976-985):
bool llama_kv_cache::get_can_shift() const {
    if (model.arch == LLM_ARCH_STEP35) { return false; }
    if (hparams.n_pos_per_embd() > 1) { return false; }  // M-RoPE guard
    return true;
}

// OLD (fork's vendored llama.cpp, line 968-970):
bool llama_kv_cache::get_can_shift() const {
    return true;  // always true — no M-RoPE check
}
```

K-shift (context window sliding) requires a single scalar position per token. M-RoPE models have 4 position values per token (temporal, height, width, extra). Attempting K-shift on M-RoPE models would corrupt positional encoding.

**Which models are affected:** Models with `LLAMA_ROPE_TYPE_IMROPE` (Qwen 3.5 dense/MoE use IMROPE) or `LLAMA_ROPE_TYPE_MROPE` (Qwen2VL, Qwen3VL).

**Fork impact — nuanced:** The fork's `can_shift()` returns `true` unconditionally, but this only affects the **C++ `llamarunner`** path. Qwen3next models use the **Go `ollamarunner`** (because they're in the `OllamaEngineRequired()` list). The Go runner has its own cache management and handles shift failures gracefully:

```go
// ollamarunner/cache.go:298-314 — graceful degradation:
err := c.cache.Remove(slot.Id, numKeep, numKeep+discard)
if err != nil {
    // Falls back to full reprocessing instead of corrupt shift
    return &ErrReprocessInputs{Inputs: newInputs}
}
```

So for Qwen 3.5 specifically, this is **not a correctness risk in the fork** because the Go runner never calls `can_shift()`. However, if other M-RoPE models (Qwen2VL, Qwen3VL) use the C++ `llamarunner` path, they would be affected. **Should still be adopted** as a defensive fix.

### 1.3 KDA Chunk Size (commit `a0ed91a`) — NO IMPACT

**What changed:** `delta-net-base.cpp` — chunk size changed from hardcoded `64` to `kda ? 16 : 64`.

**KDA** (Key-Dependent Attention) is used by Kimi-linear models where the gate tensor has shape `[head_dim, n_head, ...]`. **GDA** (Gate-Dependent Attention) is used by Qwen 3.5 where the gate has shape `[1, num_v_heads, ...]`.

Qwen 3.5 uses GDA → chunk size remains 64 → no change in behavior. The fork's `chunkSize = 64` at `deltanet.go:12` is correct.

**If** the fork ever adds KDA model support (Kimi-linear), it would need conditional chunk sizing. Not needed now.

### 1.4 Qwen 3.5 Model Type Detection (commit `872646b`)

**What changed:** `llama-model.cpp` — updated size detection:
- `QWEN35` (dense): Added 0.8B (24 layers, 1024 embd), 4B (32 layers, 2560 embd), 9B (32 layers), 27B (64 layers)
- `QWEN35MOE`: Fixed layer counts (28→40 for 35B_A3B, added 122B_A10B and 397B_A17B)

**Fork impact:** Ollama's Go runner doesn't use llama.cpp's model type detection system — it reads architecture and layer config directly from GGUF metadata. The fork routes `qwen35`/`qwen35moe`/`qwen3next` all through the same `qwen3next` Go model code. No action needed.

**However, this reveals that the Qwen 3.5 model family is larger than initially thought:** There are now 0.8B, 2B, 4B, 9B, and 27B dense variants, plus 35B-A3B, 122B-A10B, and 397B-A17B MoE variants. The fork should verify that its `qwen3next` model code handles all these sizes correctly (particularly the smaller dense models which may have different layer counts and embedding dimensions).

### 1.5 Text-Only Qwen 3.5 Converter Registration (commit `cf23251`)

**What changed:** `convert_hf_to_gguf.py` — now registers `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM` in addition to `ForConditionalGeneration` variants. Text-only Qwen 3.5 HuggingFace checkpoints use `ForCausalLM` class names.

**Fork impact:** Ollama has its own Go converter (`convert/convert_qwen3next.go`) that determines architecture from config fields, not HF class names. Not directly applicable.

### 1.6 Multi-Modal Prompt Caching (commits `f20469d`, `d7d826b`)

**What changed:** llama.cpp's server now caches multi-modal prompts. Previously, when a cached prompt contained images, the cache was invalidated and the prompt was reprocessed from text tokens only. Now:
- `server_tokens::clone()` preserves multi-modal data across cache entries
- Context checkpoints support non-contiguous token/position mappings from vision embeddings
- `pos_next` variable decouples position tracking from token count

**Fork impact:** These are llama.cpp server-specific changes. Ollama's Go server has its own cache system. However, the **concept** is important: if a user sends repeated requests with the same image (common in agentic vision workflows), the first request processes the image through the 27-layer vision encoder, but subsequent requests must re-encode it because the cache doesn't preserve multi-modal embeddings. Implementing multi-modal prompt caching in Ollama's Go runner would be a significant performance win for vision-heavy workloads.

### 1.7 CUDA Performance — CDNA3 MFMA Flash Attention (commit `ecbcb7e`)

**What changed:** Added AMD MI300X (gfx942 CDNA3) tensor core support for flash attention using `v_mfma_f32_16x16x16f16` MFMA intrinsics. Benchmarks: pp512 +7% to pp4096 +39%.

**Fork impact:** Only relevant for AMD Instinct datacenter GPUs. The fork's vendored GGML lacks `AMD_MFMA_AVAILABLE`. Low priority for consumer hardware.

### 1.8 Vulkan Flash Attention AMD Fix (commit `723c710`)

**What changed:** Fixed `subgroupShuffleXor` with `f16vec4` operands broken on AMD Windows RDNA2 and below (proprietary driver). Workaround: cast to `vec4` (float) for shuffle, cast back.

**Fork impact:** Affects AMD GPU users on Windows with RDNA2 or older. The fork's vendored Vulkan shaders lack this fix.

### 1.9 MXFP4 CPU Repack (commit `d903f30`)

**What changed:** Added CPU repack (interleaving) kernels for MXFP4 quantization: `block_mxfp4x4`, `block_mxfp4x8` formats with corresponding `gemv`/`gemm` kernels for x86 and ARM.

**Fork impact:** MXFP4 is the microscaled FP4 format. The fork's vendored GGML has the `QK_MXFP4` type but lacks these repack kernels. This affects CPU-only inference with MXFP4 quantizations (Unsloth uses dynamic mixed precision, but the primary Q4_K_XL format uses Q4_K/Q6_K/Q8_0 types, not MXFP4). **Low priority** unless users load MXFP4-specific GGUFs on CPU.

### 1.10 Context Checkpoint Restore Fix (commit `01cd448`)

**What changed:** Prevents `n_past` from exceeding checkpoint's actual token count during restore. Single-line fix in `server-context.cpp`.

**Fork impact:** llama.cpp server-specific. Ollama's Go server has its own checkpoint system (`kvcache/recurrent_checkpoints.go`).

### 1.11 Sampler/Penalty System — NO CHANGES

The penalty sampler in `llama-sampler.cpp` is **byte-identical** between `d969e933` and the latest. Same ring buffer, same `accept()` behavior, same prompt-token feeding in `server-context.cpp:197-219`. Defaults unchanged: `penalty_repeat=1.0` (disabled), `penalty_last_n=64`.

**The fork's penalty architecture remains the only correct implementation per the original Keskar et al. (2019) CTRL paper.** Both llama.cpp and upstream Ollama continue to feed prompt tokens into the penalty window.

---

## Part 2: What Changed in Upstream Ollama Since 82848a78

### 2.1 New Commits (4 total)

| Commit | Description | Qwen 3.5 Impact |
|--------|-------------|-----------------|
| `9896e36` | Fix cloud model limit lookups in integrations | None — IDE integration only |
| `15732f0` | Use native Ollama API endpoint for OpenClaw | None — OpenClaw integration |
| `562c76d` | Add qwen3.5 context length for launch | Minor — adds `{Context: 262144, Output: 32768}` for IDE integrations |
| `122c68c` | Loosen thinking level constraint | **Medium** — removes validation that rejected string-valued `think` parameters |

### 2.2 MEDIUM: Thinking Level Constraint (commit `122c68c`) — FORK SHOULD INCORPORATE

Upstream removed the `IsString()` validation from both `GenerateHandler` and `ChatHandler` in `server/routes.go`. This allows passing string thinking levels (e.g., `"think": "detailed"`) to any model, not just harmony/gptoss models.

The fork still has this restrictive validation. While Qwen 3.5 doesn't currently use string thinking levels, removing the constraint is correct — it's a server-level policy that shouldn't restrict model capabilities. **The fork should cherry-pick this.**

### 2.3 CRITICAL BUGS — STILL UNFIXED IN UPSTREAM

Every critical bug identified in the previous analysis remains present in upstream Ollama at `9896e36`:

| Bug | Upstream Status | Verification |
|-----|----------------|--------------|
| **Prefill bug** — `qwen35.go:136` missing `&& len(message.ToolCalls) == 0` | **STILL BROKEN** | Verified: `prefill := lastMessage && message.Role == "assistant"` |
| **`mropeInterleaved` default** — `model.go:641` defaults to `false` | **STILL BROKEN** | Verified: `c.Bool("mrope_interleaved", false)` |
| **`repeat_last_n` not wired** — `NewSampler` ignores API parameter | **STILL BROKEN** | Verified: `DefaultPenaltyLookback = 64` hardcoded |
| **Penalty sampler feeds prompt tokens** — `Accept()` called on prompt | **STILL BROKEN** | Verified: `seq.sampler.Accept(inp.Token)` at `runner.go:700,956` |
| **`</think>` not closed in `Qwen3VLRenderer`** — gated on `content != ""` | **STILL BROKEN** | Not touched by any new commit |

### 2.4 Upstream Sampler Architecture (commit `86513cb`, pre-82848a78)

For completeness, the upstream sampler architecture (added in `86513cb`, which predates `82848a78`) uses:
- Growing `[]int32` slice, trimmed to `DefaultPenaltyLookback = 64` on overflow (O(n) per trim)
- Public `Accept()` method called from 3 runner call sites (cache load, pending inputs, generated tokens)
- `Reset()` on sequence initialization and reprocess error
- No `repeatLastN` parameter — hardcoded to 64

The fork's architecture is strictly superior:
- Fixed-size ring buffer, O(1) per token, zero allocations after warmup
- Private `recordToken()` called only from `Sample()` — no prompt token contamination
- Configurable `repeatLastN` from API
- All 3 penalty types (repeat, presence, frequency) operate on generated tokens only

---

## Part 3: Speculative Decoding — Confirmed Impossible for Hybrid Recurrent

### 3.1 llama.cpp's Explicit Block

llama.cpp has a compatibility test at `common/speculative.cpp:801-835`:

```cpp
bool common_speculative_is_compat(llama_context * ctx) {
    // ... decodes 2 dummy tokens ...
    bool ok = llama_memory_seq_rm(mem, 0, 1, -1);  // partial tail removal
    // ... if false, speculative decoding disabled ...
}
```

For hybrid models (Qwen3Next, Qwen35), the memory is `llama_memory_hybrid`, which delegates `seq_rm` to its recurrent sub-cache. The recurrent cache's `seq_rm` at `llama-memory-recurrent.cpp:142-180` **explicitly rejects partial tail removal**:

```cpp
// partial intersection is invalid if it includes the final pos
if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
    return false;  // cannot roll back recurrent state
}
```

Both `LLM_ARCH_QWEN35` and `LLM_ARCH_QWEN3NEXT` are classified as hybrid in `llama-arch.cpp:2825-2842`. The `can_spec` flag evaluates to `false` → speculative decoding is disabled → the server logs `"speculative decoding not supported by this context"`.

### 3.2 The Fundamental Blocker

Recurrent hidden state (GatedDeltaNet, Mamba) is **destructively updated** token by token: `S_{t+1} = α ⊙ S_t + β_t^T v_t`. If a draft token at position `t` is rejected, the state `S_{t+1}` has already been computed and stored. Rolling back to `S_t` requires either:

1. **Fine-grained per-token checkpointing** — store a copy of `S` before each draft token. For Qwen 3.5 27B with 48 recurrent layers, each with conv state + recurrent state matrices, this is prohibitively expensive in memory.
2. **Recomputation from the last checkpoint** — replay all tokens from the last good checkpoint to the rejection point. Ollama actually has a checkpoint system (`kvcache/recurrent_checkpoints.go`) that saves state every 1664 tokens, but this granularity is far too coarse for speculative decoding (which needs token-level rollback).
3. **Run recurrent layers non-speculatively** — only speculate on the 16 attention layers (25% of model), run recurrent layers (75%) normally. This would negate most of the speedup benefit.

### 3.3 Ollama's Checkpoint System — An Interesting Building Block

The fork has a `CheckpointCache` system at `kvcache/recurrent_checkpoints.go` that llama.cpp lacks:
- Saves full snapshots of conv + recurrent state during forward passes
- Default: every 1664 tokens, up to 24 checkpoints per slot
- Implements `PrepareRestore(seq, targetPos)` to find the best checkpoint before a target position
- Used for **prefix reuse** (resuming from a previous prompt), not speculative rollback

This is architecturally interesting because it demonstrates that recurrent state checkpointing is feasible. However, to support speculative decoding, the checkpoint interval would need to shrink from 1664 to 1 (per draft token), and the memory cost would multiply by the number of draft tokens (typically 4-16). For a 27B model with 48 recurrent layers, each checkpoint is ~hundreds of MB.

### 3.4 Bottom Line

Speculative decoding for Qwen 3.5 is not a matter of implementation difficulty — it's a **fundamental architectural incompatibility**. The only viable path would be a fundamentally different approach:
- **Medusa-style** (parallel generation heads without draft model) — but this requires model retraining
- **Lookahead decoding** (n-gram-based speculative execution) — but this still requires state rollback
- **Speculate only on attention layers** — severely limited benefit since 75% of the model is recurrent
- **Accept the overhead of fine-grained checkpointing** — trading memory for throughput

None of these are near-term practical. **The fork should not invest in speculative decoding for Qwen 3.5.**

---

## Part 4: Parallelism — Restricted but Architecturally Supported

### 4.1 The Restriction

**`server/sched.go:448-453`** — both the fork and upstream Ollama force `numParallel = 1` for all hybrid/recurrent architectures:

```go
if slices.Contains([]string{
    "mllama", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe",
    "qwen3next", "lfm2", "lfm2moe", "nemotron_h", "nemotron_h_moe",
}, req.model.Config.ModelFamily) && numParallel != 1 {
    numParallel = 1
    slog.Warn("model architecture does not currently support parallel requests", ...)
}
```

This passes `numParallel = 1` to `NewLlamaServer`, which sets `KvSize = opts.NumCtx * 1` and `Parallel: 1`.

### 4.2 The Runner DOES Support Parallelism

The Go runner (`ollamarunner`) and the cache system (`kvcache/`) are architecturally capable of handling multiple parallel sequences:

- **`kvcache/recurrent.go:179-267`**: The `Recurrent` cache maintains a `slotForSeq` map, allocating separate slots with independent conv + recurrent states per sequence. Multiple sequences are supported simultaneously.
- **`ollamarunner/runner.go:1216-1219`**: The runner only restricts parallelism when `Config().Cache == nil`. For qwen3next, `m.Cache = NewHybridCache(...)` is non-nil → no runner-level restriction.
- **llama.cpp's approach**: `llama_memory_hybrid` creates recurrent state slots sized to `n_seq_max` (which equals `n_parallel`). Each sequence gets its own cell. No architecture-specific restriction in the server.

### 4.3 Why It's Restricted Anyway

The `ErrUnsupportedBatchLayout` at `deltanet.go:80-87` requires all sequences in a batch to have the **same token count**. This is a structural requirement of the chunked linear attention computation — the chunk boundaries must align across sequences. In practice:
- During prompt evaluation, different requests have different prompt lengths → can't batch together
- During generation, each sequence produces 1 token per step → could theoretically batch (all have count=1)
- But mixed prompt-eval + generation batches (one sequence evaluating, another generating) can't be batched

llama.cpp handles this with `split_equal(n_ubatch, true)` in `llama_memory_hybrid` — each sequence is processed in its own sub-batch. The sequences run sequentially within a step, but they can be interleaved across steps.

### 4.4 Opportunity: Lift the Restriction with Careful Batching

The scheduler restriction is conservative. With per-sequence sub-batching (which the Go runner already partially supports via its batch construction logic), multiple parallel sequences could run without violating the equal-token-count constraint. Each sequence would get its own sub-batch. The throughput gain would come from:
- **Pipeline overlap**: while one sequence's sub-batch runs on GPU, the next can be prepared
- **Shared GPU context**: all sequences share the same loaded model weights

This is a non-trivial change but architecturally possible. llama.cpp demonstrates it works. The blocker is Ollama's batch construction logic in `ollamarunner/runner.go`, which currently builds a single batch with all sequences' tokens mixed together.

### 4.5 Vision and Parallelism

Vision adds another dimension to the parallelism story:

**Ollama's vision approach** (unified GGUF):
- The 27-layer vision encoder runs inside `EncodeMultimodal` at `model.go:297-319`
- This is called from the runner at `ollamarunner/runner.go:236-284`
- Vision encoding happens **synchronously** within the request processing pipeline
- With `numParallel = 1`, only one request can encode images at a time

**llama.cpp's vision approach** (separate mmproj):
- Vision encoder loaded via `--mmproj` flag into separate `clip_ctx`
- Vision prompt caching (commits `f20469d`, `d7d826b`) allows reusing encoded images
- Multi-GPU support: vision encoder and text model can theoretically run on different devices

**Key difference for vision parallelism:** llama.cpp's recent multi-modal prompt caching means repeated vision requests (same image, different text prompts — common in agentic workflows) avoid redundant vision encoding. Ollama re-encodes the image for every request. This is a significant performance gap for vision-intensive workloads.

---

## Part 5: GGML Vendor Differences — The Fork's Hidden Debt

The fork vendors GGML from the `v0.17.4` tag. Since then, llama.cpp has accumulated several GGML-level improvements that affect the fork:

### 5.1 Performance-Affecting GGML Changes

| Change | Commit | Impact | Priority |
|--------|--------|--------|----------|
| **CUDA async CPU→GPU copy + saaasg sync pattern** | `2cd20b7` | Reduces CPU-GPU sync points from O(inputs) to O(1) per split. Most impactful for partial offload and multi-GPU. | **HIGH** |
| **CDNA3 MFMA flash attention** | `ecbcb7e` | +7-39% prompt eval on AMD MI300X | Low (datacenter only) |
| **MXFP4 CPU repack kernels** | `d903f30` | Faster CPU inference with MXFP4 quantizations | Low (Q4_K_XL uses Q4_K, not MXFP4) |
| **Vulkan partial offload** | `3191462` | Improved performance on AMD Vulkan | Medium |
| **AMX batched support fix** | `4e76d24` | Intel AMX matmul fix | Low |
| **kleidiai SME fp16 for q4_0** | `137435f` | ARM aarch64 performance | Low |

### 5.2 Correctness-Affecting GGML Changes

| Change | Commit | Impact | Priority |
|--------|--------|--------|----------|
| **M-RoPE `can_shift()` guard** | `99bd67c` | Prevents corrupt K-shift on M-RoPE models. Go runner gracefully degrades, but C++ llamarunner is affected. | **MEDIUM** |
| **`ggml_is_contiguous_n` fix** | `7f5ee54` | Fixes contiguity check when `ne == 1` | Low (edge case) |
| **Vulkan fp16 FA AMD fix** | `723c710` | Fixes broken flash attention on AMD RDNA2 Windows | Medium (for AMD users) |
| **Vulkan memory overlap fusion check** | `3769fe6` | Prevents fusion when memory overlaps | Low |

### 5.3 How to Update

The fork vendors GGML at `ml/backend/ggml/ggml/src/`. Updating individual files is possible but risky due to interdependencies. The recommended approach:
1. **Targeted cherry-pick** of the CUDA async copy change (`ggml-cuda.cu` and `ggml-backend.cpp`) — highest ROI
2. **Targeted cherry-pick** of the M-RoPE `can_shift()` fix (`llama-kv-cache.cpp`) — straightforward 3-line addition
3. **Full GGML vendor update** when Ollama v0.18 or v0.19 is released — picks up everything but requires more testing

---

## Part 6: Correctness Deep-Dive

### 6.1 Fork Correctness Advantages (Unchanged)

All findings from the previous analysis remain valid:

| Fix | Fork Status | Upstream Status | llama.cpp Status |
|-----|-------------|----------------|-----------------|
| Prefill bug (`qwen35.go:136`) | Fixed | **BROKEN** | N/A (uses `add_generation_prompt`) |
| `mropeInterleaved` default | Architecture-based | **BROKEN** (`false`) | Hardcoded per arch |
| `repeat_last_n` wiring | Wired from API | **DEAD** (hardcoded 64) | Wired |
| Penalty sampler (prompt tokens) | Generated-only ring buffer | **Prompt-contaminated** | **Prompt-contaminated** |
| `</think>` closure in Qwen3VL | Always closes | **Broken** | Always closes |
| GGUF converter KV emission | Unconditional | Conditional | N/A |

### 6.2 New Correctness Considerations

**M-RoPE K-shift** (section 1.2): The fork's vendored llama.cpp allows K-shift on M-RoPE models. For the Go runner path (used by qwen3next), this is safely handled by the fallback-to-reprocessing logic. For the C++ llamarunner path (used by older models), this is a potential correctness issue. **Adopt the fix.**

**Qwen 3.5 model family size diversity** (section 1.4): llama.cpp now recognizes 0.8B through 397B Qwen 3.5 variants. The fork should verify that its qwen3next model code handles the full range of layer counts (24, 32, 40, 48, 60, 64) and embedding dimensions (1024, 2560, etc.) without hardcoded assumptions.

**`full_attention_interval` from metadata** (minor): The fork hardcodes `full_attention_interval = 4` at `model.go:513`. llama.cpp reads it from GGUF metadata with a default of 4. For all current Qwen 3.5 models, the value is 4. But future models could use a different interval. The fork should read this from GGUF metadata instead of hardcoding.

### 6.3 JSON Serialization Bugs (Unchanged)

The 3 JSON serialization bugs identified in the previous report (HTML escaping, key ordering, compact separators) remain unfixed in both the fork and upstream Ollama. llama.cpp continues to get this right by executing the Jinja2 template directly.

---

## Part 7: Performance Opportunities — Updated Ranking

### 7.1 Actionable Performance Wins

| Opportunity | Estimated Impact | Effort | Blocked By |
|-------------|-----------------|--------|------------|
| **CUDA async copy + reduced sync** (section 1.1) | 5-15% throughput on CUDA, more with partial offload | Medium (2 files, conflict resolution) | Nothing — pure GGML vendor update |
| **`num_batch 2048`** in Modelfile | ~10% prompt eval speed | Zero (Modelfile change) | Nothing |
| **Multi-modal prompt caching** | 2-10x for repeated-image vision workloads | High (Go server architecture change) | Ollama API design |
| **Lift parallelism restriction** with per-sequence sub-batching | Enables concurrent requests | High (batch construction rewrite) | `deltanet.go` equal-token constraint |
| **Vision flash attention** | Faster vision encoding | Medium (GGML integration) | GGML vendor update |
| **Fused GatedDeltaNet kernel** (from previous report) | Reduced graph compilation overhead | Very High (custom CUDA/Metal kernel) | Kernel development |

### 7.2 NOT Actionable

| Opportunity | Why Not |
|-------------|---------|
| **Speculative decoding** | Fundamentally impossible for hybrid recurrent (section 3) |
| **Continuous batching** | Blocked by `numParallel = 1` and equal-token constraint |
| **Dynamic NTK RoPE** | Static YaRN sufficient for 131K training length |

---

## Part 8: Corrections to Previous Report

### 8.1 Parallelism — `numParallel = 1` IS Enforced (Previous Report Was Correct)

The previous report (`fork_vs_upstream_analysis.md`) stated `sched.go:448` forces `numParallel = 1`. This is **correct**. The parallelism research agent initially reported otherwise (checking only the runner level, not the scheduler), but verification confirms `server/sched.go:450` explicitly sets `numParallel = 1` for all `qwen35`, `qwen35moe`, and `qwen3next` models.

The runner and cache systems architecturally support parallelism, but the scheduler prevents it from being used.

### 8.2 `repeatPenalty` Default — Fork Reverted to 1.0

The previous report noted the fork uses `repeatPenalty: 1.1`. The latest fork commit (`57e3f80f`) reverted this to `1.0`, matching upstream. The commit message: "revert RepeatPenalty default". Both codebases now default to `1.0`.

### 8.3 Vision — Ollama Also Rejects Split Vision GGUFs

The previous report noted this, and it remains true: `llm/server.go:148-152` rejects models with `len(projectors) > 0` when using the Go engine. Third-party GGUFs with separate mmproj files cannot be used with vision in Ollama.

---

## Remaining Action Items for the Fork (Updated)

### From This Research

| Priority | Item | Effort | Section |
|----------|------|--------|---------|
| **P0** | Adopt CUDA async copy + reduced sync from `ggml-cuda.cu` and `ggml-backend.cpp` | Medium | 1.1 |
| **P1** | Adopt M-RoPE `can_shift()` guard in vendored `llama-kv-cache.cpp` | Small (3 lines) | 1.2 |
| **P1** | Cherry-pick thinking level constraint removal from upstream `122c68c` | Small | 2.2 |
| **P2** | Read `full_attention_interval` from GGUF metadata instead of hardcoding 4 | Small | 6.2 |
| **P3** | Adopt Vulkan fp16 FA fix for AMD RDNA2 | Small | 1.8 |

### From Thinking/Non-Thinking Research (Part 9)

| Priority | Item | Effort | Section |
|----------|------|--------|---------|
| **P1** | Fix history `<think>` block rendering — remove `isThinking &&` gate, always include for messages after `lastQueryIndex` | Small | 9.3 |
| **P2** | Adopt thinking level constraint removal from upstream `122c68c` | Small | 9.6 |
| **P3** | Verify `Qwen3VLRenderer` thinking variant needs `emitEmptyThinkOnNoThink` | Research | 9.7 |

### From Previous Report (Still Open)

| Priority | Item | Effort | Section (prev report) |
|----------|------|--------|----------------------|
| **P0** | Fix HTML escaping in `marshalWithSpaces` + `formatToolCallArgument` | Small | 3.1 Bug A |
| **P0** | Fix `required`/`properties` key ordering in `ToolFunctionParameters` | Small | 3.1 Bug B |
| **P1** | Fix compact separators in `formatToolCallArgument` | Small | 3.1 Bug C |
| **P2** | Add dedicated prefill bug fix test in `qwen35_test.go` | Small | 1.1 |

---

## Appendix: File-Level Change Map

### llama.cpp files changed since d969e933 (relevant to Qwen 3.5)

| File | Change | Fork Equivalent |
|------|--------|-----------------|
| `src/models/delta-net-base.cpp` | KDA chunk 16 (no impact on Qwen 3.5) | `model/models/qwen3next/deltanet.go` |
| `src/llama-model.cpp` | Model type detection | N/A (Go handles sizing) |
| `src/llama-kv-cache.cpp` | M-RoPE `can_shift()` guard | `llama/llama.cpp/src/llama-kv-cache.cpp` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Async CPU→CUDA copy | `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` |
| `ggml/src/ggml-backend.cpp` | saaasg sync pattern | `ml/backend/ggml/ggml/src/ggml-backend.cpp` |
| `convert_hf_to_gguf.py` | ForCausalLM registration | `convert/convert_qwen3next.go` (different approach) |
| `tools/server/server-context.cpp` | Multi-modal caching | `runner/ollamarunner/runner.go` (no equivalent) |
| `src/llama-sampler.cpp` | **No changes** | `sample/samplers.go` |
| `common/chat.cpp` | **No changes** | `model/renderers/qwen35.go` |

### Upstream Ollama files changed since 82848a78

| File | Change | Fork Status |
|------|--------|-------------|
| `cmd/config/integrations.go` | qwen3.5 context length, glm-5 | Missing |
| `cmd/config/opencode.go` | Cloud model limit lookup fix | Missing |
| `cmd/config/openclaw.go` | Native API endpoint | Missing |
| `server/routes.go` | Loosen thinking level constraint | **Missing — should adopt** |

### Files verified unchanged (no action needed)

- `src/models/qwen35.cpp` (llama.cpp) — no changes to graph build
- `src/models/qwen35moe.cpp` (llama.cpp) — no changes
- `src/models/qwen3next.cpp` (llama.cpp) — no changes
- `src/llama-sampler.cpp` (llama.cpp) — byte-identical
- `common/chat.cpp` (llama.cpp) — byte-identical
- `tools/mtmd/models/qwen3vl.cpp` (llama.cpp) — no substantive changes

---

## Part 9: Thinking / Non-Thinking Mode — Deep Analysis

### 9.1 The Official Qwen 3.5 Template: One Template, Two Modes

The official Qwen 3.5 Jinja2 template (verified from [`Qwen/Qwen3.5-27B/tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/tokenizer_config.json) and [`Qwen/Qwen3.5-35B-A3B/tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/tokenizer_config.json) — byte-identical `chat_template` fields) uses a single `enable_thinking` parameter that affects **only the generation prompt prefill**:

```jinja2
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

| `enable_thinking` | Generation Prompt Prefill | Model Behavior |
|-------------------|-------------------------|----------------|
| `true` (or undefined — **default**) | `<\|im_start\|>assistant\n<think>\n` | Model generates reasoning inside `<think>...</think>`, then `\n\n`, then content |
| `false` | `<\|im_start\|>assistant\n<think>\n\n</think>\n\n` | Empty think block → model generates content directly |

There are **no `/think` or `/no_think` tokens**, no special token IDs for mode switching, and no per-turn thinking level control. The `<think>` and `</think>` are token IDs 248068-248069 in the vocabulary, marked as `"special": false` — they are regular tokens, not control tokens. The mode is determined entirely by the prefill.

### 9.2 CRITICAL: History Messages Are NOT Gated on `enable_thinking` in the Official Template

The official template renders assistant messages in history (after `last_query_index`) **unconditionally** with `<think>` blocks:

```jinja2
{%- if loop.index0 > ns.last_query_index %}
    {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
{%- else %}
    {{- '<|im_start|>' + message.role + '\n' + content }}
{%- endif %}
```

Note: there is NO `enable_thinking` check here. Whether thinking is enabled or disabled for the current turn, all assistant messages after `last_query_index` include their `<think>` blocks with whatever reasoning content they contain.

### 9.3 BUG: Both Fork and Upstream Gate History `<think>` Blocks on `isThinking`

**Fork** (`model/renderers/qwen35.go:143-147`):
```go
if isThinking && i > lastQueryIndex {
    sb.WriteString(imStartTag + message.Role + "\n<think>\n" + contentReasoning + "\n</think>\n\n" + content)
} else {
    sb.WriteString(imStartTag + message.Role + "\n" + content)
}
```

**Upstream Ollama** (`qwen35.go:143-147`) — **identical code**.

When `think: false` → `isThinking = false` → the `if isThinking && i > lastQueryIndex` check fails → the `else` branch executes → the assistant message is rendered WITHOUT `<think>` blocks, and `contentReasoning` is silently discarded.

**What the official template does:** Regardless of `enable_thinking`, assistant messages after `last_query_index` ALWAYS get `<think>\n{reasoning}\n</think>\n\n{content}`.

**When this bug triggers:** Multi-turn conversations where a user switches from `think: true` to `think: false` between turns. The previous assistant responses had thinking content (in `message.Thinking` or embedded in `message.Content`). With `think: false`, the renderer strips the reasoning from history, producing a prompt the model was never trained to see.

**The `splitQwen35ReasoningContent` function** (`qwen35.go:64-80`) correctly extracts reasoning from either `message.Thinking` or embedded `<think>...</think>` tags in content. But when `isThinking=false`:
1. Line 65: `if isThinking && messageThinking != ""` — skipped (isThinking is false)
2. Lines 69-77: Falls through to content-parsing path — reasoning IS extracted
3. Line 143: `if isThinking && i > lastQueryIndex` — FALSE → reasoning discarded

The reasoning is extracted and then thrown away. The correct fix is to change the condition to match the official template:

```go
// CORRECT (matches official template):
if i > lastQueryIndex {
    sb.WriteString(imStartTag + message.Role + "\n<think>\n" + contentReasoning + "\n</think>\n\n" + content)
} else {
    sb.WriteString(imStartTag + message.Role + "\n" + content)
}
```

And the `splitQwen35ReasoningContent` call should also not gate on `isThinking` for history:

```go
// Line 65 should be unconditional for history messages:
if messageThinking != "" {
    return strings.TrimSpace(messageThinking), content
}
```

**Severity:** Medium. The bug only manifests when a user switches `think` between turns in the same conversation. In practice, most users either always use thinking or always don't. But for agentic frameworks that might toggle thinking per-turn for latency reasons, this is a real distribution shift that could degrade quality.

### 9.4 The `emitEmptyThinkOnNoThink` Field — Correct Implementation

The `Qwen35Renderer` is constructed with `emitEmptyThinkOnNoThink: true` at `renderer.go:60`:

```go
case "qwen3.5":
    return &Qwen35Renderer{isThinking: true, emitEmptyThinkOnNoThink: true, useImgTags: RenderImgTags}
```

This controls the generation prompt behavior at `qwen35.go:185-189`:

```go
if isThinking {
    sb.WriteString("<think>\n")
} else if r.emitEmptyThinkOnNoThink {
    sb.WriteString("<think>\n\n</think>\n\n")
}
```

This correctly matches the official template:
- `think: true` → `<think>\n` (model generates reasoning)
- `think: false` → `<think>\n\n</think>\n\n` (empty think block, model generates content)
- Without `emitEmptyThinkOnNoThink`: nothing emitted — would be WRONG (model starts without any think context)

The `emitEmptyThinkOnNoThink` field exists because some model families (e.g., Qwen3-VL-Instruct, constructed with `isThinking: false` and no `emitEmptyThinkOnNoThink`) were never trained with `<think>` blocks at all. For these models, omitting the think block entirely is correct. Qwen 3.5 was trained with both modes, so it needs the empty think block.

### 9.5 Parser Behavior — Correct for Both Modes

The `Qwen35Parser.Init()` at `parsers/qwen35.go:46-66`:

```go
thinkingEnabled := thinkValue != nil && thinkValue.Bool()
if thinkValue == nil {
    thinkingEnabled = true  // default: thinking enabled
}

if thinkingEnabled && !assistantPrefill {
    p.state = qwen35ParserStateCollectingThinking     // parse <think>...</think>
} else {
    p.state = qwen35ParserStateCollectingContent       // all output is content
}
```

- `think: true` (or nil): Parser starts in `CollectingThinking` state → extracts reasoning from `<think>...</think>` → emits thinking content separately from response content
- `think: false`: Parser starts in `CollectingContent` state → all model output goes to content → no thinking extraction
- `think: false` + renderer prefilled `<think>\n\n</think>\n\n`: Model sees the empty think block and generates content directly. Parser sees only content (no `<think>` in output because it was already closed in the prefill).

This is correct. The parser behavior matches the renderer behavior.

### 9.6 Upstream Commit `122c68c` — Correct Direction, Wrong Execution

**What it does:** Removes the validation that rejected string-valued `think` parameters for non-harmony/gptoss models:

```go
// REMOVED by 122c68c:
if req.Think != nil && req.Think.IsString() && m.Config.Parser != "harmony" {
    c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("think value %q is not supported for this model", ...)})
    return
}
```

**Is it correct?** It's a **pragmatic hack**, not a principled fix.

**What string think values mean for different models:**

| Model Family | `think: "high"/"medium"/"low"` | `think: true/false` |
|-------------|-------------------------------|---------------------|
| **harmony/gptoss** | Meaningful — thinking level/budget control via channels | Boolean on/off |
| **Qwen 3.5** | **Silently maps to `true`** via `ThinkValue.Bool()` (any valid string → true) | Correct boolean control |
| **Qwen3-Coder** | No thinking support — `isThinking: false`, no `emitEmptyThinkOnNoThink` | No effect |
| **DeepSeek 3.1** | **Silently maps to `true`** | Boolean on/off |

The `ThinkValue.Bool()` method at `api/types.go:1122-1137`:

```go
func (t *ThinkValue) Bool() bool {
    switch v := t.Value.(type) {
    case bool:
        return v
    case string:
        return v == "high" || v == "medium" || v == "low"  // any valid string → true
    default:
        return false
    }
}
```

**The problem:** A user sending `think: "low"` to Qwen 3.5 expects reduced/minimal thinking. Instead they get full thinking (identical to `think: true`). The error was more informative than silently accepting — at least it told the user the value wasn't supported.

**The correct fix would be:**
1. Keep the validation for models where string values are meaningless
2. OR: Add a `SupportsThinkingLevels() bool` method to the renderer interface, and only accept strings when the model supports levels
3. OR: Accept strings universally but document that they map to boolean for non-level-aware models

**Should the fork adopt this?** The fork should adopt it with the understanding that it's a hack. The alternative (keeping the error) is worse for client compatibility. The best long-term fix is to make the API more explicit about what each model supports.

### 9.7 Qwen3-Coder vs Qwen 3.5: Different Thinking Architectures

**Qwen 3.5** (fork's `Qwen35Renderer`):
- `isThinking: true` by default
- `emitEmptyThinkOnNoThink: true` — prefills empty `<think>` block when thinking disabled
- Thinking blocks in history are ALWAYS present (per official template — but Ollama gates on `isThinking`, which is a bug)
- `enable_thinking` parameter controls the prefill

**Qwen3-Coder** (fork's `Qwen3CoderRenderer`):
- `isThinking: false` by default (constructed with zero-value struct at `renderer.go:51`)
- `emitEmptyThinkOnNoThink: false`
- Thinking blocks in history are conditional on `message.Thinking != ""` AND `isThinking` AND `i > lastQueryIndex` — this is actually correct for Qwen3-Coder, whose official template only includes think blocks when `reasoning_content` is non-empty
- No `enable_thinking` in the official Qwen3-Coder template — thinking is always optional

**Qwen3-VL-Thinking** (fork's `Qwen3VLRenderer`):
- `isThinking: true`
- `emitEmptyThinkOnNoThink: false` (NOT set!) — this means when `think: false`, NO think block is prefilled
- This is potentially wrong if the model was trained with the same `enable_thinking=false` → empty think block pattern

**Qwen3-VL-Instruct** (fork's `Qwen3VLRenderer`):
- `isThinking: false` — no thinking support
- No think blocks at all — correct for non-thinking variant

### 9.8 The `Qwen3CoderRenderer` Thinking Asymmetry with the Fork

The fork added thinking support to `Qwen3CoderRenderer` (in commit `fbae6976`) that upstream lacks:

**Fork's `Qwen3CoderRenderer`** (`qwen3coder.go:164`):
```go
if isThinking && message.Thinking != "" && i > lastQueryIndex {
    sb.WriteString("<think>\n" + strings.Trim(message.Thinking, "\n") + "\n</think>\n\n")
}
```

**Upstream's `Qwen3CoderRenderer`** — no thinking code at all. `isThinking` field exists but is never set to `true` by any constructor.

This means the fork supports thinking for `qwen3-coder` models if the `think: true` API parameter is passed (which overrides `isThinking` at line 143: `if think != nil { isThinking = think.Bool() }`). Upstream silently ignores `think: true` for qwen3-coder.

Since Qwen3-Coder models don't have an official `enable_thinking` mechanism in their Jinja2 template, the fork's thinking support for qwen3-coder is an extension beyond the training data. It could work well (the model has seen `<think>` tokens) or badly (the model wasn't explicitly trained to use them in the Coder template context).

### 9.9 Summary of Thinking/Non-Thinking Issues

| Issue | Affects | Severity | Status |
|-------|---------|----------|--------|
| **History `<think>` blocks gated on `isThinking`** — official template includes them unconditionally | Both fork and upstream | **Medium** — distribution shift when switching think modes between turns | **BUG — OPEN in both** |
| **`emitEmptyThinkOnNoThink` correctly implements prefill** | Fork and upstream | N/A | **Correct** |
| **Parser correctly handles both modes** | Fork and upstream | N/A | **Correct** |
| **`think: "low"` silently treated as `think: true`** | All non-harmony models | **Low** — misleading but not harmful | **By design** (after `122c68c`) |
| **Qwen3-VL-Thinking missing `emitEmptyThinkOnNoThink`** | Fork and upstream | **Low** — think=false produces no prefill at all | **Potential bug** — needs verification against official template |
| **Fork's Qwen3CoderRenderer has thinking support upstream lacks** | Fork only | **Informational** — extension beyond training data | Fork-specific feature |

### 9.10 Updated Action Items

| Priority | Item | Effort |
|----------|------|--------|
| **P1** | Fix history `<think>` block rendering — remove `isThinking &&` gate at `qwen35.go:143`, always include `<think>` blocks for messages after `lastQueryIndex` | Small (1-line change + update `splitQwen35ReasoningContent` to not gate on `isThinking` for history) |
| **P2** | Adopt thinking level constraint removal from `122c68c` | Small |
| **P3** | Verify whether `Qwen3VLRenderer` needs `emitEmptyThinkOnNoThink: true` for the thinking variant | Research |
