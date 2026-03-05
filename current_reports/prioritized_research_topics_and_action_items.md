# Prioritized Research Topics and Action Items for the BigBIueWhale/ollama Fork (Qwen 3.5 27B Focus)

**Date:** 2026-03-06
**Fork:** `BigBIueWhale/ollama` @ commit `57e3f80f` (8 commits atop the Ollama `v0.17.4` merge base `cc90a035`)
**Model:** Qwen 3.5 27B (Alibaba Qwen), running via Unsloth Dynamic 2.0 `UD-Q4_K_XL` GGUF quantization (17.6 GB, 851 tensors, text-only)
**Inference engine:** Ollama (Go-based inference engine that vendors the GGML tensor math library from llama.cpp but reimplements everything else — chat template rendering, sampling, model runner — in Go)

---

## Prioritized Action Items

| Priority | Item | Effort | Potential Impact |
|----------|------|--------|-----------------|
| **High** | JSON serialization mismatch fix (3 sub-bugs: HTML escaping, key ordering, compact separators) — see Section A.1 | Small | Improved tool calling accuracy for all tool-using models |
| **High** | CUDA asynchronous copy and reduced synchronization vendor update from llama.cpp — see Section B.3 | Medium | 5-15% throughput improvement on NVIDIA GPU hardware |
| **High** | History `<think>` block rendering fix (remove incorrect `isThinking` gate) — see Section A.2 | Tiny (1 line of code) | Correct multi-turn thinking/non-thinking conversations |
| **Medium** | `num_batch 2048` Modelfile configuration change — see Section B.4 | Zero (configuration-only, no code change) | Approximately 10% faster prompt evaluation speed |
| **Medium** | Multi-Resolution Rotary Position Embedding (M-RoPE) `can_shift()` guard — see Section C.5 | Tiny (3 lines of code) | Defensive correctness fix preventing corrupt positional encoding |
| **Medium** | Thinking level constraint removal (cherry-pick upstream Ollama commit `122c68c`) — see Section E.11 | Small | Better compatibility with API clients that send string-valued thinking levels |
| **Medium** | Parallelism investigation (lifting the `numParallel = 1` scheduler restriction) — see Section D.9 | High | Enables serving concurrent requests instead of queuing them |
| **Low** | Qwen 3.5 model family size diversity verification — see Section C.6 | Small | Ensures all Qwen 3.5 model sizes (0.8B through 397B) work correctly |
| **Low** | Read `full_attention_interval` from GGUF metadata instead of hardcoding — see Section C.7 | Tiny | Future-proofing against models that use a different attention interval |
| **Low** | Qwen3-VL-Thinking renderer missing `emitEmptyThinkOnNoThink` field — see Section E.12 | Research required | Correct `think: false` behavior for Qwen3 Vision-Language Thinking models |
| **Low** | Vulkan flash attention fix for AMD Radeon RDNA2 GPUs on Microsoft Windows — see Section G.14 | Small | Fixes broken flash attention for AMD GPU users on older Windows drivers |
| **Informational** | Speculative decoding confirmed impossible for hybrid recurrent architectures — see Section D.8 | Not applicable | Confirmed dead end — do not pursue |
| **Informational** | Multi-modal prompt caching (vision image re-encoding avoidance) — see Section D.10 | High | Future vision performance improvement (2-10x for repeated-image workflows) |
| **Informational** | Enabling vision capabilities with Unsloth third-party GGUF files — see Section F.13 | Very High | Third-party GGUF vision capability (currently only official Ollama unified GGUFs support vision) |

---

## Section A: Bugs In the Fork That Are Still Open

### A.1 — JSON Serialization Mismatches in Tool Definitions and Tool Call Arguments (3 Sub-Bugs)

The fork (and upstream Ollama) serialize tool definitions into the model's prompt differently from how the Qwen 3.5 model was originally trained by Alibaba. The model was trained using HuggingFace Transformers' `apply_chat_template` function, which uses a custom `tojson` Jinja2 filter override (defined at `chat_template_utils.py:463-466` in HuggingFace Transformers version 5.2.0). Ollama reimplements this serialization in Go using `json.Marshal`, which produces different output in three ways:

**Sub-bug A — HTML character escaping (high impact for coding and API tools):**
Go's standard library `json.Marshal` function automatically converts `<`, `>`, and `&` characters into Unicode escape sequences (`\u003c`, `\u003e`, `\u0026`). This is a deliberate Go design decision intended for safely embedding JSON inside HTML `<script>` tags — a concern that has zero relevance to Ollama, where JSON is serialized into a chat template prompt fed to a neural network. HuggingFace Transformers' `tojson` override outputs literal `<`, `>`, `&` characters. The result: a tool description like `"Returns temperature in <fahrenheit> & <celsius>"` becomes completely different tokens when HTML-escaped (`\u003cfahrenheit\u003e` is 6 bytes per character instead of 1). The fix is straightforward: use Go's `json.NewEncoder` with `SetEscapeHTML(false)`. Ollama's own `LFM2Renderer` (for Liquid Foundation Model 2) already does this correctly at `model/renderers/lfm2.go:55-56`.

**Sub-bug B — Key ordering at the `ToolFunctionParameters` level (medium impact):**
Go's `json.Marshal` outputs struct fields in the order they are declared in the Go source code. In Ollama's `api/types.go:486-491`, the `Required` field is declared before the `Properties` field, so the serialized JSON always outputs `"required"` before `"properties"`. HuggingFace Transformers' `get_json_schema()` function and the OpenAI function calling convention both output `"properties"` before `"required"`. This is a positional encoding mismatch on every tool-using prompt — the model's attention patterns learned one key layout and sees another. The fix: swap the `Required` and `Properties` field declarations in `api/types.go`.

**Sub-bug C — Compact separators in `formatToolCallArgument` (medium impact):**
The `formatToolCallArgument` function (at `qwen3coder.go:195-217`, shared by all renderers for rendering nested map/list values inside XML `<parameter>` tags) uses raw `json.Marshal`, which produces compact JSON without spaces (`{"key":"value"}`). HuggingFace Transformers' `tojson` uses Python's default separators (`(', ', ': ')`), producing spaced JSON (`{"key": "value"}`). The model was trained on the spaced form.

**Why these bugs matter:** Every time the model uses tools, it sees a prompt format that differs from its training data. This creates a "distribution shift" — the model still works, but tool calling reliability is degraded compared to what it could be. The canonical C++ inference engine llama.cpp gets all three right because it executes the official Jinja2 template directly, producing byte-identical output to HuggingFace Transformers.

### A.2 — History `<think>` Blocks Incorrectly Gated on the `isThinking` Boolean

Both the fork and upstream Ollama gate the rendering of `<think>` blocks in assistant message history on the `isThinking` boolean (controlled by the API's `think: true/false` parameter). The relevant code is at `qwen35.go:143`:

```go
if isThinking && i > lastQueryIndex {
    // render with <think> block
} else {
    // render without <think> block — reasoning content silently discarded
}
```

The official Qwen 3.5 Jinja2 template (verified from both `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.5-35B-A3B` on HuggingFace — byte-identical `chat_template` fields) does NOT gate history `<think>` blocks on `enable_thinking`. All assistant messages after `last_query_index` ALWAYS include their `<think>` blocks, regardless of whether thinking is enabled or disabled for the current generation turn.

**When this bug triggers:** Multi-turn conversations where a user (or an agentic framework) switches from `think: true` to `think: false` between turns. Previous assistant responses had thinking content (stored in `message.Thinking`). With `think: false`, the renderer extracts the reasoning content via `splitQwen35ReasoningContent` and then throws it away because the `isThinking && ...` check fails. The model sees its own previous responses with reasoning stripped out — a prompt format it was never trained on.

**The fix:** Remove `isThinking &&` from the condition at `qwen35.go:143`, so that `<think>` blocks are always included for messages after `lastQueryIndex`, matching the official template.

---

## Section B: Performance Improvements Available From Upstream llama.cpp

### B.3 — CUDA Asynchronous Copy and Reduced CPU-GPU Synchronization

The llama.cpp project (the canonical C++ inference engine for GGUF models, maintained by ggml-org) gained a major CUDA performance optimization in commit `2cd20b7`. Two files changed:

**`ggml-cuda.cu`:** The `ggml_backend_cuda_cpy_tensor_async()` function now supports asynchronous CPU-to-CUDA memory copies (using `cudaMemcpyAsync` with `cudaMemcpyHostToDevice`) in addition to the existing CUDA-to-CUDA async copies. The fork's vendored version (at `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu:2923`) requires both source and destination to be CUDA backends, which forces a synchronous CPU thread block on every CPU-to-GPU tensor copy.

**`ggml-backend.cpp`:** The scheduler's `compute_splits()` function was restructured from a per-input synchronization pattern (O(inputs) sync points per compute split) to a "sync-async-async-async-sync-graph" (saaasg) pattern (O(1) sync points per compute split). The old pattern (in the fork at `ggml-backend.cpp:1603`) calls `ggml_backend_synchronize()` inside the copy loop for each input tensor that fails async copy. The new pattern synchronizes once before input copies begin and once after graph compute, using event-based signaling between backends.

**This is NOT a Blackwell/next-generation NVIDIA GPU specific optimization.** There are zero references to sm_100, Blackwell, or Ada Lovelace in the commit. This is a general CUDA improvement benefiting all NVIDIA GPUs, from consumer GeForce cards to datacenter hardware.

**Why it matters:** This is the single most impactful performance change llama.cpp has made since the fork's vendored GGML version (`v0.17.4`). The improvement is most noticeable with partial GPU offload (where the model is split across GPU VRAM and system RAM, requiring many input tensors to be copied per compute split) and multi-GPU setups. For a fully GPU-loaded Qwen 3.5 27B at Q4_K_XL (17.6 GB), the improvement is smaller but still meaningful — reducing CPU thread blocking on every forward pass.

**How to adopt:** Update the fork's vendored GGML files at `ml/backend/ggml/ggml/src/ggml-backend.cpp` and `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu`. This is a focused change to two files in the GGML layer — it does not touch model code or the Go runner. However, the scheduler restructuring is a moderately complex diff because surrounding code also changed (expert-level tensor copying was reworked). Cherry-picking cleanly may require manual conflict resolution.

### B.4 — Increase `num_batch` to 2048 in the Modelfile (Zero Code Changes Required)

Ollama defaults to a batch size of 512 (set at `api/types.go:1074`). The llama.cpp project defaults to 2048. For long prompts (the Qwen 3.5 27B model supports up to 131,072 tokens of context), a larger batch size reduces the number of forward passes required during prompt evaluation. The GatedDeltaNet recurrent layer implementation uses a chunk size of 64 (set at `deltanet.go:12`), so batch sizes should be multiples of 64 for optimal padding alignment.

**Why it matters:** Adding `PARAMETER num_batch 2048` to the custom Modelfile at `/home/user/Desktop/vibe_web_terminal/ollama-models/qwen3.5-custom.Modelfile` could yield approximately 10% faster prompt evaluation with zero code changes. The tradeoff is higher peak memory usage during prompt evaluation (larger batches require more intermediate tensor memory). Testing is recommended to confirm the model still fits in GPU VRAM with the larger batch size.

---

## Section C: Correctness Fixes Available From Upstream llama.cpp

### C.5 — Multi-Resolution Rotary Position Embedding (M-RoPE) `can_shift()` Guard

The llama.cpp project added a correctness fix in commit `99bd67c` that prevents context window sliding (K-shift) on models that use Multi-Resolution Rotary Position Embedding (M-RoPE). K-shift is a memory management technique where the KV cache "slides" to reuse space when the context window fills up. It requires a single scalar position value per token. M-RoPE models (including Qwen 3.5 dense and MoE, which use Interleaved M-RoPE / IMROPE) have 4 position values per token (temporal, height, width, extra). Attempting K-shift on M-RoPE models would corrupt the positional encoding.

The fork's vendored `llama-kv-cache.cpp` (at `llama/llama.cpp/src/llama-kv-cache.cpp:968-970`) unconditionally returns `true` for `can_shift()`. The fix adds a check: if `hparams.n_pos_per_embd() > 1`, return `false`.

**Current risk level for Qwen 3.5 specifically:** Low. Qwen 3.5 models use the Go-based `ollamarunner` (because `qwen3next` is in the `OllamaEngineRequired()` list), not the C++ `llamarunner`. The Go runner has its own cache management that gracefully degrades — it falls back to full reprocessing instead of attempting a corrupt shift (see `ollamarunner/cache.go:298-314`). However, if other M-RoPE models (such as Qwen2VL or Qwen3VL) get routed through the C++ `llamarunner` path, they would be affected. The fix is a 3-line addition and adds a valuable safety net.

### C.6 — Qwen 3.5 Model Family Size Diversity Verification

The llama.cpp project (commit `872646b`) updated its Qwen 3.5 model type detection to recognize the full range of model sizes that Alibaba has released:

- **Dense variants:** 0.8B (24 layers, 1024 embedding dimension), 2B, 4B (32 layers, 2560 embedding dimension), 9B (32 layers), 27B (64 layers)
- **Mixture-of-Experts (MoE) variants:** 35B-A3B (40 layers, 3 billion active parameters per token), 122B-A10B, 397B-A17B

The fork's `qwen3next` model code handles the 27B dense model correctly. It should be verified that the layer interval formula (`(i+1) % 4 != 0` for determining which layers are recurrent versus full attention) and embedding dimension handling work correctly across all these sizes — particularly the smaller dense models (0.8B with 24 layers, 4B with 32 layers) which have very different shapes from the 27B model.

**Why it matters:** If someone loads a Qwen 3.5 0.8B or 4B GGUF into the fork, incorrect layer classification would cause the wrong layer types to be used (recurrent where attention is expected, or vice versa), producing silent output quality degradation.

### C.7 — Read `full_attention_interval` From GGUF Metadata Instead of Hardcoding

The fork hardcodes `full_attention_interval = 4` at `model.go:513` (meaning every 4th layer is a full attention layer, and the other 3 are GatedDeltaNet recurrent layers). The llama.cpp project reads this value from GGUF metadata with a default of 4 via `ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false)`.

**Why it matters:** All current Qwen 3.5 models use an interval of 4, so the hardcoded value is correct today. But if Alibaba changes the architecture pattern in future Qwen 3.5 model variants (or if other `qwen3next`-architecture models use a different interval), the hardcoded value would silently produce incorrect layer classification. The fix is trivial: change `int(c.Uint("full_attention_interval", 4))` to read from metadata with a default of 4 (which is what the fork already does syntactically — but verifying this is actually reading from GGUF metadata rather than just using the default is worth confirming).

---

## Section D: Architectural Limitations Worth Understanding

### D.8 — Speculative Decoding Confirmed Impossible for Hybrid Recurrent Architectures

Speculative decoding is a technique where a small "draft" model generates N candidate tokens cheaply, then the full model verifies them all in a single forward pass. Correct tokens are accepted; wrong ones are rejected and regenerated. For quantized models on consumer hardware (where generation is memory-bandwidth-bound — moving 17.6 GB of weights through memory for every single token), speculative decoding typically provides a 2-4x speedup.

The reports confirm this is **fundamentally incompatible** with Qwen 3.5's hybrid recurrent architecture (GatedDeltaNet + standard attention layers). The reasons:

1. **llama.cpp explicitly blocks it:** The compatibility test at `common/speculative.cpp:801-835` checks whether partial tail removal works on the model's memory. For hybrid models, the recurrent sub-cache at `llama-memory-recurrent.cpp:142-180` explicitly rejects partial tail removal — the recurrent state cannot be rolled back.

2. **The mathematical blocker:** Recurrent hidden state is destructively updated token by token: `S_{t+1} = alpha * S_t + beta_t^T * v_t`. If a draft token at position `t` is rejected, the state `S_{t+1}` has already been computed and stored. Rolling back to `S_t` requires either prohibitively expensive per-token checkpointing (for Qwen 3.5 27B with 48 recurrent layers, each checkpoint is hundreds of megabytes) or recomputation from the last checkpoint (the fork's checkpoint system saves state every 1,664 tokens — far too coarse for speculative decoding which needs token-level rollback).

3. **All alternatives are impractical near-term:**
   - Medusa-style parallel generation heads require model retraining
   - Lookahead decoding still requires state rollback
   - Speculating only on the 16 attention layers (25% of the model) while running the 48 recurrent layers (75%) normally would provide severely limited benefit
   - Fine-grained checkpointing would trade enormous memory for throughput

**Bottom line:** This is the single biggest potential speedup that has been ruled out. Token generation speed for Qwen 3.5 is fundamentally limited to one token at a time through the full model. Do not invest time investigating speculative decoding for this architecture.

### D.9 — Parallelism Restriction: `numParallel = 1` Enforced at the Scheduler Level

Both the fork and upstream Ollama force `numParallel = 1` for all hybrid/recurrent model architectures at `server/sched.go:448-453`. This means only one request can be processed at a time — concurrent requests are queued.

**The restriction is conservative, not fundamental.** The Go runner (`ollamarunner`) and the cache system (`kvcache/`) architecturally support multiple parallel sequences. The `Recurrent` cache maintains a `slotForSeq` map allocating separate slots with independent convolution and recurrent states per sequence. The runner only restricts parallelism when `Config().Cache == nil`, which is not the case for `qwen3next` (it uses `NewHybridCache`).

**The real blocker** is `deltanet.go:80-87`, which requires all sequences in a batch to have the same token count (`ErrUnsupportedBatchLayout`). This is a structural requirement of the chunked linear attention computation — chunk boundaries must align across sequences. In practice, different requests have different prompt lengths and can't be batched together. However, during generation (where each sequence produces 1 token per step), they could theoretically be batched.

**llama.cpp demonstrates this works** using `split_equal(n_ubatch, true)` in `llama_memory_hybrid` — each sequence is processed in its own sub-batch. The sequences run sequentially within a step but can be interleaved across steps, sharing the loaded model weights.

**Why it matters:** If you have multiple users or an agentic framework making concurrent requests to the Ollama instance, they queue instead of running concurrently. Lifting this restriction would require rewriting the batch construction logic in the Go runner to use per-sequence sub-batching, which is a non-trivial effort.

### D.10 — Multi-Modal Prompt Caching for Vision Workloads

The llama.cpp project recently added caching of multi-modal (vision) prompts in commits `f20469d` and `d7d826b`. Previously, when a cached prompt contained images, the cache was invalidated and the prompt was reprocessed from text tokens only. Now, llama.cpp's server preserves multi-modal data across cache entries, supports non-contiguous token/position mappings from vision embeddings, and decouples position tracking from token count.

Ollama's Go server has its own cache system but has no equivalent multi-modal caching. If a user sends repeated requests with the same image (a common pattern in agentic vision workflows — for example, repeatedly asking questions about a screenshot), the first request processes the image through the 27-layer vision encoder, but every subsequent request must re-encode the same image from scratch.

**Why it matters:** For vision-intensive workloads, this is a significant performance gap between Ollama and llama.cpp — potentially 2-10x slower for repeated-image scenarios. Implementing multi-modal prompt caching in Ollama's Go runner would be a significant architectural change (it touches the cache system, the runner's prompt processing, and the vision encoding pipeline), but it would be a major performance win for users who work with vision models.

---

## Section E: Upstream Ollama Changes to Sync

### E.11 — Thinking Level Constraint Removal (Upstream Ollama Commit `122c68c`)

Upstream Ollama removed validation in `server/routes.go` that rejected string-valued `think` parameters (such as `"think": "high"`, `"think": "medium"`, `"think": "low"`) for models other than harmony/gptoss. The fork still has this restrictive validation, which means API clients that send string thinking levels receive an HTTP 400 error.

**The nuance:** For Qwen 3.5, string thinking levels are silently mapped to boolean `true` via `ThinkValue.Bool()` (at `api/types.go:1122-1137`), because the method returns `true` for any valid string (`"high"`, `"medium"`, or `"low"`). A user sending `think: "low"` expecting reduced/minimal thinking will instead get full thinking, identical to `think: true`. The error message was arguably more informative than silently accepting the value — but rejecting the request entirely is worse for client compatibility than accepting it with boolean mapping.

**Why it matters:** Some API clients and agentic frameworks send string-valued thinking levels. Removing the validation is a small cherry-pick that improves compatibility, with the understanding that string values are silently treated as `true` for all non-harmony models.

### E.12 — Qwen3 Vision-Language Thinking Renderer Missing `emitEmptyThinkOnNoThink` Field

The `Qwen3VLRenderer` for Qwen3-VL-Thinking models is constructed with `isThinking: true` but WITHOUT the `emitEmptyThinkOnNoThink: true` field. This means when `think: false` is passed via the API, no empty think block (`<think>\n\n</think>\n\n`) is prefilled — the model starts generating without any think context at all.

For comparison, the `Qwen35Renderer` IS constructed with `emitEmptyThinkOnNoThink: true`, correctly matching the official Qwen 3.5 template where `enable_thinking=false` produces an empty think block (not the absence of a think block).

**Why it matters:** If the Qwen3-VL-Thinking model was trained the same way as Qwen 3.5 (where `think: false` should produce an empty `<think>\n\n</think>\n\n` block rather than no block at all), then the missing field means the model sees a prompt format it wasn't trained on when thinking is disabled. Verifying this requires checking the official Qwen3-VL-Thinking Jinja2 template on HuggingFace to determine whether the `enable_thinking=false` path produces an empty think block or no think block.

---

## Section F: Vision With Third-Party GGUF Files

### F.13 — Enabling Vision Capabilities With Unsloth Third-Party GGUF Files

The fork's current setup uses the text-only Unsloth Dynamic 2.0 `UD-Q4_K_XL` GGUF (851 tensors, no vision weights, no `vision.block_count` metadata). Vision works correctly with the official `ollama.com/library/qwen3.5:27b` unified GGUF (1,307 tensors: 851 text + 456 vision, with `vision.block_count = 27`), but not with third-party GGUFs from Unsloth, bartowski, or the llama.cpp converter that split text and vision into separate files (main GGUF + separate mmproj GGUF).

Ollama's Go engine explicitly rejects split vision models at `llm/server.go:148-152`. When pulling from HuggingFace (e.g., `FROM hf.co/unsloth/Qwen3.5-27B-GGUF:Q4_K_XL`), Ollama auto-downloads the mmproj file alongside the main GGUF, classifies it as a projector, and then the Go engine rejects the entire model load because `qwen3next` is in the `OllamaEngineRequired()` list with no fallback to the C++ runner.

**Two potential approaches exist, both requiring significant research:**

1. **Offline GGUF merge tool:** Write a tool that merges the Unsloth text GGUF and mmproj GGUF into a single unified GGUF matching Ollama's expected format. This avoids engine changes but requires understanding tensor naming conventions (Ollama uses `v.blk.{n}.attn_qkv.weight`, llama.cpp's converter may use different names), metadata compatibility, and mixed-precision handling (mmproj is BF16/F16/F32 while the text model is Q4_K_XL).

2. **Go engine multi-GGUF loading:** Modify Ollama's model loading pipeline (at `ml/backend/ggml/ggml.go`) to accept multiple GGUF files, with tensor name mapping between the two files and correct buffer allocation across devices and quantization types.

**Why it matters:** If you want vision capabilities with the Unsloth quantization quality (which uses dynamic mixed precision — sensitive layers get Q6_K or Q8_0, less sensitive layers get Q4_K), there is currently no path. The Qwen 3.5 27B vision encoder is substantial — 27 layers, 456 tensors, 1,152 hidden size, 16 attention heads, patch size 16 — making this a non-trivial integration regardless of approach.

---

## Section G: GGML Vendor Debt (Accumulated Divergence From Upstream llama.cpp)

### G.14 — Vulkan Flash Attention Fix for AMD Radeon RDNA2 GPUs on Microsoft Windows

The llama.cpp project (commit `723c710`) fixed broken `subgroupShuffleXor` with `f16vec4` operands on AMD Radeon RDNA2 and older GPUs using AMD's proprietary Windows driver. The workaround casts `f16vec4` to `vec4` (float32) for the shuffle operation, then casts back. The fork's vendored Vulkan shaders lack this fix.

**Why it matters:** Only relevant for users running on AMD Radeon GPUs with RDNA2 architecture (Radeon RX 6000 series) or older, on Microsoft Windows with the proprietary AMD driver. Low priority for NVIDIA GPU or Linux setups.

### G.15 — Full GGML Vendor Update (Growing Divergence)

The fork vendors the GGML tensor math library from the Ollama `v0.17.4` tag. Beyond the CUDA asynchronous copy optimization (Section B.3) and the M-RoPE `can_shift()` guard (Section C.5), additional smaller changes have accumulated in upstream llama.cpp:

- **AMD Instinct MI300X (gfx942 CDNA3) tensor core support** for flash attention using MFMA intrinsics: +7% to +39% prompt evaluation speed (commit `ecbcb7e`) — datacenter AMD GPU only
- **MXFP4 (microscaled FP4) CPU repack kernels** for x86 and ARM (commit `d903f30`) — only relevant for MXFP4-specific GGUFs on CPU, not the Q4_K_XL format
- **Vulkan partial offload improvements** (commit `3191462`) — AMD Vulkan GPU users
- **Intel AMX (Advanced Matrix Extensions) batched matmul fix** (commit `4e76d24`) — Intel server CPU users
- **ARM aarch64 kleidiai SME fp16 for q4_0** (commit `137435f`) — ARM CPU performance
- **`ggml_is_contiguous_n` contiguity check fix** when tensor dimension equals 1 (commit `7f5ee54`) — edge case correctness
- **Vulkan memory overlap fusion guard** (commit `3769fe6`) — prevents incorrect operation fusion when memory overlaps

None of these are individually critical for the fork's current use case (NVIDIA GPU, Qwen 3.5 27B Q4_K_XL), but they represent growing divergence from upstream. Each Ollama release makes catching up harder.

**Recommended approach:**
1. **Now:** Targeted cherry-pick of the CUDA asynchronous copy change (Section B.3) — highest return on investment
2. **Now:** Targeted cherry-pick of the M-RoPE `can_shift()` guard (Section C.5) — straightforward 3-line addition
3. **Later:** Full GGML vendor update when Ollama v0.18 or v0.19 is released — picks up everything but requires comprehensive testing
