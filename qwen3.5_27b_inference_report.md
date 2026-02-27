# Qwen 3.5 27B Dense Language Model — Inference Library Support Report

**Date:** February 27, 2026 (updated from February 26)
**Target Hardware:** NVIDIA RTX 5090 graphics card (32 GB VRAM — Video Random Access Memory)
**Model:** Qwen 3.5 27B dense language model (all 27 billion parameters active every forward pass)
**Inference Libraries Evaluated:** vLLM inference server, Ollama inference framework (+ llama.cpp C++ backend)

### Versions Pinned in This Report

| Component | Version | Commit | Link |
|-----------|---------|--------|------|
| Ollama inference framework (original analysis) | v0.17.1 | `9bf4196` | [ollama/ollama@9bf4196](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a) |
| Ollama inference framework (re-verified) | master (post-v0.17.4) | `79917cf` | [ollama/ollama@79917cf](https://github.com/ollama/ollama/tree/79917cf80bf74538a4ae694e6b61adb908b0f8df) |
| llama.cpp C++ inference library (Ollama-pinned) | — | `ec98e2002` | [ggml-org/llama.cpp@ec98e2002](https://github.com/ggml-org/llama.cpp/tree/ec98e2002) + [34 Ollama patches](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/llama/patches) |
| llama.cpp C++ inference library (upstream) | — | `723c710` | [ggml-org/llama.cpp@723c710](https://github.com/ggml-org/llama.cpp/tree/723c71064da0908c19683f8c344715fbf6d986fd) |
| vLLM inference server | nightly (post-v0.16.1rc0) | `a1f53ad` | [vllm-project/vllm@a1f53ad](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329) |

---

## Table of Contents

- [Bug Report Summary](#bug-report-summary) — **start here** for the three verified, bug-report-worthy findings

1. [Model Overview](#1-model-overview)
2. [VRAM Fit and Quantization Options](#2-vram-fit-and-quantization-options)
3. [Recommended Inference Parameters](#3-recommended-inference-parameters)
4. [vLLM Support Status](#4-vllm-support-status)
5. [Ollama Support Status](#5-ollama-support-status)
6. [Ollama Deep Dive: Parameter Flow Validation](#6-ollama-deep-dive-parameter-flow-validation)
7. [Ollama Critical Bug: Missing Penalty Sampling](#7-ollama-critical-bug-missing-penalty-sampling)
8. [Ollama Critical Bug: Tool Calling Format Mismatch](#8-ollama-critical-bug-tool-calling-format-mismatch)
9. [Ollama Registry Model Validation: `qwen3.5:27b-q4_K_M`](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)
10. [Ollama Context Length](#10-ollama-context-length)
11. [Ollama Per-Use-Case Parameter Settings](#11-ollama-per-use-case-parameter-settings)
12. [Tool Calling Verdict](#12-tool-calling-verdict)
13. [llama.cpp Backend Validation](#13-llamacpp-backend-validation)
14. [Pinned llama.cpp Version in Ollama](#14-pinned-llamacpp-version-in-ollama)
15. [Summary: Is Everything Truly Correct?](#15-summary-is-everything-truly-correct)
16. [Cloned Repositories](#16-cloned-repositories)
17. [Sources](#17-sources)

---

## Bug Report Summary

The following three bugs in the Ollama inference framework were originally identified against v0.17.1 (commit [`9bf4196`](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a)) and **verified against source code, the Ollama model registry at `registry.ollama.ai`, and the HuggingFace model repository ground truth**. All three are unfixable by end users via API parameters or configuration — they require changes to Ollama's source code or registry model configuration by the Ollama team.

### Re-Verification Against Ollama Master (February 27, 2026)

All three bugs were re-verified against Ollama master at commit [`79917cf`](https://github.com/ollama/ollama/tree/79917cf80bf74538a4ae694e6b61adb908b0f8df) (February 26, 2026, one commit ahead of v0.17.4). Stable releases checked: v0.17.1 through v0.17.4.

| Bug | v0.17.1 | v0.17.2 | v0.17.3 | v0.17.4 | master (`79917cf`) |
|-----|---------|---------|---------|---------|-------------------|
| Bug 1: Missing penalty sampling | Present | Present | Present | Present | **Present** |
| Bug 2: Tool calling format mismatch | Present | Present | Present | Present | **Present** |
| Bug 3: Unclosed `</think>` tag (renderer) | Present | Present | Present | Present | **Present** |
| Bug 3 (parser-side mitigation) | — | — | **Fixed** | Fixed | Fixed |

**Bug 3 partial fix detail:** Commit [`d98dda4`](https://github.com/ollama/ollama/commit/d98dda4676d44a3882fd38492cc00db257f35974) ("model: fix qwen3 tool calling in thinking", PR [#14477](https://github.com/ollama/ollama/pull/14477), merged February 26, 2026) added parser-side handling so that `Qwen3Parser` and `Qwen3VLParser` now detect `<tool_call>` while still in thinking-collection state, treating it as the end of thinking and transitioning to tool-call parsing. This fix entered the **v0.17.3** stable release. However, the **renderer side** of the bug — the `Qwen3VLRenderer` at [`qwen3vl.go:98-99`](https://github.com/ollama/ollama/blob/79917cf80bf74538a4ae694e6b61adb908b0f8df/model/renderers/qwen3vl.go#L98-L99) still gating `</think>` emission on `content != ""` — remains unfixed. Multi-turn prompts sent to the model still contain unclosed `<think>` tags when an assistant turn has thinking + tool calls + empty content. The tool-call thinking tests in `qwen3vl_thinking_test.go` (lines 119–323) remain commented out.

### Bug 1: Ollama Go-Based Runner Silently Ignores All Penalty Sampling Parameters

The Ollama Go-based runner (`ollamarunner`) — the only runner available for Qwen 3.5, which is forced onto it via the `OllamaEngineRequired()` list — has **no implementation of penalty sampling whatsoever**. The `repeat_penalty`, `presence_penalty`, and `frequency_penalty` parameters are accepted by the Ollama API without error or warning, stored in the `Options` struct, and then silently discarded when the sampler is constructed. The Qwen 3.5 27B model card explicitly recommends `presence_penalty=1.5` to prevent repetition loops during extended thinking sequences. Setting this via the API has zero effect.

- **Evidence:** [Section 7](#7-ollama-critical-bug-missing-penalty-sampling) and [Section 6](#6-ollama-deep-dive-parameter-flow-validation)
- **Key source files:** [`sample/samplers.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/sample/samplers.go) (no penalty fields in `Sampler` struct or `NewSampler` function), [`runner/ollamarunner/runner.go:890-897`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/ollamarunner/runner.go#L890-L897) (penalty fields from `req.Options` not passed to sampler)
- **Contrast:** The Ollama C++-based runner (`llamarunner`, backed by llama.cpp) at [`runner/llamarunner/runner.go:651-654`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/llamarunner/runner.go#L651-L654) correctly passes all penalty parameters to llama.cpp's native sampling pipeline

### Bug 2: Ollama Registry Config Blob Wires Qwen 3.5 to the Wrong Tool Calling Pipeline

The `qwen3.5:27b-q4_K_M` model published at the Ollama model registry (`registry.ollama.ai`) has its config blob set to `renderer: "qwen3.5"` and `parser: "qwen3.5"`. In Ollama's source code, these strings map to `Qwen3VLRenderer` + `Qwen3Parser` — the **Qwen 3 Hermes-style JSON** tool calling pipeline. However, Qwen 3.5 was trained on the **Qwen3-Coder XML** tool calling format (`<function=name><parameter=key>value</parameter></function>`), as confirmed by the [HuggingFace chat template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json). The correct `Qwen3CoderRenderer` + `Qwen3CoderParser` pipeline already exists in the Ollama codebase but is wired only to the `"qwen3-coder"` model family string.

- **Evidence:** [Section 8](#8-ollama-critical-bug-tool-calling-format-mismatch) and [Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)
- **Registry proof:** Config blob (`sha256:2c5b293a...`) contains `"renderer": "qwen3.5"`, `"parser": "qwen3.5"`
- **Key source files:** [`model/renderers/renderer.go:59-61`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/renderer.go#L59-L61) (wiring), [`model/renderers/qwen3vl.go:57`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go#L57) (JSON system prompt), [`model/parsers/qwen3.go:313`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3.go#L313) (JSON parser via `json.Unmarshal`)

### Bug 3: Unclosed `</think>` Tag Before Tool Calls in Multi-Turn Conversation History

This bug has two sides — the **renderer** (prompt construction for the model) and the **parser** (interpreting model output). The parser side was partially fixed in v0.17.3; the renderer side remains broken as of master (`79917cf`).

**Renderer (STILL BROKEN):** The `Qwen3VLRenderer` (used for the `"qwen3.5"` renderer) has a bug where the `</think>` closing tag is only emitted when the assistant message has non-empty `Content` (at [`qwen3vl.go:98-99`](https://github.com/ollama/ollama/blob/79917cf80bf74538a4ae694e6b61adb908b0f8df/model/renderers/qwen3vl.go#L98-L99)). When an assistant message has thinking text + tool calls but empty content (the common case for a "think then call a tool" turn), the `<tool_call>` block is rendered **inside an unclosed `<think>` block**, corrupting the conversation structure the model sees in multi-turn history. All tool-call thinking tests in `qwen3vl_thinking_test.go` are commented out (lines 119–323).

**Parser (FIXED in v0.17.3):** Commit [`d98dda4`](https://github.com/ollama/ollama/commit/d98dda4676d44a3882fd38492cc00db257f35974) (PR [#14477](https://github.com/ollama/ollama/pull/14477), February 26, 2026) fixed the `Qwen3Parser` and `Qwen3VLParser` to detect `<tool_call>` tags while still in thinking-collection state, correctly transitioning to tool-call parsing without requiring `</think>` first. This means model output like `<think>reasoning<tool_call>...</tool_call>` is now parsed correctly. This fix entered the v0.17.3 stable release.

- **Evidence:** [Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m) (Prompt Template Diff subsection)
- **Key source files:** [`model/renderers/qwen3vl.go:95-120`](https://github.com/ollama/ollama/blob/79917cf80bf74538a4ae694e6b61adb908b0f8df/model/renderers/qwen3vl.go#L95-L120) (renderer, still broken), [`model/parsers/qwen3.go:207-227`](https://github.com/ollama/ollama/blob/79917cf80bf74538a4ae694e6b61adb908b0f8df/model/parsers/qwen3.go#L207-L227) (parser, fixed)

### What Was Investigated But Does NOT Belong in the Bug Report

- **Missing stop tokens in params blob:** The `qwen3.5` params blob has no `stop` sequences, while `qwen3` and `qwen3-coder` do. However, this is **not a bug** — the Ollama Go-based runner has a separate token-ID-based EOS (End of Sequence) check ([`runner.go:768-775`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/ollamarunner/runner.go#L768-L775)) that catches `<|im_end|>` (token 248046) and `<|endoftext|>` (token 248044) from the GGUF file metadata before any string matching occurs. Turn-boundary stopping works correctly without the `stop` field. See the disclaimer in [Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m).
- **Missing `presence_penalty` in params blob:** True but moot — the Go-based runner ignores it anyway (Bug 1 above).

---

## 1. Model Overview

**Released February 24, 2026.** Qwen 3.5 27B is a dense multimodal language model from Alibaba's Qwen team, part of the Qwen 3.5 Medium Model Series.

| Spec | Value |
|------|-------|
| Parameters | 27 billion (dense, all active every forward pass) |
| Layers | 64 (48 linear attention + 16 full attention) |
| Architecture | Hybrid Gated DeltaNet (linear) + sparse attention |
| Hidden Dimension | 5,120 |
| Vocabulary Size | 248,320 tokens (BPE — Byte Pair Encoding) |
| Context Length | 262,144 tokens native, extendable to 1,010,000 via YaRN (Yet another RoPE extensioN) |
| License | Apache 2.0 |
| Modalities | Text + Image + Video (natively trained from scratch) |

### Layer Architecture Pattern

```
16 x (3 x (Gated DeltaNet -> FFN) -> 1 x (Gated Attention -> FFN))
```

A 3:1 ratio of linear attention (Gated DeltaNet) to full attention layers. The Gated DeltaNet provides linear complexity relative to sequence length; periodic full attention layers maintain quality.

### What's New vs Qwen 3

| Feature | Qwen 3 | Qwen 3.5 |
|---------|--------|-----------|
| Architecture | Standard Transformer attention | Gated DeltaNet hybrid (linear + sparse attention) |
| Multimodal | Separate vision encoder bolted on | Natively trained on text, image, and video simultaneously |
| Languages | 119 languages | 201 languages and dialects |
| Vocabulary | 150K tokens | 250K tokens (10-60% encoding efficiency gain) |
| Context | 128K-1M | 262K native, extensible to 1.01M |
| Speed | Baseline | 19x faster decoding at 256K context vs Qwen3-Max |
| Training | Standard | Multi-token prediction (MTP), RL across 1M agent environments |
| Agentic | Basic | Terminal-Bench 2.0: 52.5 (vs 22.5 for Qwen3-Max-Thinking) |
| Tool calling format | Hermes-style JSON in `<tool_call>` tags | Qwen3-Coder XML format: `<function=name><parameter=key>` |

### Key Benchmarks

| Benchmark | Score |
|-----------|-------|
| MMLU-Pro | 86.1 |
| SuperGPQA | 65.6 |
| IFEval | 95.0 |
| SWE-bench Verified | 72.4 (ties GPT-5 mini) |
| LiveCodeBench v6 | 80.7 |
| CodeForces | 1,899 (rating) |
| MMMU | 82.3 |
| MathVision | 86.0 |
| MMBench | 92.6 |
| VideoMME | 87.0 |
| LongBench v2 (262K) | 60.6 |
| BFCL-V4 | 72.2 (122B-A10B variant) |

---

## 2. VRAM Fit and Quantization Options

### Available Quantizations

**Official from Qwen:**
- BF16: ~53.8 GB — [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B)
- FP8 (block size 128): ~28 GB — [Qwen/Qwen3.5-27B-FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8)

**Community GGUF from Unsloth** ([unsloth/Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF)):

| Format | Size | Notes |
|--------|------|-------|
| BF16 | 53.8 GB | Full precision baseline |
| Q8_0 | 28.6 GB | Extremely high quality |
| UD-Q8_K_XL | 32.4 GB | Unsloth Dynamic optimized |
| UD-Q6_K_XL | 23.1 GB | Unsloth Dynamic |
| Q6_K | 22.5 GB | Near-original quality |
| UD-Q5_K_XL | 19.6 GB | Unsloth Dynamic optimized |
| Q5_K_M | 19.6 GB | High quality |
| Q4_K_M | 16.7 GB | Good quality, default for most |
| Q4_K_S | 15.8 GB | Good quality |
| Q3_K_M | 13.5 GB | Lower quality |
| Q2_K | 10.5 GB | Very low quality |

**Community GGUF from bartowski** ([bartowski/Qwen_Qwen3.5-27B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-27B-GGUF)):
Similar spread plus additional imatrix quants (IQ4_XS, IQ3_M, IQ2_M, etc.).

**Community AWQ:**
- `cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4`, `cyankiwi/Qwen3.5-27B-AWQ-4bit`, `QuantTrio/Qwen3.5-27B-AWQ`

### NVIDIA RTX 5090 Graphics Card (32 GB VRAM) Fit Table

| Quantization | Size | VRAM Left | Quality | Recommendation |
|-------------|------|-----------|---------|---------------|
| BF16 | 53.8 GB | Won't fit | Perfect | Multi-GPU only |
| FP8 (official) | ~28 GB | ~4 GB (tight) | Near-perfect | Possible but tight KV cache |
| Q8_0 | 28.6 GB | ~3.4 GB | Excellent | Tight, short context only |
| **UD-Q6_K_XL** | **23.1 GB** | **~9 GB** | **Very high** | **Best quality/fit balance** |
| Q5_K_M | 19.4 GB | ~12.6 GB | High | Good for longer context |
| **Q4_K_M** | **16.5 GB** | **~15.5 GB** | Good | Most context headroom |

**Recommendation:** Start with **UD-Q6_K_XL from Unsloth** (23.1 GB) for best quality-to-fit ratio. If you need maximum context window, use Q4_K_M (16.5 GB).

---

## 3. Recommended Inference Parameters

From the [official Qwen 3.5 27B model card](https://huggingface.co/Qwen/Qwen3.5-27B) on the HuggingFace model repository:

### Thinking Mode (default — model generates `<think>` blocks)

**General tasks:**

| Parameter | Value |
|-----------|-------|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 20 |
| min_p | 0.0 |
| presence_penalty | 1.5 |
| repetition_penalty | 1.0 |

**Precise coding (e.g., WebDev):**

| Parameter | Value |
|-----------|-------|
| temperature | 0.6 |
| top_p | 0.95 |
| top_k | 20 |
| min_p | 0.0 |
| presence_penalty | 0.0 |
| repetition_penalty | 1.0 |

### Non-Thinking / Instruct Mode (`enable_thinking=False`)

**General tasks:**

| Parameter | Value |
|-----------|-------|
| temperature | 0.7 |
| top_p | 0.8 |
| top_k | 20 |
| min_p | 0.0 |
| presence_penalty | 1.5 |
| repetition_penalty | 1.0 |

**Reasoning-heavy tasks:**

| Parameter | Value |
|-----------|-------|
| temperature | 1.0 |
| top_p | 1.0 |
| top_k | 40 |
| min_p | 0.0 |
| presence_penalty | 2.0 |
| repetition_penalty | 1.0 |

### Output Length Recommendations
- General queries: `max_tokens = 32768`
- Complex math/programming: `max_tokens = 81920`

### Repetition Penalty Notes
- The `presence_penalty` parameter ranges from 0 to 2.
- Higher values reduce endless repetitions but may cause language mixing at extreme values.
- The model card uses `presence_penalty` (not `repetition_penalty`) as the primary anti-repetition knob, with `repetition_penalty` kept at 1.0.
- Looping and over-thinking are caused by incorrect inference settings; the recommended parameters resolve these.

---

## 4. vLLM Inference Server Support Status

### What Works

- `Qwen3_5ForConditionalGeneration` is **registered** in the vLLM model registry ([`registry.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/models/registry.py))
- Full Gated DeltaNet implementation with custom Triton GPU kernels (Flash Linear Attention ops) at [`vllm/model_executor/layers/fla/ops/`](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/layers/fla/ops)
- `qwen3_coder` tool call parser implemented — XML-based: `<function=name><parameter=key>value</parameter>` ([`qwen3coder_tool_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py))
- `qwen3` reasoning parser covers both Qwen 3 and Qwen 3.5 ([`qwen3_reasoning_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/reasoning/qwen3_reasoning_parser.py))
- FP8, AWQ quantization supported through the standard pipeline
- MTP (Multi-Token Prediction) speculative decoding supported via `Qwen3_5MTP`
- Dedicated V1 attention backend (`GDNAttentionBackend`) for hybrid KV (Key-Value) cache management

### What's Broken / Concerning

1. **Stable vLLM v0.16.0 does NOT support Qwen 3.5** — you must use the **nightly build**:
   ```bash
   pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly
   ```
   Or Docker: `vllm/vllm-openai:nightly` (or `vllm/vllm-openai:cu130-nightly` for Blackwell GPUs).
   Documented in [Issue #35391](https://github.com/vllm-project/vllm/issues/35391).

2. **Reasoning parser bug ([#35221](https://github.com/vllm-project/vllm/issues/35221)):** If generation is truncated mid-reasoning (before `</think>`), the reasoning text is misclassified as content in non-streaming mode. The code at [`qwen3_reasoning_parser.py:70-73`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/reasoning/qwen3_reasoning_parser.py#L70-L73) returns `(None, model_output)` when no `</think>` is found. The serving layer mitigates this for the `enable_thinking=False` case via `prompt_is_reasoning_end_arr`, but the truncation case is unhandled.

3. **Tool calling bugs:**
   - [#21711](https://github.com/vllm-project/vllm/issues/21711): `json.loads` fails because output includes XML `<tool_call>` tags
   - [#20611](https://github.com/vllm-project/vllm/issues/20611): Streaming + tool calls fails with thinking disabled
   - [#19051](https://github.com/vllm-project/vllm/issues/19051): `tool_choice: required` + reasoning parsing causes 400 errors
   - [#29192](https://github.com/vllm-project/vllm/issues/29192): `qwen3_coder` parser can produce infinite "!" streams with long inputs

4. **FP8 limitation:** The Gated DeltaNet's `ba_proj` (beta/alpha projection) does not support blockwise FP8 quantization.

5. **No default parameter injection:** vLLM's `Qwen3_5TextConfig` defines architecture parameters only — no sampling parameters. You must set `temperature`, `top_p`, `top_k`, `presence_penalty` explicitly in every API request.

### Recommended vLLM Inference Server Launch Command

```bash
vllm serve Qwen/Qwen3.5-27B-FP8 \
    --port 8000 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

---

## 5. Ollama Inference Framework Support Status

### What Works

- Architecture fully registered: `qwen35` in GGUF (GPT-Generated Unified Format), `Qwen3_5ForConditionalGeneration` in the Ollama model converter ([`convert.go:323`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/convert/convert.go#L323))
- Complete Gated DeltaNet implementation in both Go ([`deltanet.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/models/qwen3next/deltanet.go)) and C++ (llama.cpp backend)
- CUDA GPU kernels for `ssm_conv` and `solve_tri` operations
- Dedicated `qwen3.5` chat renderer with thinking support ([`renderer.go:59-61`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/renderer.go#L59-L61))
- V-head reordering (grouped to tiled) handled correctly during GGUF conversion ([`convert_qwen3next.go:236-251`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/convert/convert_qwen3next.go#L236-L251))
- Custom quantization rules for linear attention tensors — e.g., Q6_K for value heads in Q4_K_M ([`quantization.go:65-80`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/server/quantization.go#L65-L80))
- 27B confirmed working on Ollama v0.17.1-rc2 ([issue #14433](https://github.com/ollama/ollama/issues/14433))

### What's Broken / Concerning

> **Items marked BUG REPORT are verified, source-linked, and ready for inclusion in a bug report to the Ollama team. Items marked INFORMATIONAL are true observations but should not be filed as bugs.**

1. **BUG REPORT — Penalty sampling completely missing** — see [Section 7](#7-ollama-critical-bug-missing-penalty-sampling) and [Section 6](#6-ollama-deep-dive-parameter-flow-validation)
2. **BUG REPORT — Tool calling format mismatch** — the Ollama inference framework sends Qwen 3 Hermes-style JSON format but the Qwen 3.5 model was trained on Qwen3-Coder XML format — see [Section 8](#8-ollama-critical-bug-tool-calling-format-mismatch)
3. **BUG REPORT — Unclosed `</think>` tag** — in multi-turn tool call history, thinking blocks are never closed before tool calls when assistant content is empty — see [Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)
4. **INFORMATIONAL — Thinking mode disable broken** ([#14418](https://github.com/ollama/ollama/issues/14418)) — `think: false` does not work. **This is actually desirable for agent use cases** — thinking should always be on. The model's agentic training assumes thinking mode.
5. **INFORMATIONAL — Parallel requests forced to 1** for `qwen35` architecture ([`sched.go:450`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/server/sched.go#L450)) due to hybrid recurrent state — this is an expected architectural limitation
6. **INFORMATIONAL — Only one 27B tag** in the Ollama model library: `qwen3.5:27b-q4_K_M` (17 GB)

---

## 6. Ollama Inference Framework Deep Dive: Parameter Flow Validation

### The Priority Chain (from [`routes.go:116-131`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/server/routes.go#L116-L131))

```
Step 1: DefaultOptions()        — hardcoded baseline in Ollama source code
Step 2: VRAM-based num_ctx      — if DefaultOptions.NumCtx == 0, auto-set based on available GPU VRAM
Step 3: model.Options            — from the model's params blob in the Ollama model registry (overrides Steps 1-2)
Step 4: requestOpts              — from the user's API call (overrides everything)
```

### Step 1: DefaultOptions() (from [`types.go:1054-1080`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/api/types.go#L1054-L1080))

```go
Temperature:      0.8
TopK:             40
TopP:             0.9
TypicalP:         1.0
RepeatLastN:      64
RepeatPenalty:    1.1
PresencePenalty:  0.0
FrequencyPenalty: 0.0
NumPredict:       -1
NumKeep:          4
Seed:             -1
NumCtx:           0  // sentinel, replaced by VRAM tier
NumBatch:         512
NumGPU:           -1
```

### Step 3: Model Params Blob

The values `temperature=1, top_k=20, top_p=0.95` shown on the [Ollama model library page for `qwen3.5:27b-q4_K_M`](https://ollama.com/library/qwen3.5:27b-q4_K_M) come from the `application/vnd.ollama.image.params` JSON blob in the model manifest hosted at the Ollama model registry (`registry.ollama.ai`). This blob is baked in by the Ollama team when they publish the model to the registry. When you run `ollama pull`, this blob is downloaded as-is.

These are NOT from GGUF file metadata (the Ollama inference framework ignores `general.sampling.*` GGUF metadata keys entirely — confirmed: zero references to those keys in the Ollama Go codebase).

### Effective Parameters for `qwen3.5:27b-q4_K_M` With No API Overrides

| Parameter | DefaultOptions() | Model Params Blob (from Ollama registry) | Effective Value | Actually Used by Ollama Go-Based Runner (`ollamarunner`)? |
|-----------|-----------------|-------------------|-----------------|---------------------------|
| temperature | 0.8 | 1.0 | **1.0** | **YES** |
| top_k | 40 | 20 | **20** | **YES** |
| top_p | 0.9 | 0.95 | **0.95** | **YES** |
| min_p | 0.0 | *(not set)* | **0.0** | **YES** |
| repeat_penalty | 1.1 | *(not set)* | **1.1** | **NO** (ignored) |
| presence_penalty | 0.0 | *(not set)* | **0.0** | **NO** (ignored) |
| frequency_penalty | 0.0 | *(not set)* | **0.0** | **NO** (ignored) |
| repeat_last_n | 64 | *(not set)* | **64** | **NO** (ignored) |
| typical_p | 1.0 | *(not set)* | **1.0** | **NO** (ignored) |
| seed | -1 | *(not set)* | **-1** (random) | **YES** |
| num_ctx | 0 -> 32768 | *(not set)* | **32768** (VRAM tier) | **YES** |

---

## 7. Ollama Critical Bug: Missing Penalty Sampling

> **BUG REPORT ITEM** — This is a verified bug in the Ollama inference framework's Go-based runner. It affects all models forced onto the Go-based runner via the `OllamaEngineRequired()` list, not just Qwen 3.5. It is unfixable by end users via API parameters or configuration.

### The Problem

The Qwen 3.5 27B model is in the `OllamaEngineRequired()` list ([`ggml.go:278-301`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/fs/ggml/ggml.go#L278-L301)), which forces it to use the **Ollama Go-based runner (`ollamarunner`)** instead of the **Ollama C++-based runner (`llamarunner`, backed by the llama.cpp library)**.

The Go-based sampler at [`samplers.go:130-165`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/sample/samplers.go#L130-L165) creates a `Sampler` struct with only these fields:

```go
return Sampler{
    rng:         rng,
    topK:        topK,
    topP:        topP,
    minP:        minP,
    temperature: temperature,
    grammar:     grammar,
}
```

The `sample()` method at [line 86](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/sample/samplers.go#L86) applies only:
1. `topK` — sorts and truncates to top K tokens
2. `temperature` — scales logits
3. `softmax` — normalizes to probabilities
4. `topP` — nucleus sampling
5. `minP` — minimum probability filtering

**Not implemented at all in the Ollama Go-based runner's sampler:**
- `repeat_penalty` — silently accepted by the Ollama API, silently ignored by the sampler
- `presence_penalty` — silently accepted by the Ollama API, silently ignored by the sampler
- `frequency_penalty` — silently accepted by the Ollama API, silently ignored by the sampler
- `repeat_last_n` — silently accepted by the Ollama API, silently ignored by the sampler
- `typical_p` — silently accepted by the Ollama API, silently ignored by the sampler

The Ollama C++-based runner (`llamarunner`, at [`runner.go:645-656`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/llamarunner/runner.go#L645-L656)) DOES pass all penalty parameters to llama.cpp's native sampling pipeline — but the Qwen 3.5 model **cannot use it** because it is forced onto the Go-based runner via the `OllamaEngineRequired()` list.

### Impact

The Qwen 3.5 27B model card on the HuggingFace model repository explicitly recommends `presence_penalty=1.5` to prevent repetition loops in thinking mode. Even if you set it via the Ollama API:

```json
{"options": {"presence_penalty": 1.5}}
```

**It will be silently accepted by the Ollama API but completely ignored by the Go-based runner's sampler.** The Qwen 3.5 model may enter repetition loops during long thinking sequences with no API-level workaround available in the Ollama inference framework.

### Confirmation Path

The Go-based sampler is invoked from the Ollama Go-based runner at [`ollamarunner/runner.go:890-897`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/ollamarunner/runner.go#L890-L897):

```go
sampler := sample.NewSampler(
    req.Options.Temperature,
    req.Options.TopK,
    req.Options.TopP,
    req.Options.MinP,
    req.Options.Seed,
    grammar,
)
```

No penalty parameters are passed. The `req.Options.RepeatPenalty`, `req.Options.PresencePenalty`, and `req.Options.FrequencyPenalty` fields are available in the `req.Options` struct but are simply not passed to the `NewSampler` function. They are not part of the `NewSampler` function signature.

---

## 8. Ollama Critical Bug: Tool Calling Format Mismatch

> **BUG REPORT ITEM** — This is a verified wiring bug in the Ollama inference framework's source code AND in the Ollama model registry config blob for `qwen3.5:27b-q4_K_M`. The correct tool calling pipeline already exists in the Ollama codebase but is wired to the wrong model family string. This is unfixable by end users — only the Ollama team can change the source code wiring or the registry config blob.

### The Core Problem

**Qwen 3.5 changed its tool calling format from Qwen 3.** The older Qwen 3 model family used Hermes-style JSON inside `<tool_call>` tags. The newer Qwen 3.5 model family adopted the Qwen3-Coder XML format. This is confirmed by the [chat template in `tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) on the HuggingFace model repository.

**The Ollama inference framework v0.17.1 wires the `"qwen3.5"` model family to the wrong renderer and parser.** It uses the Qwen 3 JSON pipeline instead of the Qwen3-Coder XML pipeline.

### What Qwen 3.5 Was Trained to Output (HuggingFace Ground Truth)

The Qwen 3.5 model's chat template on the HuggingFace model repository (extracted from `tokenizer_config.json`) produces this format for tool calls:

```xml
<tool_call>
<function=get_temperature>
<parameter=location>
San Francisco
</parameter>
<parameter=unit>
celsius
</parameter>
</function>
</tool_call>
```

And the chat template injects tools into the system prompt as:

```
# Tools

You have access to the following functions:

<tools>
{"type": "function", "function": {"name": "get_temperature", ...}}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>

<IMPORTANT>
...Function calls MUST follow the specified format...
</IMPORTANT>
```

### What the Ollama Inference Framework Actually Sends and Expects

**The renderer** (`Qwen3VLRenderer` at [`qwen3vl.go:37-147`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go#L37-L147)) generates the **Qwen 3 Hermes-style JSON** system prompt:

```
For each function call, return a json object with function name and arguments
within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
```

Source: [`qwen3vl.go:57`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go#L57) — the system prompt instruction hardcoded in the renderer.

**The parser** (`Qwen3Parser` at [`qwen3.go:307-335`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3.go#L307-L335)) expects JSON inside `<tool_call>` tags:

```go
func parseQwen3ToolCall(raw qwen3EventRawToolCall, tools []api.Tool) (api.ToolCall, error) {
    var parsed struct {
        Name      string         `json:"name"`
        Arguments map[string]any `json:"arguments"`
    }
    if err := json.Unmarshal([]byte(raw.raw), &parsed); err != nil {
        return api.ToolCall{}, fmt.Errorf("failed to parse JSON: %w", err)
    }
```

Source: [`qwen3.go:307-315`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3.go#L307-L315)

### The Wiring

The name `"qwen3.5"` is mapped in two places:

**Renderer** ([`renderer.go:59-61`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/renderer.go#L59-L61)):
```go
case "qwen3.5":
    renderer := &Qwen3VLRenderer{isThinking: true, emitEmptyThinkOnNoThink: true, useImgTags: RenderImgTags}
    return renderer
```

**Parser** ([`parsers.go:52-53`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/parsers.go#L52-L53)):
```go
case "qwen3.5":
    p = &Qwen3Parser{hasThinkingSupport: true, defaultThinking: true}
```

### The Correct Pipeline Already Exists in the Ollama Codebase

The Ollama inference framework has a complete Qwen3-Coder pipeline that correctly handles the XML format — it just isn't wired to the `"qwen3.5"` model family string:

**`Qwen3CoderRenderer`** ([`qwen3coder.go:60-193`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3coder.go#L60-L193)):
- Formats tool definitions as structured XML: `<function><name>...</name><parameters>...</parameters></function>`
- Tells the model to use `<function=name><parameter=key>value</parameter></function>` — matching the model's training
- Re-renders prior tool calls in XML format for multi-turn conversations

**`Qwen3CoderParser`** ([`qwen3coder.go:31-194`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3coder.go#L31-L194)):
- Parses `<function=name><parameter=key>value</parameter></function>` XML via `transformToXML()` + `xml.Unmarshal`
- Does schema-aware type coercion: matches parameter values against tool definitions, converts strings to int/float/bool/array/object as appropriate ([`qwen3coder.go:281-388`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3coder.go#L281-L388))
- Handles XML escaping of special characters in parameter values

These are wired only to `"qwen3-coder"` ([`renderer.go:50-51`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/renderer.go#L50-L51), [`parsers.go:54-55`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/parsers.go#L54-L55)).

### What Happens In Practice

Two failure modes:

1. **Model obeys system prompt (JSON mode):** The `Qwen3VLRenderer`'s system prompt instructs JSON output. If the Qwen 3.5 model complies with the JSON instruction, the `Qwen3Parser` successfully parses it. But the model is operating outside its primary training distribution for tool calls, likely producing lower-quality tool selections and argument formatting. The Qwen 3.5 model was RL (Reinforcement Learning)-trained across 1 million agent environments using the XML format, not the JSON format.

2. **Model ignores system prompt (XML mode):** If the Qwen 3.5 model falls back to its trained XML format (`<function=get_temperature><parameter=location>...`), the `json.Unmarshal` call in `parseQwen3ToolCall` fails immediately. The error propagates up through [`qwen3.go:106-107`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3.go#L106-L107), which returns the error and **loses the entire content chunk** — no fallback to treating it as regular content.

### Why the Old Qwen 3 GitHub Issues Don't Apply to Qwen 3.5

The previously cited GitHub issues are **Qwen 3 issues using the Qwen 3 Hermes-style JSON format**, and do not directly apply to Qwen 3.5:

- [Ollama issue #11135](https://github.com/ollama/ollama/issues/11135) (tool hallucination) — Reported against Qwen 3 models using the Hermes-style JSON format. Qwen 3.5's XML format is structurally different and the Qwen 3.5 model was trained with far more agentic reinforcement learning data (1 million environments vs Qwen 3's standard training).

- [Ollama issue #11662](https://github.com/ollama/ollama/issues/11662) (JSON-as-content) — A Qwen 3-era issue where the model output JSON tool calls as content text instead of structured output. With Qwen 3.5 using XML, this specific failure pattern changes entirely.

- [Goose framework issue #6883](https://github.com/block/goose/issues/6883) (format switching with many tools) — A Qwen 3 behavior where the model switched between JSON and XML formats. Qwen 3.5 was trained natively on XML, so format consistency should be stronger (but is now undermined by the Ollama inference framework sending JSON instructions).

### The Fix

The correct fix is to wire `"qwen3.5"` to the Qwen3-Coder pipeline with thinking support added:

**Renderer:** Use `Qwen3CoderRenderer` (or a variant that adds thinking prefill, since `Qwen3CoderRenderer` currently has no thinking support — [`qwen3coder.go:43`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3coder.go#L43): `HasThinkingSupport() returns false`).

**Parser:** Use `Qwen3CoderParser` with thinking support added. Currently `HasThinkingSupport()` returns `false` ([`qwen3coder.go:41-43`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3coder.go#L41-L43)), so `<think>` blocks would not be extracted. A combined parser that handles both thinking tags (from `Qwen3Parser`) and XML tool calls (from `Qwen3CoderParser`) is needed.

This is an Ollama-side change that only the Ollama team can make. The Qwen 3.5 model's tool calling capability is strong — it was trained on 20,000 parallel rollout environments for agentic tasks. The issue is entirely in the Ollama inference framework's wiring, not in the model.

### Additional Parser Quality Issues

Even if the parser/renderer mismatch were fixed, two secondary issues remain in the `Qwen3Parser` (and would need attention in any replacement):

1. **No tool name validation:** [`qwen3.go:321`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3.go#L321) explicitly discards the tools list (`_ = tools`). Hallucinated tool names pass through unchecked. The `Qwen3CoderParser` does validate against the tool list ([`qwen3coder.go:234-240`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3coder.go#L234-L240)) — another reason to prefer it.

2. **Fatal error on malformed tool calls:** At [`qwen3.go:106-107`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3.go#L106-L107), a parse error kills the entire content chunk with no fallback. Should degrade to treating the raw text as content instead.

---

## 9. Ollama Model Registry Validation: `qwen3.5:27b-q4_K_M`

> **BUG REPORT ITEM** — The registry config blob and the renderer behavior validated here contain bug-report-worthy findings (wrong renderer/parser wiring, unclosed `</think>` tag).

This section validates the actual model published to the Ollama model registry at [ollama.com/library/qwen3.5:27b-q4_K_M](https://ollama.com/library/qwen3.5:27b-q4_K_M) against the ground truth from the [HuggingFace model repository chat template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json). The manifest, config, and params blobs were pulled directly from the Ollama model registry API at `registry.ollama.ai`.

### Registry Manifest

```
GET https://registry.ollama.ai/v2/library/qwen3.5/manifests/27b-q4_K_M
```

| Layer | Media Type | Size | Digest |
|-------|-----------|------|--------|
| GGUF weights | `application/vnd.ollama.image.model` | 17.4 GB | `sha256:7935de6e…` |
| License | `application/vnd.ollama.image.license` | 11 KB | `sha256:7339fa41…` |
| Params | `application/vnd.ollama.image.params` | 42 bytes | `sha256:f6417cb1…` |

**Notable absence:** No `application/vnd.ollama.image.template` layer. Unlike `qwen3` (which has a 1,723-byte Go template blob), `qwen3.5` relies entirely on the built-in renderer compiled into the Ollama binary. This is the newer Ollama architecture pattern — not a bug per se, but it means the prompt format is hardcoded in Go source code, not overridable by the registry or by end users.

### Config Blob

```json
{
    "model_format": "gguf",
    "model_family": "qwen35",
    "model_type": "27.8B",
    "file_type": "Q4_K_M",
    "renderer": "qwen3.5",
    "parser": "qwen3.5",
    "requires": "0.17.1"
}
```

**`renderer: "qwen3.5"`** maps to `Qwen3VLRenderer` with `isThinking: true, emitEmptyThinkOnNoThink: true` ([`renderer.go:59-61`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/renderer.go#L59-L61)). This renders Qwen 3-style Hermes JSON tool call prompts. **This is wrong for Qwen 3.5** — see [Section 8](#8-ollama-critical-bug-tool-calling-format-mismatch).

**`parser: "qwen3.5"`** maps to `Qwen3Parser` with `hasThinkingSupport: true, defaultThinking: true` ([`parsers.go:52-53`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/parsers.go#L52-L53)). This parses Qwen 3-style JSON inside `<tool_call>` tags via `json.Unmarshal`. **This is wrong for Qwen 3.5** — the model outputs XML, not JSON.

For comparison, the `qwen3-coder` model in the Ollama model registry uses `renderer: "qwen3-coder"` + `parser: "qwen3-coder"` — the correct XML pipeline that matches the Qwen 3.5 model's training format.

### Params Blob

```json
{"temperature": 1, "top_k": 20, "top_p": 0.95}
```

**42 bytes. Three parameters. That's it.**

| Parameter | Params Blob (Ollama Registry) | Model Card (HuggingFace, Thinking General) | Match? |
|-----------|-------------|-------------------------------|--------|
| temperature | 1 | 1.0 | YES |
| top_k | 20 | 20 | YES |
| top_p | 0.95 | 0.95 | YES |
| presence_penalty | *(missing, defaults to 0.0)* | 1.5 | **NO** — missing from params blob, but moot since the Ollama Go-based runner ignores it anyway (see [Bug 1](#7-ollama-critical-bug-missing-penalty-sampling)) |
| stop | *(missing, defaults to none)* | — | See disclaimer below — not a bug |

For comparison, `qwen3-coder` bakes in 5 parameters:
```json
{"repeat_penalty": 1.05, "stop": ["<|im_start|>", "<|im_end|>", "<|endoftext|>"], "temperature": 0.7, "top_k": 20, "top_p": 0.8}
```

And `qwen3` bakes in 4:
```json
{"repeat_penalty": 1, "stop": ["<|im_start|>", "<|im_end|>"], "temperature": 0.6, "top_k": 20, "top_p": 0.95}
```

### Stop Tokens: Not Missing, Not a Bug

> **NOT A BUG — Do not include in the bug report.** After detailed verification, the missing `stop` field in the params blob does not cause incorrect behavior. Including this in a bug report would weaken credibility.

The `qwen3.5` params blob defines **no string-based stop sequences**. Both `qwen3` and `qwen3-coder` do define them:

| Model (Ollama Registry) | Stop Sequences in Params Blob |
|-------|---------------|
| qwen3 | `<\|im_start\|>`, `<\|im_end\|>` |
| qwen3-coder | `<\|im_start\|>`, `<\|im_end\|>`, `<\|endoftext\|>` |
| **qwen3.5** | ***(none)*** |

However, the Ollama Go-based runner has **two independent stop mechanisms** ([`runner.go:768-836`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/ollamarunner/runner.go#L768-L836)):

1. **GGUF EOS (End of Sequence) token ID check** (line 768): The `qwen3next` model code reads `tokenizer.ggml.eos_token_id` and `tokenizer.ggml.eos_token_ids` from the GGUF file metadata ([`model.go:601-604`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/models/qwen3next/model.go#L601-L604)). From the HuggingFace model repository [`generation_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/generation_config.json), Qwen 3.5 defines two EOS token IDs: **248046** (`<|im_end|>`) and **248044** (`<|endoftext|>`). These are baked into the GGUF file and checked on every sampled token **before** any string matching occurs.

2. **String-based stop sequences** (line 798): From the `stop` field in the params blob or API request. For `qwen3.5`, this is empty.

**Why this is not a bug:** The Qwen 3.5 model is trained to end its turn by generating the `<|im_end|>` token (ID 248046), which is caught by mechanism #1 (the token-ID EOS check). The `<|im_start|>` token (ID 248045) is NOT an EOS token, but the model is not trained to generate it — that token only appears in the prompt template, prepended by the application layer. The string-based `stop` field is a second line of defense, not the primary mechanism. The fact that `qwen3` and `qwen3-coder` include `<|im_start|>` in their `stop` field is an extra safety measure, not a necessity for correct behavior.

### Prompt Template Diff: HuggingFace Ground Truth vs Ollama Renderer Output

The [HuggingFace model repository chat template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) is the ground truth — it's the Jinja2 template the Qwen 3.5 model was trained against. The Ollama `Qwen3VLRenderer` is what the Ollama inference framework actually sends to the model. Both were rendered for identical test inputs and diffed.

**Scenario: No tools, thinking enabled** — **Identical.** No issues.

**Scenario: Thinking disabled** — **Identical.** Both produce `<think>\n\n</think>\n\n` prefill.

**Scenario: With tools, thinking enabled** — **Massively different.** Here is the unified diff of the system prompt:

> **BUG REPORT ITEM** — The diff below shows 6 concrete mismatches between what the Ollama inference framework sends to the model and what the model was trained on. The most critical mismatch is #3 (JSON vs XML format instruction).

```diff
--- OLLAMA INFERENCE FRAMEWORK (what is actually sent to the model)
+++ HUGGINGFACE MODEL REPOSITORY (what the model was trained on)
@@ system message @@
 <|im_start|>system
-You are a helpful assistant.
-
 # Tools

-You may call one or more functions to assist with the user query.
+You have access to the following functions:

-You are provided with function signatures within <tools></tools> XML tags:
 <tools>
 {JSON tool definition}
 </tools>

-For each function call, return a json object with function name and
-arguments within <tool_call></tool_call> XML tags:
+If you choose to call a function ONLY reply in the following format
+with NO suffix:
+
 <tool_call>
-{"name": <function-name>, "arguments": <args-json-object>}
-</tool_call><|im_end|>
+<function=example_function_name>
+<parameter=example_parameter_1>
+value_1
+</parameter>
+</function>
+</tool_call>
+
+<IMPORTANT>
+Reminder:
+- Function calls MUST follow the specified format...
+- Required parameters MUST be specified
+- You may provide optional reasoning... BEFORE the function call, but NOT after
+- If there is no function call available, answer the question like normal...
+</IMPORTANT>
+
+You are a helpful assistant.<|im_end|>
```

Six distinct mismatches in the system prompt alone:

| # | Mismatch | Severity |
|---|----------|----------|
| 1 | **System message position**: Ollama inference framework puts user's system message first; HuggingFace ground truth puts it last (after tool instructions) | Moderate — changes relative priority the model assigns to user vs tool instructions |
| 2 | **Tool header text**: Ollama says "You may call one or more functions…provided with function signatures within XML tags". HuggingFace says "You have access to the following functions:" | Low — different wording, same meaning |
| 3 | **Format instruction shows JSON instead of XML**: Ollama shows `{"name": …, "arguments": …}`. HuggingFace shows `<function=name><parameter=key>value</parameter></function>` | **CRITICAL — BUG REPORT ITEM** — tells the model to produce the wrong output format (see [Section 8](#8-ollama-critical-bug-tool-calling-format-mismatch)) |
| 4 | **Missing `<IMPORTANT>` block**: HuggingFace ground truth includes 4 rules about format compliance, required params, reasoning placement. Ollama has nothing | Moderate — model loses formatting guardrails |
| 5 | **Missing format example**: HuggingFace shows a multi-line, multi-parameter example with realistic structure. Ollama shows a 1-line JSON placeholder | Moderate — model has less guidance on how to format complex tool calls |
| 6 | **`<tools>` preamble text**: Ollama includes "You are provided with function signatures within `<tools></tools>` XML tags:" before the `<tools>` block. HuggingFace has no such preamble. | Low |

**Scenario: Multi-turn with thinking + tool call in history** — **Critically broken.**

> **BUG REPORT ITEM** — The unclosed `</think>` tag below (issue #1) is a renderer bug that corrupts multi-turn conversation history.

```diff
--- OLLAMA INFERENCE FRAMEWORK (what is sent for prior assistant turn)
+++ HUGGINGFACE MODEL REPOSITORY (what model expects)
 <|im_start|>assistant
 <think>
-I need to check the weather.<tool_call>
-{"name": "get_weather", "arguments": {"location": "Paris"}}
+I need to check the weather.
+</think>
+
+<tool_call>
+<function=get_weather>
+<parameter=location>
+Paris
+</parameter>
+</function>
 </tool_call><|im_end|>
```

Three critical issues in the multi-turn rendering:

| # | Issue | Severity |
|---|-------|----------|
| 1 | **Missing `</think>` closing tag**: When an assistant message has thinking + tool calls but empty content, the Ollama renderer never closes the `<think>` block. The `<tool_call>` appears *inside* the unclosed think block. This corrupts the model's understanding of the conversation structure. Source: [`qwen3vl.go:98-99`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go#L98-L99) — `</think>` only emitted when `content != ""`. All tool-call thinking tests commented out in `qwen3vl_thinking_test.go` lines 119-323. | **CRITICAL — BUG REPORT ITEM** |
| 2 | **Tool calls in JSON format in history**: Prior tool calls are re-rendered as `{"name": …, "arguments": …}`. The model was trained seeing XML tool calls in its history. | **CRITICAL — BUG REPORT ITEM** — same issue as the system prompt (Section 8), but now in the conversation history |
| 3 | **No blank line separation**: HuggingFace ground truth has `</think>\n\n<tool_call>` (blank line between thinking and tool call). Ollama has no separator. | Low |

### Summary: Ollama Model Registry Issues for `qwen3.5:27b-q4_K_M`

| Issue | Category | Severity | Bug Report? |
|-------|----------|----------|-------------|
| Wrong `renderer: "qwen3.5"` (JSON) instead of `"qwen3-coder"` (XML) | Config blob at Ollama registry | **CRITICAL** | **YES** |
| Wrong `parser: "qwen3.5"` (JSON) instead of `"qwen3-coder"` (XML) | Config blob at Ollama registry | **CRITICAL** | **YES** |
| Empty `stop` field (no string-based stop sequences) | Params blob at Ollama registry | Not a bug — EOS token IDs in GGUF file handle stopping | No |
| Missing `presence_penalty` in params | Params blob at Ollama registry | Moot — Ollama Go-based runner ignores it anyway ([Bug 1](#7-ollama-critical-bug-missing-penalty-sampling)) | No (covered by Bug 1) |
| No template blob (intentional — uses compiled-in renderer) | Manifest at Ollama registry | Neutral | No |
| Unclosed `</think>` tag in multi-turn tool call history | Renderer bug in Ollama source code | **CRITICAL** | **YES** |
| System message placed before tool instructions (should be after) | Renderer bug in Ollama source code | Moderate | Optional |
| Missing `<IMPORTANT>` format compliance block | Renderer bug in Ollama source code | Moderate | Optional |

---

## 10. Ollama Inference Framework Context Length

### VRAM-Based Tiered Defaults (from [`routes.go:1758-1767`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/server/routes.go#L1758-L1767))

```go
switch {
case totalVRAM >= 47*format.GibiByte:
    s.defaultNumCtx = 262144
case totalVRAM >= 23*format.GibiByte:
    s.defaultNumCtx = 32768
default:
    s.defaultNumCtx = 4096
}
```

**Your NVIDIA RTX 5090 graphics card (32 GB VRAM) falls in the >= 23 GB tier: default num_ctx = 32,768 tokens.**

The old Ollama documentation still says "Default: 2048" ([`modelfile.mdx:153`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/docs/modelfile.mdx#L153)) but this is dead text — overridden by the VRAM-based logic.

### Priority Chain for num_ctx

```
1. VRAM-based default (32768 for your NVIDIA RTX 5090 graphics card)
2. OLLAMA_CONTEXT_LENGTH environment variable (if set, overrides #1)
3. Model params blob num_ctx from the Ollama model registry (if model bakes one in, overrides #1 and #2)
4. API request num_ctx (always wins)
```

Confirmed by test cases in [`routes_options_test.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/server/routes_options_test.go).

### Maximum Practical Context Length

With `qwen3.5:27b-q4_K_M` (17 GB model weights) on 32 GB VRAM, approximately 15 GB remains for KV (Key-Value) cache and overhead.

The hybrid architecture is favorable here:
- 48 out of 64 layers use **linear attention (Gated DeltaNet)** — these have **fixed-size** recurrent state, NOT per-token KV cache
- Only **16 layers** use full attention with traditional KV cache
- Full attention KV cache: 16 layers x 2 (K+V) x 4 heads x 256 dim x 2 bytes (FP16) = **64 KB per token**
- Plus fixed recurrent state for 48 linear layers (one-time cost)

With approximately 14 GB available after recurrent state overhead, the theoretical maximum is 200K+ tokens of KV cache. The VRAM default of 32,768 is conservative — you could push to 65,536 or 131,072 with testing.

---

## 11. Ollama Inference Framework Per-Use-Case Parameter Settings

### Thinking Mode — General (default, model bakes these in)

```json
{
  "options": {
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "num_ctx": 32768
  }
}
```

Already baked in to the Ollama model registry params blob — no need to set unless overriding. Note: `presence_penalty=1.5` is recommended by the Qwen 3.5 model card on HuggingFace but **cannot be applied** (the Ollama Go-based runner silently ignores it — see [Bug 1](#7-ollama-critical-bug-missing-penalty-sampling)).

### Thinking Mode — Precise Coding

```json
{
  "options": {
    "temperature": 0.6,
    "top_k": 20,
    "top_p": 0.95,
    "num_ctx": 32768
  }
}
```

Override temperature from 1.0 to 0.6. The Qwen 3.5 model card says `presence_penalty=0.0` for precise coding, which is the effective value anyway (since the Ollama Go-based runner ignores it).

### Non-Thinking Mode — General

```json
{
  "options": {
    "temperature": 0.7,
    "top_k": 20,
    "top_p": 0.8,
    "num_ctx": 32768
  },
  "think": false
}
```

Override temperature and top_p. **Warning:** `think: false` is reportedly broken ([Ollama issue #14418](https://github.com/ollama/ollama/issues/14418)), but as noted, thinking-always-on is desirable for agentic use. The Qwen 3.5 model card recommends `presence_penalty=1.5` — impossible to apply in the Ollama inference framework (see [Bug 1](#7-ollama-critical-bug-missing-penalty-sampling)).

### Non-Thinking Mode — Heavy Reasoning

```json
{
  "options": {
    "temperature": 1.0,
    "top_k": 40,
    "top_p": 1.0,
    "num_ctx": 32768
  },
  "think": false
}
```

The Qwen 3.5 model card recommends `presence_penalty=2.0` — impossible to apply in the Ollama inference framework (see [Bug 1](#7-ollama-critical-bug-missing-penalty-sampling)).

---

## 12. Tool Calling Verdict

### Format Change: Qwen 3 vs Qwen 3.5

This is the single most important fact for framework integration: **Qwen 3.5 uses a completely different tool calling format from Qwen 3.**

| Aspect | Qwen 3 (Hermes JSON) | Qwen 3.5 (Coder XML) |
|--------|----------------------|----------------------|
| Tool call output | `<tool_call>{"name":"fn","arguments":{...}}</tool_call>` | `<tool_call><function=fn><parameter=key>value</parameter></function></tool_call>` |
| Tool definition in prompt | JSON in `<tools></tools>` | JSON in `<tools></tools>` (same) |
| System prompt header | "You may call one or more functions..." | "You have access to the following functions:" |
| vLLM parser flag | `--tool-call-parser hermes` | `--tool-call-parser qwen3_coder` |
| Tool result wrapping | `<tool_response>...</tool_response>` in user msg | Same |

### vLLM

The `qwen3_coder` tool call parser ([`qwen3coder_tool_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py)) handles the correct XML format with:
- Non-streaming extraction via regex
- Streaming extraction with incremental XML parsing
- Schema-aware type conversion (`_convert_param_value`)

Launch with:
```bash
--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
```

Multiple open bugs exist: XML parsing failures, streaming conflicts, infinite "!" streams. The `qwen3_coder` parser also has a [security vulnerability](https://github.com/vllm-project/vllm/security/advisories/GHSA-79j6-g2m3-jgfw) — it uses Python's `eval()` for unknown parameter types.

### Ollama Inference Framework

**The tool calling pipeline in the Ollama inference framework is fundamentally misconfigured for Qwen 3.5.** See [Section 8](#8-ollama-critical-bug-tool-calling-format-mismatch) for the full analysis.

Summary: The Ollama inference framework uses `Qwen3VLRenderer` (Hermes-style JSON format) + `Qwen3Parser` (JSON parser via `json.Unmarshal`) for a model trained on `Qwen3CoderRenderer` (XML format) + `Qwen3CoderParser` (XML parser via `xml.Unmarshal`). The correct pipeline exists in the Ollama codebase but is wired to `"qwen3-coder"` instead of `"qwen3.5"`.

**This is an Ollama inference framework wiring bug, not a model training issue.** The Qwen 3.5 model has strong agentic tool calling capabilities (BFCL-V4: 72.2, Terminal-Bench: 52.5, trained across 20,000 parallel reinforcement learning environments). The framework is just pointed at the wrong plumbing.

| Engine | Tool Calling Status | Root Cause |
|--------|-------------------|------------|
| vLLM inference server | **Correct format, buggy implementation** | Right parser (`qwen3_coder`), multiple open issues in streaming/parsing |
| Ollama inference framework | **Wrong format entirely** | Uses Qwen 3 Hermes-style JSON pipeline for a Qwen 3.5 model trained on Qwen3-Coder XML format |

---

## 13. llama.cpp C++ Inference Library Backend Validation

### Gated DeltaNet Implementation

**Complete and numerically sound.** Implemented across:
- [`src/models/qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/723c71064da0908c19683f8c344715fbf6d986fd/src/models/qwen35.cpp) — Qwen 3.5 model graph
- [`src/models/delta-net-base.cpp`](https://github.com/ggml-org/llama.cpp/blob/723c71064da0908c19683f8c344715fbf6d986fd/src/models/delta-net-base.cpp) — Core DeltaNet algorithms

Two compute paths:
- `build_delta_net_chunking` (prefill, chunk size 64)
- `build_delta_net_autoregressive` (single-token generation)

### Numerical Correctness

- **All unary ops (softplus, exp, sigmoid) computed in FP32** regardless of tensor type ([`unary-ops.cpp:100-108`](https://github.com/ggml-org/llama.cpp/blob/723c71064da0908c19683f8c344715fbf6d986fd/ggml/src/ggml-cpu/unary-ops.cpp#L100-L108))
- Softplus has overflow guard: `(x > 20.0f) ? x : logf(1.0f + expf(x))` ([`unary-ops.cpp:80`](https://github.com/ggml-org/llama.cpp/blob/723c71064da0908c19683f8c344715fbf6d986fd/ggml/src/ggml-cpu/unary-ops.cpp#L80))
- Gate values are always negative (softplus * -exp(A_log)), so cumulative sums are monotonically decreasing
- All `exp()` calls on gate-derived quantities produce values in [0, 1] — overflow is mathematically impossible
- The Python reference uses `torch.clamp(max=50.0)` which is absent in C++ but unnecessary since the exponent is always <= 0

### CUDA Kernels

- `ssm_conv` ([`ssm-conv.cu`](https://github.com/ggml-org/llama.cpp/blob/723c71064da0908c19683f8c344715fbf6d986fd/ggml/src/ggml-cuda/ssm-conv.cu)): Custom kernel for 1D convolution, supports kernel sizes 3, 4, 9
- `solve_tri` ([`solve_tri.cu`](https://github.com/ggml-org/llama.cpp/blob/723c71064da0908c19683f8c344715fbf6d986fd/ggml/src/ggml-cuda/solve_tri.cu)): Triangular solve using cuBLAS `cublasStrsmBatched` for large matrices, custom warp-based kernel for small matrices (n <= 64)

### Model Type Cosmetic Issue

The dense Qwen 3.5 architecture at [`llama-model.cpp:2530-2533`](https://github.com/ggml-org/llama.cpp/blob/723c71064da0908c19683f8c344715fbf6d986fd/src/llama-model.cpp#L2530-L2533) only recognizes 24-layer models:

```cpp
switch (hparams.n_layer) {
    case 24: type = LLM_TYPE_2B; break;
    default: type = LLM_TYPE_UNKNOWN;
}
```

The 27B model (64 layers) displays as `"qwen35 ?B Q4_K_M"`. **This is purely cosmetic** — confirmed by searching all uses of `model.type`: the only functional checks are for Baichuan 7B and LLaMA 70B quantization, neither of which applies to Qwen 3.5.

### GGUF Sampling Metadata

The llama.cpp library defines `general.sampling.*` GGUF metadata keys ([`constants.py:28-40`](https://github.com/ggml-org/llama.cpp/blob/723c71064da0908c19683f8c344715fbf6d986fd/gguf-py/gguf/constants.py#L28-L40)) and reads them in `common_params_sampling_apply_gguf()` ([`common.cpp:1008-1028`](https://github.com/ggml-org/llama.cpp/blob/723c71064da0908c19683f8c344715fbf6d986fd/common/common.cpp#L1008-L1028)). User CLI flags take priority.

The `convert_hf_to_gguf.py` conversion script reads `generation_config.json` from the HuggingFace model repository and embeds `temperature`, `top_k`, `top_p` into the GGUF file. However, `repetition_penalty` from HuggingFace does NOT match the expected key name `penalty_repeat`, so it is **not embedded**.

**The Ollama inference framework ignores all of this anyway** — zero references to `general.sampling.*` in the Ollama Go codebase.

### Chat Template

No Qwen 3.5-specific handler exists in the llama.cpp library. Detection falls through to either:
- **Qwen3-Coder** handler (if template contains `<tool_call>`, `<function=`, `<parameter=`) — this is the correct match for Qwen 3.5
- **Hermes 2/3 Pro** handler (if template contains `<tool_call>` without the XML format)

Both handlers support `<think>` tag processing and `enable_thinking` parameter. The llama.cpp library also had a template issue ([llama.cpp issue #19872](https://github.com/ggml-org/llama.cpp/issues/19872)) where `tool_call.arguments|items` fails when arguments arrive as a string rather than a dict — fixed in [llama.cpp PR #19635](https://github.com/ggml-org/llama.cpp/pull/19635).

---

## 14. Pinned llama.cpp Version in the Ollama Inference Framework

**Ollama inference framework v0.17.1 pins the llama.cpp C++ library at commit [`ec98e2002`](https://github.com/ggml-org/llama.cpp/tree/ec98e2002)** (from [`Makefile.sync`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/Makefile.sync) and [`llama/build-info.cpp`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/llama/build-info.cpp)).

The Ollama team carries **34 patches** on top ([`llama/patches/`](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/llama/patches)), including:
- `0027-interleave-multi-rope.patch` — relevant to Qwen 3.5's IMRoPE (Interleaved Multi-scale Rotary Position Embedding)
- `0033-ggml-metal-solve_tri.patch` — Apple Metal GPU backend fix for DeltaNet triangular solve

The CUDA multi-GPU crash fix ([llama.cpp PR #19866](https://github.com/ggml-org/llama.cpp/pull/19866)) appears included since Ollama v0.17.1-rc2 users confirm the 27B model works.

---

## 15. Summary: Is Everything Truly Correct?

### vLLM Inference Server

| Aspect | Verdict | Detail |
|--------|---------|--------|
| Model loads | **Nightly only** | Stable vLLM v0.16.0 fails; nightly required |
| temperature | **Must set manually** | No defaults injected |
| top_k | **Must set manually** | No defaults injected |
| top_p | **Must set manually** | No defaults injected |
| presence_penalty | **Works if set** | vLLM inference server implements it correctly |
| Tool calling format | **CORRECT** | `qwen3_coder` parser matches model's trained XML format |
| Tool calling stability | **Buggy** | Multiple open issues ([#21711](https://github.com/vllm-project/vllm/issues/21711), [#20611](https://github.com/vllm-project/vllm/issues/20611), [#19051](https://github.com/vllm-project/vllm/issues/19051), [#29192](https://github.com/vllm-project/vllm/issues/29192)) |
| Reasoning parser | **Known bug** | Truncated reasoning misclassified as content ([#35221](https://github.com/vllm-project/vllm/issues/35221)) |
| DeltaNet implementation | **Complete** | Triton kernels, dedicated attention backend |
| MTP speculative decoding | **Supported** | `Qwen3_5MTP` registered |

### Ollama Inference Framework

| Aspect | Verdict | Detail | Bug Report? |
|--------|---------|--------|-------------|
| Model loads and runs | **YES** | Confirmed on Ollama v0.17.1-rc2 ([Ollama issue #14433](https://github.com/ollama/ollama/issues/14433)) | — |
| temperature=1.0 | **CORRECT** | Baked in Ollama registry params blob, applied by Go-based sampler | — |
| top_k=20 | **CORRECT** | Baked in, applied | — |
| top_p=0.95 | **CORRECT** | Baked in, applied | — |
| min_p=0.0 | **CORRECT** | Default, applied | — |
| presence_penalty | **BROKEN** | Ollama Go-based runner silently ignores it ([Section 7](#7-ollama-critical-bug-missing-penalty-sampling)) | **YES** |
| repeat_penalty | **BROKEN** | Ollama Go-based runner silently ignores it | **YES** |
| frequency_penalty | **BROKEN** | Ollama Go-based runner silently ignores it | **YES** |
| Registry renderer/parser | **WRONG** | Ollama registry config blob has `renderer: "qwen3.5"` (JSON) + `parser: "qwen3.5"` (JSON) — should be `"qwen3-coder"` (XML) ([Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)) | **YES** |
| Tool system prompt | **WRONG** | 6 mismatches vs HuggingFace ground truth template: wrong format instruction, wrong header, missing `<IMPORTANT>` block, wrong system message position ([Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)) | **YES** |
| Tool call rendering in history | **WRONG + BROKEN** | Uses JSON format AND fails to close `</think>` tag before tool calls ([Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)) | **YES** |
| Tool calling parser | **WRONG** | `Qwen3Parser` (JSON via `json.Unmarshal`) instead of `Qwen3CoderParser` (XML via `xml.Unmarshal`) ([Section 8](#8-ollama-critical-bug-tool-calling-format-mismatch)) | **YES** |
| Stop tokens | **Adequate** | Params blob has no string-based `stop` sequences, but EOS token IDs in the GGUF file (`<\|im_end\|>` 248046, `<\|endoftext\|>` 248044) handle turn-boundary stopping correctly ([Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)) | No |
| Non-tool prompt format | **CORRECT** | Matches HuggingFace ground truth template byte-for-byte for simple chat (no tools) | — |
| Thinking prefill | **CORRECT** | Both enabled (`<think>\n`) and disabled (`<think>\n\n</think>\n\n`) match HuggingFace ground truth | — |
| think=false | **BROKEN (OK)** | Does not disable thinking; this is desirable for agent use | No |
| Context length default | **CORRECT** | 32,768 tokens for 32 GB VRAM | — |
| DeltaNet numerics | **CORRECT** | FP32 unary ops, no overflow possible | — |
| DeltaNet implementation | **CORRECT** | Full chunked + autoregressive in Go and C++ | — |
| Parallel requests | **Limited** | Forced to 1 (expected architectural limitation of hybrid recurrent state) | No |
| GGUF sampling metadata | **IGNORED** | Ollama inference framework reads only the registry params blob, ignores GGUF `general.sampling.*` keys | — |

### Bottom Line

**For agent coding use cases, the Ollama inference framework v0.17.1 has three layers of critical bugs that make Qwen 3.5 27B tool calling non-functional:**

> These three bugs are the verified, source-linked findings that belong in a bug report to the Ollama team. See the [Bug Report Summary](#bug-report-summary) at the top of this document for a self-contained summary with all links.

1. **BUG REPORT — Missing penalty sampling** (Ollama Go-based runner limitation) — The Qwen 3.5 model's recommended `presence_penalty=1.5` is silently accepted by the Ollama API and silently ignored by the Go-based runner's sampler. Extended thinking sequences may enter repetition loops with no API-level workaround available.

2. **BUG REPORT — Wrong tool calling format** (wiring bug in Ollama source code + Ollama model registry config blob) — The registry config blob sets `renderer: "qwen3.5"` and `parser: "qwen3.5"`, which map to the Qwen 3 Hermes-style JSON pipeline. The Qwen 3.5 model was trained on the Qwen3-Coder XML format. The correct `Qwen3CoderParser` + `Qwen3CoderRenderer` already exist in the Ollama codebase but are wired to `"qwen3-coder"` instead of `"qwen3.5"`.

3. **BUG REPORT — Unclosed `</think>` tag in multi-turn tool call history** (renderer bug in Ollama source code) — The `Qwen3VLRenderer` fails to close `</think>` tags before tool calls when the assistant message has empty content, placing `<tool_call>` blocks inside unclosed thinking blocks and corrupting the conversation structure the model sees.

**For simple chat without tools, the Ollama inference framework works correctly.** The prompt format matches the HuggingFace ground truth template byte-for-byte, the sampling parameters (temperature, top_k, top_p) are correct, and thinking mode works. The bugs are specific to tool calling and penalty sampling.

**The vLLM inference server is the only viable path for agent/tool use** despite requiring a nightly build. It has working penalty sampling, the correct tool call parser (`qwen3_coder` matching the model's XML format), and proper thinking mode. Tool calling has bugs but the format is right — bugs are fixable, wrong format is architectural.

**Neither library auto-applies the Qwen 3.5 model's recommended defaults** — you must set parameters explicitly in both cases.

**Thinking mode always-on is correct for agent use.** The Qwen 3.5 model's agentic training (1 million reinforcement learning environments, Terminal-Bench 52.5) assumes thinking mode. Disabling it removes the model's strongest capability for multi-step tool use.

---

## 16. Cloned Repositories

| Repository | Local Path | Version | Commit | Link |
|------|------|---------|--------|------|
| vLLM inference server | `/tmp/vllm` | nightly (post-v0.16.1rc0) | `a1f53addb` | [vllm-project/vllm@a1f53ad](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329) |
| Ollama inference framework | `/tmp/ollama` | v0.17.1 | `9bf41969f` | [ollama/ollama@9bf4196](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a) |
| llama.cpp C++ inference library (upstream) | `/tmp/llama.cpp` | latest | `723c71064` | [ggml-org/llama.cpp@723c710](https://github.com/ggml-org/llama.cpp/tree/723c71064da0908c19683f8c344715fbf6d986fd) |
| llama.cpp C++ inference library (Ollama-pinned) | `/tmp/ollama/ml/backend/ggml/ggml/` | — | `ec98e2002` + 34 patches | [ggml-org/llama.cpp@ec98e2002](https://github.com/ggml-org/llama.cpp/tree/ec98e2002) |

Note: `/tmp/llama.cpp` is the latest upstream llama.cpp, NOT the same as the Ollama-pinned version. The Ollama inference framework's vendored copy with patches is at `/tmp/ollama/ml/backend/ggml/ggml/`. The [34 patch files](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/llama/patches) are applied on top of the pinned commit.

---

## 17. Sources

### Qwen 3.5 27B Dense Language Model
- [Qwen/Qwen3.5-27B on HuggingFace model repository](https://huggingface.co/Qwen/Qwen3.5-27B) — model card with recommended parameters
- [Qwen/Qwen3.5-27B `tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) — contains the Jinja2 chat template defining the XML tool call format (the ground truth for what the model was trained on)
- [Qwen/Qwen3.5-27B-FP8 on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B-FP8)
- [unsloth/Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF)
- [bartowski/Qwen_Qwen3.5-27B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-27B-GGUF)
- [QwenLM/Qwen3.5 GitHub](https://github.com/QwenLM/Qwen3.5)
- [Official Qwen 3.5 Blog Post](https://qwen.ai/blog?id=qwen3.5)
- [Qwen Function Calling Docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html) — covers Qwen3 format; not yet updated for Qwen 3.5
- [Qwen3.5 Architecture Blog (HuggingFace)](https://huggingface.co/blog/mlabonne/qwen35)

### vLLM Inference Server (pinned to commit [`a1f53ad`](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329))
- [vLLM GitHub repository](https://github.com/vllm-project/vllm)
- [vLLM GitHub Releases](https://github.com/vllm-project/vllm/releases)
- [vLLM Qwen3.5 Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [Issue #35391 — Architecture Not Supported in v0.16.0](https://github.com/vllm-project/vllm/issues/35391)
- [Issue #35221 — Reasoning Parser Bug](https://github.com/vllm-project/vllm/issues/35221)
- [Issue #21711 — Tool Call XML Parsing](https://github.com/vllm-project/vllm/issues/21711)
- [Issue #20611 — Streaming Tool Calls](https://github.com/vllm-project/vllm/issues/20611)
- [Issue #19051 — Reasoning + Tool Calling Conflict](https://github.com/vllm-project/vllm/issues/19051)
- [Issue #29192 — Parser Infinite Stream](https://github.com/vllm-project/vllm/issues/29192)
- [Security Advisory GHSA-79j6-g2m3-jgfw — `eval()` in qwen3_coder parser](https://github.com/vllm-project/vllm/security/advisories/GHSA-79j6-g2m3-jgfw)

### Ollama Inference Framework (pinned to [`v0.17.1` / `9bf4196`](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a))
- [Ollama model library — Qwen 3.5](https://ollama.com/library/qwen3.5)
- [Ollama model library — Qwen 3.5 27B Q4_K_M](https://ollama.com/library/qwen3.5:27b-q4_K_M)
- [Ollama GitHub repository](https://github.com/ollama/ollama)
- [Ollama GitHub releases](https://github.com/ollama/ollama/releases)
- [Issue #14433 — 27B Works, 35B/122B Fail](https://github.com/ollama/ollama/issues/14433)
- [Issue #14418 — Thinking/Language/Memory Bugs](https://github.com/ollama/ollama/issues/14418)
- [Issue #14416 — Model Request for 27B](https://github.com/ollama/ollama/issues/14416)
- [Issue #11135 — Tool Hallucination (Qwen 3 era, JSON format)](https://github.com/ollama/ollama/issues/11135)
- [Issue #11662 — Tool Call Parsing (Qwen 3 era, JSON format)](https://github.com/ollama/ollama/issues/11662)
- [Qwen Ollama Documentation](https://qwen.readthedocs.io/en/latest/run_locally/ollama.html)

### Ollama Model Registry (pulled via `registry.ollama.ai` API)
- Manifest: `GET https://registry.ollama.ai/v2/library/qwen3.5/manifests/27b-q4_K_M`
- Config blob: `sha256:2c5b293a8ebc073edf7422368891c296fcde68d5c8babc65878d4f3a072fa06a` (476 bytes) — contains `renderer: "qwen3.5"`, `parser: "qwen3.5"` (the wrong pipeline)
- Params blob: `sha256:f6417cb1e26962991f8e875a93f3cb0f92bc9b4955e004881251ccbf934a19d2` (42 bytes: `{"temperature":1,"top_k":20,"top_p":0.95}`)
- Comparison: `qwen3-coder` params blob: `{"repeat_penalty":1.05,"stop":["<|im_start|>","<|im_end|>","<|endoftext|>"],"temperature":0.7,"top_k":20,"top_p":0.8}`
- Comparison: `qwen3` params blob: `{"repeat_penalty":1,"stop":["<|im_start|>","<|im_end|>"],"temperature":0.6,"top_k":20,"top_p":0.95}`

### HuggingFace Model Repository Ground Truth
- [Qwen3.5-27B `tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) — contains the Jinja2 chat template (7,757 bytes) defining the XML tool call format, `<IMPORTANT>` block, and system message ordering. This is the ground truth for what the model was trained on.
- [Qwen3.5-27B `generation_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/generation_config.json) — EOS (End of Sequence) token IDs: [248046 (`<|im_end|>`), 248044 (`<|endoftext|>`)], temperature=0.6 (precise coding preset)

### Key Ollama Inference Framework Source Files (at commit `9bf4196`)
- [`model/renderers/renderer.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/renderer.go) — renderer registry, `"qwen3.5"` wiring at line 59 (maps to wrong `Qwen3VLRenderer`)
- [`model/renderers/qwen3vl.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go) — `Qwen3VLRenderer` (Hermes-style JSON format, incorrectly used for qwen3.5)
- [`model/renderers/qwen3coder.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3coder.go) — `Qwen3CoderRenderer` (XML format, correct for qwen3.5 but not wired to it)
- [`model/parsers/parsers.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/parsers.go) — parser registry, `"qwen3.5"` wiring at line 52 (maps to wrong `Qwen3Parser`)
- [`model/parsers/qwen3.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3.go) — `Qwen3Parser` (JSON parser via `json.Unmarshal`, incorrectly used for qwen3.5)
- [`model/parsers/qwen3coder.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3coder.go) — `Qwen3CoderParser` (XML parser via `xml.Unmarshal`, correct for qwen3.5 but not wired to it)
- [`sample/samplers.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/sample/samplers.go) — Go-based sampler, missing all penalty parameters (repeat, presence, frequency)
- [`runner/ollamarunner/runner.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/ollamarunner/runner.go) — Ollama Go-based runner, invokes sampler without passing penalty parameters from API options
- [`runner/llamarunner/runner.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/llamarunner/runner.go) — Ollama C++-based runner (for comparison), correctly passes all penalty parameters to llama.cpp
- [`fs/ggml/ggml.go`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/fs/ggml/ggml.go) — `OllamaEngineRequired()` function, forces Qwen 3.5 (`"qwen35"`) onto the Go-based runner

### llama.cpp C++ Inference Library
- [llama.cpp GitHub repository](https://github.com/ggml-org/llama.cpp)
- [Ollama-pinned commit `ec98e2002`](https://github.com/ggml-org/llama.cpp/tree/ec98e2002) with [34 Ollama patches](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/llama/patches)
- [Upstream commit `723c710` (latest tested)](https://github.com/ggml-org/llama.cpp/tree/723c71064da0908c19683f8c344715fbf6d986fd)
- [llama.cpp PR #19468 — Qwen 3.5 Support (Merged)](https://github.com/ggml-org/llama.cpp/pull/19468)
- [Issue #19860 — CUDA Multi-GPU Crash (Fixed)](https://github.com/ggml-org/llama.cpp/issues/19860)
- [PR #19866 — CUDA Multi-GPU Crash Fix](https://github.com/ggml-org/llama.cpp/pull/19866)
- [Issue #19869 — Chat Parsing Crash](https://github.com/ggml-org/llama.cpp/issues/19869)
- [Issue #19872 — Template `arguments|items` Filter Failure](https://github.com/ggml-org/llama.cpp/issues/19872)
- [PR #19635 — Template Detection Fix](https://github.com/ggml-org/llama.cpp/pull/19635)

### News Coverage
- [MarkTechPost — Qwen 3.5 Medium Series](https://www.marktechpost.com/2026/02/24/alibaba-qwen-team-releases-qwen-3-5-medium-model-series-a-production-powerhouse-proving-that-smaller-ai-models-are-smarter/)
- [VentureBeat — Qwen 3.5](https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17-beats-its-larger-trillion-parameter-model-at-a)
- [DataCamp — Qwen 3.5 Guide](https://www.datacamp.com/blog/qwen3-5)
