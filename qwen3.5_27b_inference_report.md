# Qwen 3.5 27B Dense Language Model — Inference Library Support Report

**Date:** March 1, 2026 UTC (GGUF comparison added; original report February 27, 2026 UTC)
**Target Hardware:** NVIDIA RTX 5090 graphics card (32 GB VRAM — Video Random Access Memory)
**Model:** Qwen 3.5 27B dense language model (all 27 billion parameters active every forward pass)
**Inference Libraries Evaluated:** vLLM inference server, Ollama inference framework (+ llama.cpp C++ backend)

### Versions Pinned in This Report

| Component | Version | Commit | Link |
|-----------|---------|--------|------|
| Ollama inference framework (original analysis) | v0.17.1 | `9bf4196` | [ollama/ollama@9bf4196](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a) |
| Ollama inference framework (re-verified) | post-v0.17.4 (Feb 26, 2026 UTC) | `79917cf` | [ollama/ollama@79917cf](https://github.com/ollama/ollama/tree/79917cf80bf74538a4ae694e6b61adb908b0f8df) |
| Ollama inference framework (fork base for fixes) | v0.17.4 | `cc90a035` | [ollama/ollama@cc90a03](https://github.com/ollama/ollama/tree/cc90a035a0cc3ae9bd0c1dc95d42b620e8dcb0e2) |
| llama.cpp C++ inference library (Ollama-pinned) | — | `ec98e2002` | [ggml-org/llama.cpp@ec98e2002](https://github.com/ggml-org/llama.cpp/tree/ec98e2002) + [34 Ollama patches](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/llama/patches) |
| llama.cpp C++ inference library (upstream) | post-b5136 (Feb 26, 2026 UTC) | `723c710` | [ggml-org/llama.cpp@723c710](https://github.com/ggml-org/llama.cpp/tree/723c71064da0908c19683f8c344715fbf6d986fd) |
| vLLM inference server (stable) | v0.16.0 (Feb 25, 2026 UTC) | — | [vllm-project/vllm v0.16.0](https://github.com/vllm-project/vllm/releases/tag/v0.16.0) — **does NOT support Qwen 3.5** (branch cut Feb 8, 2026 UTC, before Qwen 3.5 series launch Feb 16, 2026 UTC / 27B release Feb 24, 2026 UTC) |
| vLLM inference server (nightly, used in this report) | nightly (post-v0.16.1rc0) | `a1f53ad` | [vllm-project/vllm@a1f53ad](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329) |
| vLLM Docker image (Blackwell + Qwen 3.5) | `vllm/vllm-openai:qwen3_5-cu130` | — | [Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags) — CUDA 13.0.1, sm_120 kernels, published Feb 23, 2026 UTC |
| vLLM Docker image (Blackwell nightly) | `vllm/vllm-openai:cu130-nightly` | — | [Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags) — CUDA 13.0.1, sm_120 kernels, updated daily |
| Ollama registry GGUF (full file download) | `qwen3.5:27b-q4_K_M` | `sha256:7935de6e…` | 17,420,420,832 bytes, 1,307 tensors — standard Q4_K_M, includes vision encoder + MTP head |
| Unsloth GGUF Q4_K_M (full file download) | [`Qwen3.5-27B-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | — | 16,740,812,160 bytes, 851 tensors — custom imatrix + GDN/SSM layer protection, text-only |
| Unsloth GGUF UD-Q4_K_XL (full file download) | [`Qwen3.5-27B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) | — | 851 tensors — Unsloth Dynamic 2.0, Feb 24 upload (pre-fix), same size as Q4_K_M but different SSM layer tradeoffs |

---

## Effort Comparison: Fixing Qwen 3.5 27B Support (Ollama vs vLLM)

> **As of February 27, 2026 UTC.** This section compares the engineering effort required to achieve correct, fully-functional Qwen 3.5 27B inference on each framework, measured against the latest stable release of each.

### Baseline

| Framework | Latest Stable (as of Feb 27, 2026 UTC) | Qwen 3.5 Support? |
|-----------|----------------------------------------|-------------------|
| **vLLM inference server** | v0.16.0 (branch cut Feb 8, 2026 UTC) | **None** — series launched Feb 16, 27B released Feb 24, both after branch cut |
| **Ollama inference framework** | v0.17.4 (Feb 26, 2026 UTC) | Yes, but with 4 critical bugs |

### vLLM: Zero Code Fixes Needed — But Nightly-Only With Known Issues

vLLM's stable v0.16.0 has zero Qwen 3.5 support. The nightly (commit [`a1f53ad`](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329)) has full support with **correct code** for all the areas where Ollama is broken:

| Area | Status in vLLM nightly |
|------|----------------------|
| Chat template | **Correct** — uses HuggingFace Jinja2 template verbatim (no reimplementation) |
| Tool calling format | **Correct** — `qwen3_coder` parser handles Qwen3-Coder XML format |
| Reasoning parser | **Correct** — clean two-phase handoff to tool parser ([Section 4](#4-vllm-support-status)) |
| Penalty sampling | **Correct** — `presence_penalty`, `frequency_penalty`, `repetition_penalty` all functional |
| `</think>` closure | **Correct** — handled by HuggingFace template, not framework code |
| Generation prompt | **Correct** — HuggingFace Jinja2 template unconditionally appends generation prompt after message loop |
| Default parameters | **Not provided** — must set `temperature`, `top_p`, `top_k`, `presence_penalty` per-request (convenience gap, not a correctness bug) |
| Quantization for 32 GB VRAM | **FP8 is the only well-tested option** (~28 GB model, ~4 GB KV, ~64K tokens). Community AWQ-4bit checkpoints exist but are [barely tested](#deep-dive-quantization-on-32-gb-vram): one TP=2 success, one failure with garbled output, zero single-GPU reports. No official Qwen 4-bit checkpoint exists. |

**However, "correct code" does not mean "production-ready."** The Qwen 3.5 27B model was released February 24, 2026 UTC (as part of the Medium Series, 8 days after the 397B-A17B flagship launched the Qwen 3.5 series on February 16, 2026 UTC) — only 3 days before this report. Known issues as of February 27, 2026 UTC:

| Issue | Severity | Reference |
|-------|----------|-----------|
| Reasoning parser truncation (non-streaming): if `max_tokens` cuts generation mid-`<think>`, reasoning text is misclassified as content | Moderate (streaming unaffected) | [#35221](https://github.com/vllm-project/vllm/issues/35221) |
| `qwen3_coder` parser produces infinite "!" streams with long inputs | Moderate | [#29192](https://github.com/vllm-project/vllm/issues/29192) |
| Streaming + tool calls fails with `enable_thinking=false` | Moderate | [#20611](https://github.com/vllm-project/vllm/issues/20611) |
| `tool_choice: required` + reasoning parsing causes 400 errors | Moderate | [#19051](https://github.com/vllm-project/vllm/issues/19051) |
| `json.loads` fails on XML `<tool_call>` tags in output | Low (use `qwen3_coder` parser, not default) | [#21711](https://github.com/vllm-project/vllm/issues/21711) |
| **RTX 5090 (Blackwell sm_120): Docker works, pip requires source build** | Low (Docker is available) | See below |

**RTX 5090 readiness:** vLLM v0.15.1 (February 4, 2026 UTC) explicitly fixed sm_120 kernel issues for RTX Blackwell workstation GPUs — NVFP4 MoE kernel support ([#33417](https://github.com/vllm-project/vllm/issues/33416)) and FP8 CUTLASS fallback to Triton on sm_120 ([#33285](https://github.com/vllm-project/vllm/issues/33416)). These fixes are included in v0.16.0. **Official Docker images include sm_120 compiled kernels:** the vLLM team publishes `-cu130` Docker tags (CUDA 13.0.1) that compile for `TORCH_CUDA_ARCH_LIST='7.0 7.5 8.0 8.9 9.0 10.0 12.0'` — the `12.0` entry is sm_120 (Blackwell). A user on [GitHub Issue #35303](https://github.com/vllm-project/vllm/issues/35303) confirmed `vllm/vllm-openai:latest-cu130` running on RTX 5090 (driver 590.48, PyTorch 2.10.0+cu130). The `-cu130` tags are available for stable releases (`v0.16.0-cu130`, `v0.15.1-cu130`) and nightly (`cu130-nightly`). There is also a purpose-built `qwen3_5-cu130` tag published February 23, 2026 UTC. **`pip install vllm` does NOT include sm_120 kernels** — pip requires building from source with `TORCH_CUDA_ARCH_LIST="12.0"` plus manually compiling xFormers and FlashInfer. Docker is the recommended path for RTX 5090. Flash Attention 3 does not work on Blackwell — the Docker images use `VLLM_FLASH_ATTN_VERSION=2` (FA2 runs fine on sm_120, confirmed by the FA maintainer Tri Dao in [flash-attention#1853](https://github.com/Dao-AILab/flash-attention/issues/1853)). Flash Attention 4, designed specifically for Blackwell, targets sm_100 (data center B200/B300), not sm_120 (consumer RTX), and is not released in the `flash-attn` PyPI package as of February 27, 2026 UTC (latest release is v2.8.3 from August 2025). Dynamic FP8 quantization is reported slower than BF16 on RTX 5090 ([#28234](https://github.com/vllm-project/vllm/issues/28234)) — use the pre-quantized [`Qwen/Qwen3.5-27B-FP8`](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) checkpoint instead of online quantization.

**Deployment path for this specific target:** Deploying Qwen 3.5 27B on RTX 5090 via vLLM as of February 27, 2026 UTC requires the nightly build (because stable v0.16.0 doesn't include Qwen 3.5 support). The recommended path is Docker: `vllm/vllm-openai:cu130-nightly` (or the pinned `qwen3_5-cu130` tag from February 23, 2026 UTC). Docker image SHA digests provide reproducible version pins (`docker pull vllm/vllm-openai:cu130-nightly@sha256:<digest>`). The official [Qwen 3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) explicitly recommends `vllm/vllm-openai:cu130-nightly` for Blackwell GPUs with the note "Use vLLM nightly wheel until 0.17.0 is released." Note: the recipe covers only the 397B-A17B MoE model, not the 27B dense model specifically.

**Effort to fix the code: 0 lines.** The template, tool format, penalty sampling, and reasoning parser are all correct in the nightly. The "fix" is waiting for v0.17.0 (the next stable release, which will include both Qwen 3.5 support and the sm_120 kernel fixes). The RTX 5090 packaging gap and open bugs above are real operational concerns but not code-correctness bugs — they affect the deployment environment and edge cases, not the fundamental template/format/sampling pipeline.

### Ollama: ~400–800 Lines of Go Across 4 Bugs + Registry Update

Four verified bugs need fixing, each at a different layer of the Ollama codebase:

**Bug 1 — Missing penalty sampling (~200–400 lines, high blast radius):**
The Go runner's `Sampler` struct has no penalty fields. `NewSampler()` doesn't accept them. `sample()` doesn't apply them. Implementing this from scratch requires: adding `repeatPenalty`, `presencePenalty`, `frequencyPenalty`, and `repeatLastN` fields to the struct, a new `applyPenalties()` function that builds a frequency map from a ring buffer of recently generated tokens and adjusts logits (repeat penalty is multiplicative — divides positive logits, multiplies negative; frequency penalty subtracts proportional to count; presence penalty subtracts flat), integrating this into `Sample()` before both the initial sample and the grammar-fallback resample with a no-op fast path when all penalties are neutral (`repeatPenalty == 1.0 && frequencyPenalty == 0.0 && presencePenalty == 0.0`), updating all `NewSampler()` call sites (at minimum [`runner/ollamarunner/runner.go:890-897`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/ollamarunner/runner.go#L890-L897) plus ~10 test call sites), and writing tests. This touches the hot path of token generation for **every model on the Go runner**, not just Qwen 3.5.

**Bug 2 — Tool calling format mismatch (~150–300 lines, Qwen 3.5-only):**
The correct `Qwen3CoderRenderer`/`Qwen3CoderParser` pipeline exists but lacks thinking support — a blocking gap that makes a simple rewire insufficient. Specifically:
- **Renderer** (`Qwen3CoderRenderer` at [`qwen3coder.go:58-193`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3coder.go#L58-L193)): Has no `isThinking` or `emitEmptyThinkOnNoThink` fields. The `Render()` method ignores the `*api.ThinkValue` parameter (named `_`). No `<think>` blocks are emitted in assistant messages. No thinking prefill (`<think>\n` or `<think>\n\n</think>\n\n`) is appended to the generation prompt. Fix requires: adding both fields, resolving thinking state from `think` parameter at top of `Render()`, computing `lastQueryIndex` (the last `Role == "user"` message) to match the HuggingFace template's `last_query_index` convention, emitting `<think>\n{thinking}\n</think>\n\n` before content/tool calls only for assistant messages in the current round (`i > lastQueryIndex`), and adding the thinking/no-thinking prefill at the end.
- **Parser** (`Qwen3CoderParser` at [`qwen3coder.go:31-194`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3coder.go#L31-L194)): `HasThinkingSupport()` returns `false` ([line 42](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3coder.go#L42)), so `<think>` blocks are never extracted. The parser state machine has only two states (`LookingForToolStart`, `CollectingToolContent`). Fix requires: adding two new states (`CollectingThinking`, `ThinkingDoneTransition`), adding `hasThinkingSupport`, `defaultThinking`, and `maybeThinkingOpenAtBOL` fields, extending `Init()` to set the initial state based on thinking enablement (following the `Qwen3Parser.Init()` pattern from [`qwen3.go:55-73`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3.go#L55-L73)), extending `Add()` to return thinking content separately from content, and extending the `eat()` function with thinking states that strip the optional leading `<think>` tag, handle `</think>` and `<tool_call>` detection with overlap handling for streaming, and transition to the existing tool-parsing states. The `qwenEventThinkingContent` type already exists in [`qwen3vl.go:63-67`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3vl.go#L63-L67) and is package-accessible.
- **Wiring**: Update the switch statements in [`renderer.go:59-61`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/renderer.go#L59-L61) and [`parsers.go:52-53`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/parsers.go#L52-L53) to wire `"qwen3.5"` to `&Qwen3CoderRenderer{isThinking: true, emitEmptyThinkOnNoThink: true}` and `&Qwen3CoderParser{hasThinkingSupport: true, defaultThinking: true}` respectively. The registry config blob at `registry.ollama.ai` also needs updating but that is Ollama-team-only.

**Bug 3 — Unclosed `</think>` tag (~5–20 lines, may become moot):**
One-line logic fix at [`qwen3vl.go:98-99`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go#L98-L99) plus test updates. If Bug 2 is fixed by switching `"qwen3.5"` away from `Qwen3VLRenderer` entirely, this becomes moot *for Qwen 3.5* (though it still affects other models using `Qwen3VLRenderer` with thinking + tool calls).

**Bug 4 — Missing generation prompt after assistant tool call turns (~1 line, affects agentic robustness):**
In both `Qwen3VLRenderer` and `Qwen3CoderRenderer`, the `prefill` variable is defined as `lastMessage && message.Role == "assistant"` — treating ALL last-position assistant messages as text continuations (prefills), regardless of whether they have tool calls. An assistant message with tool calls is a **complete turn** (the model already finished generating), not a partial text to extend. The HuggingFace Jinja2 chat template always closes assistant turns with `<|im_end|>` and unconditionally appends the generation prompt (`<|im_start|>assistant\n<think>\n`) after the message loop. Without this fix, the model has no open assistant turn to generate into when the conversation ends with an assistant tool call message — a scenario that arises during cancellation, timeout, and error recovery in agentic coding flows. See [Section 9.5](#95-ollama-critical-bug-missing-generation-prompt-after-tool-call-turns) for full analysis.

**Registry params blob (trivial, blocked by Bug 1):**
Should set `presence_penalty: 1.5` per the [model card](https://huggingface.co/Qwen/Qwen3.5-27B), but pointless until Bug 1 is fixed since the Go runner silently ignores the parameter.

### Summary

| Dimension | Ollama (v0.17.1–v0.17.4) | vLLM (v0.16.0) |
|-----------|--------------------------|-----------------|
| Model supported at all? | Yes, with 4 critical bugs | No (nightly has full support) |
| Lines of code to fix | ~400–800 lines of Go | 0 (correct in nightly) |
| Number of distinct fixes | 4 bugs + 1 registry update | 0 |
| Highest-stakes fix | Penalty sampling (all Go-runner models) | N/A |
| Who can fix it? | Only Ollama team (source code + registry) | Already done by vLLM contributors |
| User workaround available? | None for any of the 4 bugs | Use nightly build |
| RTX 5090 ready? | Yes (runs today, bugs aside) | Docker `-cu130` images include sm_120 kernels; `pip install` requires source build |
| Nightly/edge-case bugs? | The 4 bugs above | 5 open issues ([#35221](https://github.com/vllm-project/vllm/issues/35221), [#29192](https://github.com/vllm-project/vllm/issues/29192), [#20611](https://github.com/vllm-project/vllm/issues/20611), [#19051](https://github.com/vllm-project/vllm/issues/19051), [#21711](https://github.com/vllm-project/vllm/issues/21711)) |
| Architectural root cause | Compiled-in Go renderers must be manually kept in sync per model | HuggingFace Jinja2 template used verbatim — correct by design |

### Bottom Line

**As of February 27, 2026 UTC:** vLLM requires zero code fixes — the template, tool format, and penalty sampling are all correct in the nightly and will land in the next stable release (v0.17.0, per the [official Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)). Official Docker images with sm_120 (Blackwell) kernels are available: `vllm/vllm-openai:cu130-nightly` for the latest nightly, or the pinned `qwen3_5-cu130` tag from February 23, 2026 UTC. Docker eliminates the source-build requirement — no need to compile vLLM, xFormers, or FlashInfer manually (`pip install` still requires source build). **However, 4-bit quantization on this model is uncharted territory:** no official Qwen 4-bit checkpoint exists, the only community AWQ-4bit checkpoint ([`cyankiwi/Qwen3.5-27B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-4bit)) has exactly one success report (dual RTX 3090, TP=2, `vllm/vllm-openai:nightly`) and one failure with garbled output — with zero single-GPU reports and no published quality benchmarks. The official FP8 checkpoint (`Qwen/Qwen3.5-27B-FP8`, ~28 GB) is the only well-tested quantization, leaving ~4 GB for KV cache on 32 GB VRAM. The Qwen 3.5 27B model being only 3 days old means there are still open bugs in streaming, tool parsing edge cases, and reasoning truncation. Ollama requires substantial engineering effort across three separate subsystems (sampler, renderer/parser, registry), with the penalty sampling fix being particularly high-stakes since it affects every Go-runner model. The architectural difference is the root cause: vLLM delegates to the model publisher's Jinja2 template (automatic correctness), while Ollama reimplements each model's template in compiled Go source code (perpetual maintenance burden where every new model family risks these exact kinds of mismatches).

---

## Table of Contents

- [Effort Comparison](#effort-comparison-fixing-qwen-35-27b-support-ollama-vs-vllm) — comparative analysis of what it takes to fix each framework
- [Bug Report Summary](#bug-report-summary) — **start here** for the four verified, bug-report-worthy findings

1. [Model Overview](#1-model-overview)
2. [VRAM Fit and Quantization Options](#2-vram-fit-and-quantization-options) — includes [GGUF Deep Dive: Ollama Registry vs Unsloth](#gguf-deep-dive-ollama-registry-vs-unsloth-use-unsloth) (per-tensor comparison from full file downloads, GDN/SSM layer protection, vision encoder trade-off, imatrix calibration, Modelfile instructions) and [Q4_K_M vs UD-Q4_K_XL analysis](#q4_k_m-vs-ud-q4_k_xl-use-q4_k_m) (three-way tensor comparison showing Q4_K_M better protects SSM layers)
3. [Recommended Inference Parameters](#3-recommended-inference-parameters)
4. [vLLM Support Status](#4-vllm-support-status) — includes deep dive: quantization on 32 GB VRAM (4-bit checkpoint testing status), chat template architecture, `enable_thinking` passthrough, reasoning+tool parser handoff, tool call parser architecture, special token flags, model implementation
5. [Ollama Support Status](#5-ollama-support-status)
6. [Ollama Deep Dive: Parameter Flow Validation](#6-ollama-deep-dive-parameter-flow-validation)
7. [Ollama Critical Bug: Missing Penalty Sampling](#7-ollama-critical-bug-missing-penalty-sampling)
8. [Ollama Critical Bug: Tool Calling Format Mismatch](#8-ollama-critical-bug-tool-calling-format-mismatch)
9. [Ollama Registry Model Validation: `qwen3.5:27b-q4_K_M`](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)
9.5. [Ollama Critical Bug: Missing Generation Prompt After Tool Call Turns](#95-ollama-critical-bug-missing-generation-prompt-after-tool-call-turns)
10. [Ollama Context Length](#10-ollama-context-length)
11. [Ollama Per-Use-Case Parameter Settings](#11-ollama-per-use-case-parameter-settings)
12. [Tool Calling Verdict](#12-tool-calling-verdict)
13. [llama.cpp Backend Validation](#13-llamacpp-backend-validation)
14. [Pinned llama.cpp Version in Ollama](#14-pinned-llamacpp-version-in-ollama)
15. [Summary: Is Everything Truly Correct?](#15-summary-is-everything-truly-correct)
16. [Source Code Versions Used](#16-source-code-versions-used)
17. [Sources](#17-sources)

---

## Bug Report Summary

The following four bugs in the Ollama inference framework were originally identified against v0.17.1 (commit [`9bf4196`](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a)) and **verified against source code, the Ollama model registry at `registry.ollama.ai`, and the HuggingFace model repository ground truth**. All four are unfixable by end users via API parameters or configuration — they require changes to Ollama's source code or registry model configuration by the Ollama team.

### Re-Verification Against Ollama Master (February 27, 2026 UTC)

All four bugs were re-verified against Ollama master at commit [`79917cf`](https://github.com/ollama/ollama/tree/79917cf80bf74538a4ae694e6b61adb908b0f8df) (February 26, 2026 UTC, one commit ahead of v0.17.4). Stable releases checked: v0.17.1 through v0.17.4.

| Bug | v0.17.1 | v0.17.2 | v0.17.3 | v0.17.4 | master (`79917cf`) |
|-----|---------|---------|---------|---------|-------------------|
| Bug 1: Missing penalty sampling | Present | Present | Present | Present | **Present** |
| Bug 2: Tool calling format mismatch | Present | Present | Present | Present | **Present** |
| Bug 3: Unclosed `</think>` tag (renderer) | Present | Present | Present | Present | **Present** |
| Bug 3 (parser-side mitigation) | — | — | **Fixed** | Fixed | Fixed |
| Bug 4: Missing generation prompt after tool calls | Present | Present | Present | Present | **Present** |

**Bug 3 partial fix detail:** Commit [`d98dda4`](https://github.com/ollama/ollama/commit/d98dda4676d44a3882fd38492cc00db257f35974) ("model: fix qwen3 tool calling in thinking", PR [#14477](https://github.com/ollama/ollama/pull/14477), merged February 26, 2026 UTC) added parser-side handling so that `Qwen3Parser` and `Qwen3VLParser` now detect `<tool_call>` while still in thinking-collection state, treating it as the end of thinking and transitioning to tool-call parsing. This fix entered the **v0.17.3** stable release. However, the **renderer side** of the bug — the `Qwen3VLRenderer` at [`qwen3vl.go:98-99`](https://github.com/ollama/ollama/blob/79917cf80bf74538a4ae694e6b61adb908b0f8df/model/renderers/qwen3vl.go#L98-L99) still gating `</think>` emission on `content != ""` — remains unfixed. Multi-turn prompts sent to the model still contain unclosed `<think>` tags when an assistant turn has thinking + tool calls + empty content. The tool-call thinking tests in `qwen3vl_thinking_test.go` (lines 119–323) remain commented out.

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

**Parser (FIXED in v0.17.3):** Commit [`d98dda4`](https://github.com/ollama/ollama/commit/d98dda4676d44a3882fd38492cc00db257f35974) (PR [#14477](https://github.com/ollama/ollama/pull/14477), February 26, 2026 UTC) fixed the `Qwen3Parser` and `Qwen3VLParser` to detect `<tool_call>` tags while still in thinking-collection state, correctly transitioning to tool-call parsing without requiring `</think>` first. This means model output like `<think>reasoning<tool_call>...</tool_call>` is now parsed correctly. This fix entered the v0.17.3 stable release.

- **Evidence:** [Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m) (Prompt Template Diff subsection)
- **Key source files:** [`model/renderers/qwen3vl.go:95-120`](https://github.com/ollama/ollama/blob/79917cf80bf74538a4ae694e6b61adb908b0f8df/model/renderers/qwen3vl.go#L95-L120) (renderer, still broken), [`model/parsers/qwen3.go:207-227`](https://github.com/ollama/ollama/blob/79917cf80bf74538a4ae694e6b61adb908b0f8df/model/parsers/qwen3.go#L207-L227) (parser, fixed)

### Bug 4: Missing Generation Prompt After Assistant Tool Call Turns

Both `Qwen3VLRenderer` and `Qwen3CoderRenderer` define `prefill` as `lastMessage && message.Role == "assistant"`, treating all last-position assistant messages as text continuations regardless of whether they contain tool calls. An assistant message with tool calls is a **complete turn** — the model already finished generating. The HuggingFace Jinja2 chat template unconditionally appends `<|im_start|>assistant\n<think>\n` after the message loop when `add_generation_prompt` is true, regardless of the last message type. Without the generation prompt, the model has no open assistant turn to generate into when the conversation ends with an assistant tool call message — a scenario that arises during agentic cancellation, timeout, and error recovery flows.

- **Evidence:** [Section 9.5](#95-ollama-critical-bug-missing-generation-prompt-after-tool-call-turns)
- **Key source files:** [`model/renderers/qwen3coder.go:148`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3coder.go#L148) (`prefill` definition), [`model/renderers/qwen3vl.go:85`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go#L85) (same pattern)
- **Ground truth:** [HuggingFace Jinja2 chat template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) — `add_generation_prompt` block fires unconditionally after message loop

### What Was Investigated But Does NOT Belong in the Bug Report

- **Missing stop tokens in params blob:** The `qwen3.5` params blob has no `stop` sequences, while `qwen3` and `qwen3-coder` do. However, this is **not a bug** — the Ollama Go-based runner has a separate token-ID-based EOS (End of Sequence) check ([`runner.go:768-775`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/runner/ollamarunner/runner.go#L768-L775)) that catches `<|im_end|>` (token 248046) and `<|endoftext|>` (token 248044) from the GGUF file metadata before any string matching occurs. Turn-boundary stopping works correctly without the `stop` field. See the disclaimer in [Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m).
- **Missing `presence_penalty` in params blob:** True but moot — the Go-based runner ignores it anyway (Bug 1 above).

---

## 1. Model Overview

**Released February 24, 2026 UTC.** Qwen 3.5 27B is a dense multimodal language model from Alibaba's Qwen team, part of the Qwen 3.5 Medium Model Series.

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

**Community 4-bit for vLLM (all use `compressed-tensors` or native AWQ format; see [testing details](#4-bit-checkpoint-testing-status) in Section 4):**
- [`cyankiwi/Qwen3.5-27B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-4bit) (20.1 GB, `compressed-tensors`, 10K downloads — **barely tested**: 1 TP=2 success, 1 garbled-output failure, 0 single-GPU reports)
- [`cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4`](https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4) (28 GB, `compressed-tensors`, untested — too large for practical 32 GB use)
- [`Kbenkhaled/Qwen3.5-27B-quantized.w4a16`](https://huggingface.co/Kbenkhaled/Qwen3.5-27B-quantized.w4a16) (18.6 GB, `compressed-tensors` / GPTQ, NVIDIA-affiliated author — brand new, untested, claims 100.3% accuracy recovery)
- [`QuantTrio/Qwen3.5-27B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.5-27B-AWQ) (21.9 GB, native AWQ, untested)

### NVIDIA RTX 5090 Graphics Card (32 GB VRAM) Fit Table

| Quantization | Size | VRAM Left | Quality | Recommendation |
|-------------|------|-----------|---------|---------------|
| BF16 | 53.8 GB | Won't fit | Perfect | Multi-GPU only |
| FP8 (official) | ~28 GB | ~4 GB (tight) | Near-perfect | Possible but tight KV cache |
| Q8_0 | 28.6 GB | ~3.4 GB | Excellent | Tight, short context only |
| **UD-Q6_K_XL** | **23.1 GB** | **~9 GB** | **Very high** | **Best quality/fit balance** |
| Q5_K_M | 19.4 GB | ~12.6 GB | High | Good for longer context |
| **Q4_K_M** | **16.5 GB** | **~15.5 GB** | Good | Most context headroom |

**Recommendation:** Use **Q4_K_M from Unsloth** (16.7 GB) for maximum context window with GDN/SSM layer protection — NOT the Ollama registry GGUF and NOT the UD-Q4_K_XL (see [Q4_K_M vs UD-Q4_K_XL analysis](#q4_k_m-vs-ud-q4_k_xl-use-q4_k_m) below). For best quality-to-fit ratio, use **Q6_K from Unsloth** (22.5 GB). See the [GGUF comparison](#gguf-deep-dive-ollama-registry-vs-unsloth-use-unsloth) below for why Unsloth over the Ollama registry.

### GGUF Deep Dive: Ollama Registry vs Unsloth — Use Unsloth

> **Verified March 1, 2026 UTC.** Both full GGUF files were downloaded and every tensor's quantization type was extracted and compared. This is not inference from documentation — it is ground truth from the actual model files.
>
> - Ollama registry: `sha256:7935de6e…` (17,420,420,832 bytes, 1,307 tensors)
> - Unsloth Q4_K_M: [`Qwen3.5-27B-Q4_K_M.gguf`](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) (16,740,812,160 bytes, 851 tensors)

**The Ollama registry GGUF uses standard llama.cpp Q4_K_M quantization. Unsloth's "Q4_K_M" is not standard — it systematically protects the GDN/SSM layers that define Qwen 3.5's novel hybrid architecture.** Both files say "Q4_K_M" on the label. The contents are meaningfully different in exactly the places that matter for model intelligence.

#### Per-Tensor Quantization Differences (198 of 803 common tensors differ)

| Tensor | Blocks | Ollama Q4_K_M | Unsloth Q4_K_M | What it does | Why it matters |
|--------|--------|---------------|----------------|--------------|----------------|
| **`attn_qkv`** | all 48 GDN blocks | Q4_K (4.5 bpw) | **Q5_K (5.5 bpw)** | Input projection for Gated DeltaNet linear attention | This IS the GDN mechanism. Quantizing it to 4-bit degrades the novel architecture that makes Qwen 3.5 different from a standard transformer. **+300 MB well spent.** |
| **`ssm_out`** | all 48 GDN blocks | Q4_K (4.5 bpw) | **Q5_K (5.5 bpw)** | Output projection from GDN state-space layer | Unsloth's KL divergence benchmarks found this tensor "dramatically increases KLD" when aggressively quantized. **+180 MB well spent.** |
| **`ssm_alpha`** | all 48 GDN blocks | Q4_K (4.5 bpw) | **Q8_0 (8.5 bpw)** | State-space decay parameter (controls how fast SSM memory fades) | Tiny tensor (245K params) but controls the SSM's temporal memory dynamics. Negligible size cost for 2× the precision. **+5.6 MB.** |
| **`ssm_beta`** | all 48 GDN blocks | Q4_K (4.5 bpw) | **Q8_0 (8.5 bpw)** | State-space input gate (controls what enters SSM memory) | Same story as `ssm_alpha` — tiny, sensitive, nearly free to protect. **+5.6 MB.** |
| `attn_v` | 6 of 16 attention blocks (35,39,47,51,59,63) | Q6_K (6.6 bpw) | Q4_K (4.5 bpw) | Value projection in standard attention layers | Unsloth downgrades 6 late attention blocks to partially fund the GDN upgrades. **-7.7 MB.** Acceptable trade — these are standard transformer attention layers, not the novel architecture. |

**Net size cost: +484 MB for the text model.** Unsloth's Q4_K_M is 484 MB larger than standard Q4_K_M *in text model weight alone*, because the GDN/SSM layers are kept at higher precision.

#### Tensors Never Quantized (F32 in both files, correctly)

Both files keep these at full 32-bit precision — as they should:

| Tensor | Count | Why F32 is correct |
|--------|-------|--------------------|
| All `*_norm.weight` | 128 | RMSNorm weights — tiny, quantization destroys layer normalization |
| `ssm_a` | 48 | State-space diagonal matrix — 48 params per block, quantization is pointless |
| `ssm_conv1d` | 48 | 1D convolution kernel — small, critical for SSM temporal processing |
| `ssm_dt` / `ssm_dt.bias` | 48 | Delta-time step bias — 48 params per block |

#### Independent Corroboration: Qwen's Own Guidance Says the Same Thing

Unsloth arrived at this recipe through KL divergence benchmarking of 150+ configurations. The official Qwen team's [`llm-compressor` guidance](https://huggingface.co/Qwen/Qwen3.5-27B) for vLLM quantization independently recommends: **`ignore=["re:.*linear_attn.*"]`** — skip all GDN layers during quantization entirely. Same conclusion, different methodology. The GDN layers are the most quantization-sensitive components in this architecture, and the standard Q4_K_M recipe in llama.cpp has no awareness of this.

#### Unsloth Also Uses a Custom Imatrix (Chat/Tool-Calling Calibration)

Standard GGUF quantization uses no importance matrix. Unsloth calibrates with a proprietary ~901K-token dataset (`unsloth_calibration_Qwen3.5-27B.txt`) focused on chat and tool-calling, not Wikipedia. This weights the quantization process toward conversational accuracy rather than generic perplexity, benefiting ALL tensors — not just the ones with explicit type overrides. The imatrix is available at [`imatrix_unsloth.gguf_file`](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/imatrix_unsloth.gguf_file) in the Unsloth repository.

#### What Ollama's GGUF Has That Unsloth's Doesn't

The Ollama registry GGUF (17.4 GB) contains 504 tensors not present in the Unsloth file (16.7 GB):

| Component | Extra tensors | Type | Size | Worth it? |
|-----------|---------------|------|------|-----------|
| **Vision encoder** | 441 | F16 weights, F32 biases | ~450 MB | **No.** Qwen 3.5 27B vision is a bolt-on ViT — it works, but on a 32 GB card every MB of VRAM spent on vision weights is a MB NOT available for KV cache. If you need vision, use a dedicated multimodal model or run vLLM with the separate mmproj file. For text intelligence, the vision encoder is dead weight. |
| **MTP head** | 15 | Q4_K/Q6_K/F32 | ~80 MB | **Marginal.** Multi-Token Prediction enables speculative decoding (faster inference, not smarter). Ollama does not use MTP as of v0.17.4. The tensors exist in the GGUF but are ignored by the runner. |
| **`ssm_dt` naming** | 48 | F32 | negligible | Naming difference: Ollama has `ssm_dt`, Unsloth has `ssm_dt.bias`. Functionally equivalent. |

**Verdict: The 648 MB size premium of the Ollama GGUF buys you an unused vision encoder and an unused MTP head. The 484 MB size premium of the Unsloth GGUF buys you measurably better GDN/SSM layer quality — the exact layers that define this model's intelligence advantage over standard transformers.**

#### Additional GGUF Metadata Differences

| Metadata | Ollama GGUF | Unsloth GGUF | Impact |
|----------|-------------|--------------|--------|
| `head_count_kv` | Per-layer array (64 values, 0 for GDN, 4 for attention) | Single value: 4 | Ollama's encoding is more correct for the hybrid architecture. llama.cpp resolves this via `full_attention_interval=4` — both work. |
| `ssm.v_head_reordered` | `True` | Missing | Indicates optimized SSM value head layout. Present in newer conversions. Verify inference output is coherent when using Unsloth GGUF. |
| `rope.mrope_interleaved` | `True` | Missing | Multi-scale RoPE interleaving flag. Again, should be inferred from architecture, but explicitly present only in Ollama's GGUF. |
| `eos_token_ids` | `[248046, 248044]` | `248046` only | Ollama's includes both `<\|im_end\|>` and `<\|endoftext\|>` as EOS. Unsloth only lists `<\|im_end\|>`. Not a problem in practice since the Go-based runner reads this from the GGUF directly. |
| Sampling defaults | Not embedded | `temp=0.6, top_k=20, top_p=0.95` | Unsloth embeds Qwen's "precise coding" defaults. Ollama's params blob overrides these anyway. |

#### How to Use the Unsloth GGUF With Ollama

Ollama natively supports HuggingFace Hub references — no manual download needed. The GGUF is pulled and cached automatically on first use ([docs](https://huggingface.co/docs/hub/en/ollama)).

Create a Modelfile:
```
FROM hf.co/unsloth/Qwen3.5-27B-GGUF:Q4_K_M
PARAMETER temperature 1.0
PARAMETER top_k 20
PARAMETER top_p 0.95
```

Import into Ollama:
```bash
ollama create qwen3.5-unsloth -f Modelfile
ollama run qwen3.5-unsloth
```

Ollama will detect the `qwen35` architecture from the GGUF metadata and use the `qwen3.5` renderer/parser (or your fixed versions if running the patched fork). No `RENDERER` directive needed in the Modelfile — the architecture detection handles it automatically.

#### Upstream Chat Template Bug in Both GGUFs (Irrelevant to Ollama)

Both the Ollama and Unsloth GGUFs for the 27B model contain the **same broken Jinja2 chat template** (7,756 bytes, byte-for-byte identical) from upstream [Qwen/Qwen3.5-27B `tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json). The bug: `tool_call.arguments|items` crashes llama.cpp's minja engine when arguments arrive as a JSON string instead of a parsed dict ([llama.cpp issue #19872](https://github.com/ggml-org/llama.cpp/issues/19872), [Qwen/Qwen3.5-35B-A3B discussion #4](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4)).

Unsloth has fixed this template in their 35B-A3B GGUF uploads but **has NOT yet updated the 27B** — their docs page says "Re-download 122B, 27B once they're updated." A user on HuggingFace [asked for the 27B ETA](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/discussions/13) with no response as of March 1, 2026 UTC.

**This is irrelevant to Ollama.** Ollama does not use the GGUF-embedded Jinja2 template — it has hardcoded Go renderers selected by `renderer: "qwen3.5"` in the model config. The Jinja template is only used by `llama-server --jinja` (direct llama.cpp usage). The llama.cpp side has its own fix ([PR #19635](https://github.com/ggml-org/llama.cpp/pull/19635)) that resolves the crash engine-side.

#### Q4_K_M vs UD-Q4_K_XL — Use Q4_K_M

> **Verified March 1, 2026 UTC.** All three GGUF files (Ollama registry, Unsloth Q4_K_M, Unsloth UD-Q4_K_XL) were downloaded and every tensor's quantization type was compared. The UD-Q4_K_XL analysis below is based on the Feb 24, 2026 upload — the only version available as of this writing.

Unsloth's `UD-Q4_K_XL` (Unsloth Dynamic 2.0) and `Q4_K_M` are the same file size (16.7 GB) from the same [Unsloth repository](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF), but they make **opposite tradeoffs** on the GDN/SSM layers. 192 of 851 tensors differ — all in the same four tensor types across all 48 GDN blocks:

| Tensor (48 GDN blocks each) | Ollama registry | Unsloth Q4_K_M | Unsloth UD-Q4_K_XL | What it is |
|------------------------------|-----------------|----------------|---------------------|------------|
| **`attn_qkv`** | Q4_K (4.5 bpw) | **Q5_K (5.5 bpw)** | **Q5_K (5.5 bpw)** | GDN linear attention input — both Unsloth variants protect this |
| **`attn_gate`** | Q4_K (4.5 bpw) | Q4_K (4.5 bpw) | **Q5_K (5.5 bpw)** | Attention gate — UD upgrades this, Q4_K_M leaves it at Q4_K |
| **`ssm_out`** | Q4_K (4.5 bpw) | **Q5_K (5.5 bpw)** | Q4_K (4.5 bpw) | SSM output projection — Q4_K_M protects, UD does NOT |
| **`ssm_alpha`** | Q4_K (4.5 bpw) | **Q8_0 (8.5 bpw)** | Q4_K (4.5 bpw) | SSM temporal decay — Q4_K_M keeps at nearly 2× precision, UD drops to Q4_K |
| **`ssm_beta`** | Q4_K (4.5 bpw) | **Q8_0 (8.5 bpw)** | Q4_K (4.5 bpw) | SSM input gate — same story as alpha |

**Q4_K_M wins on the most quantization-sensitive layers.** `ssm_alpha` and `ssm_beta` at Q8_0 (8.5 bpw) vs Q4_K (4.5 bpw) is nearly double the precision on the tiny tensors that control the SSM's temporal memory dynamics. `ssm_out` at Q5_K vs Q4_K is a +1 bpw upgrade on the tensor Unsloth's own benchmarks found "dramatically increases KLD" when quantized. UD-Q4_K_XL trades all of this away to upgrade `attn_gate` from Q4_K to Q5_K — a less impactful trade.

Both variants agree on `attn_qkv` (Q5_K) and everything outside the GDN blocks — the difference is purely in how they allocate the bit budget within the SSM layers.

**Note on UD re-upload status:** As of March 1, 2026 UTC, Unsloth has [flagged the 27B UD GGUFs for re-upload](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) ("112B, 27B still converting, re-download once updated") following an [MXFP4 bug fix](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/5) applied to the 35B-A3B MoE model. The MXFP4 bug was about expert routing tensors in MoE models — the dense 27B has no MoE layers, so it likely isn't affected. However, the UD quantization recipe may change after re-upload. Until then, Q4_K_M is the safer and empirically better choice for the SSM-sensitive layers.

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

> **Version note:** Latest stable release is **v0.16.0** (branch cut February 8, 2026 UTC). The Qwen 3.5 series launched February 16, 2026 UTC (397B-A17B flagship) and the 27B followed on February 24, 2026 UTC — both after the branch cut. **v0.16.0 does not include Qwen 3.5 support.** All analysis below is pinned to commit [`a1f53ad`](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329) (nightly post-v0.16.1rc0). Confirmed by vLLM maintainer @Isotr0py in [Issue #35391](https://github.com/vllm-project/vllm/issues/35391): *"vLLM 0.16.0 doesn't include Qwen3.5 support actually, because it's cut off before model support PR merged."*

### What Works

- `Qwen3_5ForConditionalGeneration` (dense) and `Qwen3_5MoeForConditionalGeneration` (MoE) are **registered** in the vLLM model registry ([`registry.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/models/registry.py))
- Full Gated DeltaNet implementation with custom Triton GPU kernels (Flash Linear Attention ops) at [`vllm/model_executor/layers/fla/ops/`](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/layers/fla/ops)
- `qwen3_coder` tool call parser implemented — XML-based: `<function=name><parameter=key>value</parameter>` ([`qwen3coder_tool_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py))
- `qwen3` reasoning parser covers both Qwen 3 and Qwen 3.5 ([`qwen3_reasoning_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/reasoning/qwen3_reasoning_parser.py))
- FP8, AWQ quantization supported through the standard pipeline
- MTP (Multi-Token Prediction) speculative decoding supported via `Qwen3_5MTP` ([`qwen3_5_mtp.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/models/qwen3_5_mtp.py))
- Dedicated V1 attention backend (`GDNAttentionBackend`) for hybrid KV (Key-Value) cache management
- `--language-model-only` flag skips the vision encoder for text-only deployment, freeing VRAM for KV cache ([`multimodal.py:258`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/config/multimodal.py#L258))
- `--default-chat-template-kwargs '{"enable_thinking": false}'` allows server-level thinking mode control ([`cli_args.py:96-101`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/cli_args.py#L96-L101))
- Penalty sampling (`presence_penalty`, `frequency_penalty`, `repetition_penalty`) **fully implemented and functional**

### What's Broken / Concerning

1. **Stable vLLM v0.16.0 does NOT support Qwen 3.5** — you must use the **nightly build**. For RTX 5090 (Blackwell), use Docker (recommended):
   ```bash
   docker run --gpus all -p 8000:8000 --ipc=host \
       -v ~/.cache/huggingface:/root/.cache/huggingface \
       vllm/vllm-openai:cu130-nightly <model-name> [options]
   ```
   For non-Blackwell GPUs, pip also works: `pip install -U vllm --extra-index-url https://wheels.vllm.ai/nightly`.
   Note: v0.16.1rc0 (tagged February 26, 2026 UTC) contains Qwen 3.5 support but is not on PyPI. The [official recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html) says "Use vLLM nightly wheel until 0.17.0 is released."
   Documented in [Issue #35391](https://github.com/vllm-project/vllm/issues/35391).

2. **Reasoning parser truncation bug ([#35221](https://github.com/vllm-project/vllm/issues/35221)):** If generation is truncated mid-reasoning (before `</think>`), the reasoning text is misclassified as content in non-streaming mode. The code at [`qwen3_reasoning_parser.py:70-73`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/reasoning/qwen3_reasoning_parser.py#L70-L73) returns `(None, model_output)` when no `</think>` is found. The serving layer mitigates this for the `enable_thinking=False` case via `prompt_is_reasoning_end_arr`, but the `max_tokens` truncation case is unhandled. Only affects non-streaming mode — streaming correctly routes pre-`</think>` tokens as reasoning via `extract_reasoning_streaming()` ([`qwen3_reasoning_parser.py:131-133`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/reasoning/qwen3_reasoning_parser.py#L131-L133)).

3. **Tool calling bugs:**
   - [#21711](https://github.com/vllm-project/vllm/issues/21711): `json.loads` fails because output includes XML `<tool_call>` tags
   - [#20611](https://github.com/vllm-project/vllm/issues/20611): Streaming + tool calls fails with thinking disabled
   - [#19051](https://github.com/vllm-project/vllm/issues/19051): `tool_choice: required` + reasoning parsing causes 400 errors
   - [#29192](https://github.com/vllm-project/vllm/issues/29192): `qwen3_coder` parser can produce infinite "!" streams with long inputs

4. **FP8 limitation:** The Gated DeltaNet's `ba_proj` (beta/alpha projection) does not support blockwise FP8 quantization. A separate FP8 shard_id weight loading bug ([Issue #35289](https://github.com/vllm-project/vllm/issues/35289)) was fixed in the nightly (present in the pinned commit).

5. **No default parameter injection:** vLLM's `Qwen3_5TextConfig` defines architecture parameters only — no sampling parameters. You must set `temperature`, `top_p`, `top_k`, `presence_penalty` explicitly in every API request. The model card recommends `presence_penalty=1.5` for 3 of 4 usage profiles — users who don't set it get vLLM's default of `0.0`, risking repetition loops during extended thinking.

6. **RTX 5090 (Blackwell sm_120) — Docker works out of the box, pip requires source build:** vLLM v0.15.1 (February 4, 2026 UTC) explicitly fixed sm_120 kernel issues — NVFP4 MoE kernels ([#33417](https://github.com/vllm-project/vllm/issues/33416)) and FP8 CUTLASS fallback to Triton ([#33285](https://github.com/vllm-project/vllm/issues/33416)). The official Docker images compile for `TORCH_CUDA_ARCH_LIST='7.0 7.5 8.0 8.9 9.0 10.0 12.0'` — the `12.0` entry is sm_120 (Blackwell RTX 5090). Available Docker tags for Blackwell: `vllm/vllm-openai:v0.16.0-cu130` (latest stable, CUDA 13.0.1, no Qwen 3.5 support), `vllm/vllm-openai:cu130-nightly` (nightly with Qwen 3.5 support), and `vllm/vllm-openai:qwen3_5-cu130` (purpose-built for Qwen 3.5, published February 23, 2026 UTC). Confirmed working on RTX 5090 (driver 590.48, PyTorch 2.10.0+cu130) by a user in [GitHub Issue #35303](https://github.com/vllm-project/vllm/issues/35303). **`pip install vllm` does NOT include sm_120 kernels** — pip users must build from source (with `TORCH_CUDA_ARCH_LIST="12.0"`) plus manually compile xFormers and FlashInfer. Docker eliminates all of this. Flash Attention 3 does not work on Blackwell; the Docker images use `VLLM_FLASH_ATTN_VERSION=2` (FA2 runs fine on sm_120, confirmed by Tri Dao in [flash-attention#1853](https://github.com/Dao-AILab/flash-attention/issues/1853)). Flash Attention 4 targets sm_100 (data center B200/B300), not sm_120 (consumer RTX), and is not released in the `flash-attn` package (latest is v2.8.3, August 2025). Dynamic FP8 quantization is reported slower than BF16 on RTX 5090 ([Issue #28234](https://github.com/vllm-project/vllm/issues/28234)) — use the pre-quantized `Qwen/Qwen3.5-27B-FP8` checkpoint instead of online quantization.

### Deep Dive: Quantization on 32 GB VRAM

The hybrid Gated DeltaNet architecture creates quantization constraints that differ from standard Transformers. The 48 linear attention layers each contain `in_proj_ba` (beta/alpha projection, output dim = 48) and `conv1d`, which cannot be blockwise FP8 quantized because 48 % 128 != 0 ([`fp8_utils.py:1457`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/layers/quantization/utils/fp8_utils.py#L1457)). These layers must remain in BF16 — but they total only ~47 MB across all 48 GDN layers, so the VRAM impact is negligible.

**Official recommendation: Do NOT quantize GDN layers.** The vLLM project's own llm-compressor examples for both FP8 ([`qwen3_next_example.py`](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w8a8_fp8/qwen3_next_example.py)) and FP4 ([`qwen3_next_example.py`](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/qwen3_next_example.py)) explicitly exclude all GDN layers via `ignore=["re:.*linear_attn.*"]`. The cyankiwi team confirmed this in a [HuggingFace discussion](https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4/discussions/1): "linear attention is more prone to quantization errors."

| Method | Checkpoint | Model VRAM (text-only) | Available for KV Cache | Tested on vLLM? |
|--------|-----------|------|---------|---------|
| BF16 | [`Qwen/Qwen3.5-27B`](https://huggingface.co/Qwen/Qwen3.5-27B) | ~53 GB | Won't fit | N/A — multi-GPU only |
| **FP8** | [**`Qwen/Qwen3.5-27B-FP8`**](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) (official Qwen) | **~28 GB** | **~4 GB (~64K tokens)** | **Yes — official checkpoint, FP8 shard_id bug fixed in nightly ([PR #35289](https://github.com/vllm-project/vllm/pull/35289))** |
| INT4 (compressed-tensors) | [`cyankiwi/Qwen3.5-27B-AWQ-4bit`](https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-4bit) (community, 2-person team, 10K downloads) | ~18 GB | ~14 GB (~224K tokens) | **Barely — 1 success on TP=2 (dual RTX 3090), 1 failure with garbled output. Zero single-GPU reports.** See [testing details](#4-bit-checkpoint-testing-status) below. |
| INT4 (compressed-tensors) | [`cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4`](https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-BF16-INT4) (community) | ~26 GB | ~6 GB (~96K tokens) | **Untested** — no community reports |
| INT4 (compressed-tensors, GPTQ algorithm) | [`Kbenkhaled/Qwen3.5-27B-quantized.w4a16`](https://huggingface.co/Kbenkhaled/Qwen3.5-27B-quantized.w4a16) (NVIDIA-affiliated, 15 downloads) | ~18 GB | ~14 GB (~224K tokens) | **Untested** — published Feb 27, 2026 UTC. Claims 100.3% accuracy recovery (GPQA Diamond, IFEval, MMLU-Redux). Zero community testing. |
| INT4 (native AWQ) | [`QuantTrio/Qwen3.5-27B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.5-27B-AWQ) (community, 2-person team, 2.5K downloads) | ~20 GB | ~12 GB (~192K tokens) | **Untested** — zero reports. Uses `dtype: float16` instead of BF16. |
| bitsandbytes | N/A | N/A | N/A | **Broken** — `NotImplementedError` for tuple shard_id in `in_proj_qkv` ([`linear.py:791-795`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/layers/linear.py#L791-L795)) |
| GGUF | N/A | N/A | N/A | **Not viable** — no Qwen 3.5 tensor name mapping in vLLM's GGUF loader ([`linear.py:738-743`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/layers/linear.py#L738-L743)) |

**KV cache math:** Only the 16 full attention layers use per-token KV cache (the 48 GDN layers use fixed-size recurrent state of ~72 MB per sequence, stored in vLLM's "mamba cache"). Full attention KV cost: 16 layers x 4 KV heads x 256 head_dim x 2 (K+V) x 2 bytes (BF16) = **64 KB per token**.

#### 4-Bit Checkpoint Testing Status

**No official Qwen 4-bit checkpoint exists.** [HuggingFace discussion #15](https://huggingface.co/Qwen/Qwen3.5-27B/discussions/15) (February 25, 2026 UTC, 7 upvotes) shows users requesting one with no Qwen team response as of February 27, 2026 UTC.

**The "AWQ" checkpoints are not standard AWQ format.** Despite having "AWQ" in their names, both `cyankiwi` checkpoints use `quant_method: "compressed-tensors"` (created with `llm-compressor` v0.13.1.a20260219), not `"awq"`. vLLM routes them through `CompressedTensorsConfig` and the Marlin WNA16 kernel path. Only `QuantTrio/Qwen3.5-27B-AWQ` uses native `quant_method: "awq"`.

**The cyankiwi team is a 2-person operation** (Ton Cao and Anton Calbert, [cyan.kiwi](https://cyan.kiwi), 200 HuggingFace followers, 227 published models). This is relevant context for assessing checkpoint reliability. The two checkpoints differ in what they quantize:
- **AWQ-4bit (20.1 GB on disk):** Quantizes GDN layers' large projections (`in_proj_qkv`, `in_proj_z`, `out_proj`, FFN) to INT4, keeping only `in_proj_a`/`in_proj_b` in BF16. This **contradicts the official llm-compressor recommendation** to exclude all GDN layers from quantization. No quality benchmarks published.
- **AWQ-BF16-INT4 (~28 GB on disk):** Keeps ALL GDN linear attention sub-layers in BF16, only quantizes full attention + FFN to INT4. Follows the official recommendation but at 28 GB on 32 GB VRAM, defeats the purpose of quantization.

**Community testing of `cyankiwi/Qwen3.5-27B-AWQ-4bit` ([HuggingFace discussion #1](https://huggingface.co/cyankiwi/Qwen3.5-27B-AWQ-4bit/discussions/1)):**
- **Success:** User Pa3kx, dual RTX 3090 (TP=2), `vllm/vllm-openai:nightly` Docker image, 63-65 tok/s decode at 8K context, 150K max context. Working command used `--tensor-parallel-size 2 --max-model-len 150000 --gpu-memory-utilization 0.90`.
- **Failure:** User thigger, 2x A6000 Ada (WSL2) — "wildly weird outputs" with `UserWarning: Input tensor shape suggests potential format mismatch: seq_len (11) < num_heads (24)`. Same user also reported SGLang crashes with `gptq_marlin_repack` error (dimension 24 not divisible by tile size 64).
- **No single-GPU reports exist.** The model fits in 32 GB VRAM (~18 GB model + ~14 GB KV) but nobody has confirmed this works on a single GPU.

**The Kbenkhaled/Qwen3.5-27B-quantized.w4a16 checkpoint** (18.6 GB, single safetensors file) is the most promising alternative: published by an NVIDIA-affiliated author, claims 100.3% average accuracy recovery on GPQA Diamond / IFEval / MMLU-Redux, uses GPTQ algorithm with `actorder: static` in compressed-tensors format. Published approximately February 27, 2026 UTC. Zero community testing.

**The FP8 ba_proj exclusion is handled cleanly** by the official checkpoint. The [`config.json`](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) contains a `modules_to_not_convert` list with 651 entries that explicitly excludes `in_proj_a`, `in_proj_b`, and `conv1d` for all 48 GDN layers (plus vision encoder layers, embeddings, and lm_head). The excluded BF16 weights total ~47 MB — negligible. A separate FP8 shard_id weight loading bug ([Issue #35287](https://github.com/vllm-project/vllm/issues/35287) / [PR #35289](https://github.com/vllm-project/vllm/pull/35289)) that caused `NotImplementedError` when loading the FP8 checkpoint was fixed in the nightly (present in the pinned commit).

### Recommended vLLM Inference Server Launch Command

**For 32 GB VRAM (RTX 5090), use the official FP8 checkpoint — the only well-tested quantization:**

```bash
docker run --gpus all -p 8000:8000 --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:cu130-nightly \
    Qwen/Qwen3.5-27B-FP8 \
    --port 8000 \
    --max-model-len 32768 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

The FP8 checkpoint (~28 GB text-only with `--language-model-only`) leaves ~4 GB for KV cache — enough for ~64K tokens single-sequence. The `--language-model-only` flag disables the ~460M-parameter vision encoder, freeing ~0.9 GB of VRAM. `--max-model-len 32768` is a conservative limit that fits comfortably within the 4 GB KV budget.

**Experimental 4-bit alternative (untested on single GPU):** If you need more KV cache headroom and are willing to use an unverified community checkpoint:

```bash
# EXPERIMENTAL — cyankiwi/Qwen3.5-27B-AWQ-4bit has 1 TP=2 success, 0 single-GPU reports
docker run --gpus all -p 8000:8000 --ipc=host \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:cu130-nightly \
    cyankiwi/Qwen3.5-27B-AWQ-4bit \
    --port 8000 \
    --max-model-len 65536 \
    --language-model-only \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

The AWQ-4bit checkpoint (~18 GB text-only) would leave ~14 GB for KV cache (~224K tokens). **Quality is unverified:** this checkpoint quantizes GDN layers to INT4 against the official llm-compressor recommendation, has no published perplexity or benchmark results, and the one successful deployment used tensor parallelism across two GPUs (not single-GPU). Use at your own risk.

### Deep Dive: Chat Template Architecture — Why vLLM Has No Template Mismatch

Unlike the Ollama inference framework (which compiles its own renderers in Go source code), **vLLM uses the HuggingFace model publisher's Jinja2 chat template verbatim**. This is the single most important architectural difference between the two frameworks for template correctness.

#### Template Resolution Chain

The function `resolve_chat_template()` in [`hf.py:102-149`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/renderers/hf.py#L102-L149) follows a priority chain:

1. **Explicit override** ([`hf.py:110-111`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/renderers/hf.py#L110-L111)): If `--chat-template` CLI flag is set, uses that template verbatim.
2. **AutoProcessor** ([`hf.py:114-120`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/renderers/hf.py#L114-L120)): For multimodal models without tools, tries the processor's template.
3. **AutoTokenizer** ([`hf.py:123-130`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/renderers/hf.py#L123-L130)): Calls `tokenizer.get_chat_template()` — this reads the `chat_template` field from the model's `tokenizer_config.json` exactly as published on HuggingFace Hub.
4. **Predefined fallbacks** ([`hf.py:132-149`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/renderers/hf.py#L132-L149)): Only if none of the above work. The fallback registry has a `"qwen"` entry ([`registry.py:42`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/transformers_utils/chat_templates/registry.py#L42)) but it only triggers for models with no chat template — Qwen 3.5 has one, so this never fires.

For Qwen 3.5, priority #3 always wins — `tokenizer.get_chat_template()` loads the 7,757-byte Jinja2 template from [`Qwen/Qwen3.5-27B/tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json). There are **no Qwen3.5-specific template overrides** anywhere in the vLLM codebase.

The template is applied unmodified at [`hf.py:472-478`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/renderers/hf.py#L472-L478):

```python
return tokenizer.apply_chat_template(
    conversation=conversation,
    tools=tools,
    chat_template=chat_template,
    tokenize=tokenize,
    **resolved_kwargs,
)
```

This is a direct call to the HuggingFace `transformers` library, which renders the Jinja2 template with the provided variables.

#### What This Means for Template Correctness

Because vLLM delegates template rendering entirely to the model publisher's Jinja2 template, every aspect of the prompt format is automatically correct:

| Aspect | How vLLM Handles It | Correct? |
|--------|-------------------|----------|
| Tool definition system prompt | Template generates `# Tools\n\nYou have access to the following functions:\n\n<tools>...` | **YES** — from HF template |
| Tool call format instruction | Template shows the XML example: `<function=name><parameter=key>value</parameter></function>` | **YES** — from HF template |
| `<IMPORTANT>` compliance block | Template includes 4 formatting rules (function calls MUST follow format, required params, etc.) | **YES** — from HF template |
| System message position | Template places user's system message **after** tool instructions | **YES** — from HF template |
| Multi-turn tool call history | Template re-renders prior tool calls in XML using `tool_call.arguments\|items` | **YES** — from HF template |
| Tool response wrapping | Template wraps `tool` role messages in `<tool_response>...</tool_response>` inside `user` messages | **YES** — from HF template |
| `last_query_index` logic | Template distinguishes real user messages from tool responses for thinking block insertion | **YES** — from HF template |
| `</think>` closure before tool calls | Template always emits `</think>\n\n` before `<tool_call>` | **YES** — from HF template |

This is a **fundamental architectural advantage** over the Ollama inference framework. Ollama's compiled-in Go renderers must be manually kept in sync with each model's training format — and for Qwen 3.5, they are not. vLLM's approach guarantees template correctness for any model that publishes a proper Jinja2 template on HuggingFace.

#### `enable_thinking` Parameter Passthrough

The `enable_thinking` parameter follows this path:

1. **Server-level default** ([`cli_args.py:96-101`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/cli_args.py#L96-L101)): Set via `--default-chat-template-kwargs '{"enable_thinking": false}'`.
2. **Per-request override** ([`protocol.py:263-269`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/chat_completion/protocol.py#L263-L269)): Client sends `"chat_template_kwargs": {"enable_thinking": true}` in the request body.
3. **Merge** ([`serving.py:917-926`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/engine/serving.py#L917-L926)): Request values override server defaults.
4. **Template variable filtering** ([`hf.py:376-439`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/renderers/hf.py#L376-L439)): The function `resolve_chat_template_kwargs()` parses the Jinja template AST via `jinja2.meta.find_undeclared_variables()` ([`hf.py:383`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/renderers/hf.py#L383)) and only passes kwargs the template actually uses. The Qwen 3.5 template declares `enable_thinking` via `{% set enable_thinking = enable_thinking | default(true) %}`, so it passes through.

When `enable_thinking=true` (default): the template ends with `<|im_start|>assistant\n<think>\n` — the model generates reasoning, then `</think>`, then content/tool calls.

When `enable_thinking=false`: the template ends with `<|im_start|>assistant\n<think>\n\n</think>\n\n` — a pre-filled empty think block. The model skips directly to final content.

The serving layer detects the disabled case without re-parsing the flag. At [`serving.py:808-817`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/chat_completion/serving.py#L808-L817), `reasoning_parser.is_reasoning_end(res.prompt_token_ids)` scans the prompt token IDs backwards ([`basic_parsers.py:69-78`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/reasoning/basic_parsers.py#L69-L78)) — if the last think-related token is `</think>` (ID 248069), it sets `prompt_is_reasoning_end_arr[i] = True`, causing all generated tokens to be routed as content without invoking the reasoning parser.

**Important:** Qwen 3.5 does **NOT** support the `/think` and `/nothink` soft-switch syntax from Qwen 3. The only way to control thinking is via the `enable_thinking` template parameter.

#### Reasoning Parser + Tool Call Parser: Clean Handoff

When both `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder` are enabled, the serving layer at [`serving.py:1036-1101`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/chat_completion/serving.py#L1036-L1101) follows a strict two-phase protocol:

**Phase 1 — Reasoning** ([`serving.py:1041-1077`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/chat_completion/serving.py#L1041-L1077)): All generated tokens are processed by `Qwen3ReasoningParser.extract_reasoning_streaming()`. Tokens before `</think>` are routed to the `reasoning` delta field. The tool parser is not invoked at all during this phase.

**Transition** ([`serving.py:1065-1076`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/chat_completion/serving.py#L1065-L1076)): When `reasoning_parser.is_reasoning_end(output_token_ids)` returns `True` (i.e., `</think>` appears in the output), the code sets `reasoning_end_arr[i] = True`, extracts post-`</think>` content token IDs via `extract_content_ids()`, and strips the reasoning text. Only the content after `</think>` is carried forward.

**Phase 2 — Tool parsing** ([`serving.py:1078-1101`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/chat_completion/serving.py#L1078-L1101)): The tool parser receives a **clean stream starting from content after `</think>`** — it never sees `<think>` tags. On the first tool-parsing invocation after reasoning ends, `previous_text` and `previous_token_ids` are reset to empty ([`serving.py:1086-1089`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/chat_completion/serving.py#L1086-L1089)), and `delta_text`/`delta_token_ids` contain only the post-reasoning content. Then `tool_parser.extract_tool_calls_streaming()` processes the `<tool_call>` XML.

For the **non-streaming path** ([`serving.py:1490-1511`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/chat_completion/serving.py#L1490-L1511)): `extract_reasoning()` splits output into `(reasoning, content)`, then `_parse_tool_calls_from_content()` runs on just the `content` portion. Same clean handoff.

This means vLLM correctly handles the standard Qwen 3.5 pattern of `<think>reasoning</think><tool_call>...</tool_call>` without any interference between the two parsers.

#### Tool Call Parser Architecture

Two Qwen3-compatible parsers are registered in [`__init__.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/__init__.py):

| Parser Name | Class | File |
|-------------|-------|------|
| `qwen3_xml` | `Qwen3XMLToolParser` | [`qwen3xml_tool_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3xml_tool_parser.py) |
| `qwen3_coder` | `Qwen3CoderToolParser` | [`qwen3coder_tool_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py) |

Both parse the same XML format (`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`). The `qwen3_coder` parser is the recommended one per the [HuggingFace model card](https://huggingface.co/Qwen/Qwen3.5-27B) and the [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html).

The `Qwen3CoderToolParser` at [`qwen3coder_tool_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py) provides:

- **Non-streaming extraction** ([`extract_tool_calls`, lines 291-341](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py#L291-L341)): Regex-based parsing of complete `<tool_call>` blocks, then per-function XML parsing via `_parse_xml_function_call()`.
- **Streaming extraction** ([`extract_tool_calls_streaming`, lines 343-783](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py#L343-L783)): Incremental state machine that detects `<tool_call>` start/end tokens by ID ([lines 76-83](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py#L76-L83)), tracks function names, and streams JSON argument fragments as they're parsed from XML parameters.
- **Schema-aware type conversion** ([`_convert_param_value`, lines 136-240](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py#L136-L240)): Matches parameter values against the tool's JSON Schema definition. Converts strings to `int`, `float`, `bool`, `array`, or `object` as declared. Falls back to `ast.literal_eval()` for unrecognized types ([line 230](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py#L230)) — this replaced a former `eval()` call that was a security vulnerability ([GHSA-79j6-g2m3-jgfw](https://github.com/vllm-project/vllm/security/advisories/GHSA-79j6-g2m3-jgfw), fixed in v0.10.1.1 via [PR #21396](https://github.com/vllm-project/vllm/pull/21396)).

#### Special Token Flags

From the model's `tokenizer_config.json`, the tool call and thinking tokens are all `special: false`:

| Token | ID | `special` flag |
|-------|------|----------------|
| `<\|endoftext\|>` | 248044 | **true** |
| `<\|im_start\|>` | 248045 | **true** |
| `<\|im_end\|>` | 248046 | **true** |
| `<tool_call>` | 248058 | false |
| `</tool_call>` | 248059 | false |
| `<tool_response>` | 248066 | false |
| `</tool_response>` | 248067 | false |
| `<think>` | 248068 | false |
| `</think>` | 248069 | false |

These are regular vocabulary tokens, not special control tokens. The vLLM tool parser uses token IDs for detection (e.g., `self.tool_call_start_token_id` at [`qwen3coder_tool_parser.py:76`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py#L76)), which is correct regardless of the `special` flag. The reasoning parser similarly uses `self.end_token_id` ([`basic_parsers.py:62`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/reasoning/basic_parsers.py#L62)).

#### Model Implementation

The main model code lives in [`qwen3_5.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/models/qwen3_5.py) (880 lines):

- **64 layers** in a 3:1 ratio — layer type determined by `config.layer_types`, defaulting to `linear_attention` for 3 out of every 4 layers
- **Custom `Qwen3_5GatedDeltaNet`** extends `Qwen3NextGatedDeltaNet` with modified QKVZ projection ordering
- **Vision support** via `Qwen3_5_VisionTransformer` inheriting from Qwen3-VL — disabled by `--language-model-only`
- **Config** at [`configs/qwen3_5.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/transformers_utils/configs/qwen3_5.py): `vocab_size=248320`, `hidden_size=4096`, `num_hidden_layers=32` (the 27B model actually has 64 layers; this config covers the 2B variant)

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
2. **BUG REPORT — Tool calling format mismatch** — the Ollama inference framework sends Qwen 3 Hermes-style JSON format but the Qwen 3.5 model was trained on Qwen3-Coder XML format. The correct Qwen3-Coder pipeline exists in the codebase but lacks thinking support, making the fix non-trivial — see [Section 8](#8-ollama-critical-bug-tool-calling-format-mismatch)
3. **BUG REPORT — Unclosed `</think>` tag** — in multi-turn tool call history, thinking blocks are never closed before tool calls when assistant content is empty — see [Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)
4. **BUG REPORT — Missing generation prompt after assistant tool call turns** — when the last message is an assistant message with tool calls, the renderer fails to append the generation prompt (`<|im_start|>assistant\n<think>\n`), leaving the model with no open turn to generate into. Affects agentic cancellation, timeout, and error recovery scenarios — see [Section 9.5](#95-ollama-critical-bug-missing-generation-prompt-after-tool-call-turns)
5. **INFORMATIONAL — Thinking mode disable broken** ([#14418](https://github.com/ollama/ollama/issues/14418)) — `think: false` does not work. **This is actually desirable for agent use cases** — thinking should always be on. The model's agentic training assumes thinking mode.
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

The correct fix is to wire `"qwen3.5"` to the Qwen3-Coder pipeline with thinking support added. This is not a simple one-line rewire — the existing `Qwen3CoderRenderer` and `Qwen3CoderParser` have no thinking support, so both must be extended before they can replace the current (wrong) `Qwen3VLRenderer`/`Qwen3Parser` wiring.

**Renderer changes** (`Qwen3CoderRenderer` at [`qwen3coder.go:58-193`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3coder.go#L58-L193)):
- Add `isThinking bool` and `emitEmptyThinkOnNoThink bool` fields to the struct (matching the pattern already used by `Qwen3VLRenderer`)
- Accept the `*api.ThinkValue` parameter in `Render()` (currently named `_` and ignored)
- Resolve thinking state at the top of `Render()`: use `think.Bool()` if the parameter is non-nil, otherwise fall back to `r.isThinking`
- Compute `lastQueryIndex` by scanning `filteredMessages` backwards for the last `Role == "user"` message (matching the HuggingFace template's `last_query_index` convention). In the `case "assistant"` block, only emit `<think>\n{thinking}\n</think>\n\n` when thinking is enabled, `message.Thinking != ""`, AND `i > lastQueryIndex` — stripping historical thinking traces from previous rounds. The `Qwen3VLRenderer` already implements this pattern at [`qwen3vl.go:61-74`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go#L61-L74).
- Append thinking prefill at the end of the prompt: `<think>\n` when thinking is enabled, or `<think>\n\n</think>\n\n` when `emitEmptyThinkOnNoThink` is set and thinking is disabled (this matches the HuggingFace Jinja2 template behavior for `enable_thinking=False`)
- Fix the `prefill` variable to exclude assistant messages with tool calls — see [Bug 4 (Section 9.5)](#95-ollama-critical-bug-missing-generation-prompt-after-tool-call-turns) for why this is critical

**Parser changes** (`Qwen3CoderParser` at [`qwen3coder.go:31-194`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3coder.go#L31-L194)):
- Add `hasThinkingSupport`, `defaultThinking`, and `maybeThinkingOpenAtBOL` fields
- Add two new parser states: `CollectingThinking` (strips the optional leading `<think>` tag, accumulates thinking content, watches for `</think>` or `<tool_call>` — whichever comes first) and `ThinkingDoneTransition` (eats whitespace between `</think>` and the start of content or tool calls)
- Extend `Init()` to set the initial state to `CollectingThinking` when thinking is enabled (following the pattern from `Qwen3Parser.Init()` at [`qwen3.go:55-73`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3.go#L55-L73))
- Extend `Add()` to return thinking content separately (the `qwenEventThinkingContent` type already exists in the `parsers` package at [`qwen3vl.go:63-67`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/qwen3vl.go#L63-L67))
- Handle streaming edge cases in the `eat()` function: partial `</think>` and `<tool_call>` tag overlaps at chunk boundaries, `<tool_call>` appearing inside an unclosed thinking block (direct transition to tool collection, skipping `ThinkingDoneTransition`)

**Wiring changes:**
- [`renderer.go:59-61`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/renderer.go#L59-L61): Change `"qwen3.5"` from `&Qwen3VLRenderer{isThinking: true, emitEmptyThinkOnNoThink: true}` to `&Qwen3CoderRenderer{isThinking: true, emitEmptyThinkOnNoThink: true}`
- [`parsers.go:52-53`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/parsers/parsers.go#L52-L53): Change `"qwen3.5"` from `&Qwen3Parser{hasThinkingSupport: true, defaultThinking: true}` to `&Qwen3CoderParser{hasThinkingSupport: true, defaultThinking: true}`

**Registry config blob** at `registry.ollama.ai` also needs updating (only the Ollama team can do this), but the config blob only specifies the string names `"qwen3.5"` for both renderer and parser — the actual behavior is determined by the switch statements in the Ollama binary. So the code-side fix takes effect immediately for users who update their Ollama binary, regardless of whether the registry blob is changed.

The Qwen 3.5 model's tool calling capability is strong — it was trained on 20,000 parallel rollout environments for agentic tasks. The issue is entirely in the Ollama inference framework's wiring, not in the model.

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
| Missing generation prompt after assistant tool call turns | Renderer bug in Ollama source code | **CRITICAL** | **YES** — see [Section 9.5](#95-ollama-critical-bug-missing-generation-prompt-after-tool-call-turns) |
| System message placed before tool instructions (should be after) | Renderer bug in Ollama source code | Moderate | Optional |
| Missing `<IMPORTANT>` format compliance block | Renderer bug in Ollama source code | Moderate | Optional |

---

## 9.5. Ollama Critical Bug: Missing Generation Prompt After Tool Call Turns

> **BUG REPORT ITEM** — When the last message in a conversation is an assistant message with tool calls, both `Qwen3VLRenderer` and `Qwen3CoderRenderer` fail to append the generation prompt. The model receives an incomplete prompt with no open assistant turn, breaking agentic cancellation, timeout, and error recovery scenarios.

### The Problem

In the Ollama rendering pipeline, the `prefill` variable controls whether the last assistant message is treated as a text continuation (no `<|im_end|>`, no generation prompt) or a complete turn (closed with `<|im_end|>`, followed by `<|im_start|>assistant\n`). In both renderers, `prefill` is defined as:

```go
prefill := lastMessage && message.Role == "assistant"
```

This treats ALL last-position assistant messages as prefills — including messages with tool calls. An assistant message with tool calls is a **complete turn** (the model already finished generating), not a partial text to extend. When such a message is last in the conversation, the renderer should close it with `<|im_end|>` and append the generation prompt (`<|im_start|>assistant\n<think>\n`), giving the model an open turn to generate into.

**Source (Ollama v0.17.4):**
- `Qwen3CoderRenderer`: [`qwen3coder.go:148`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3coder.go#L148) — `prefill := lastMessage && message.Role == "assistant"`
- `Qwen3VLRenderer`: [`qwen3vl.go:85`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go#L85) — same pattern

### HuggingFace Ground Truth

The [HuggingFace Jinja2 chat template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) always closes assistant turns with `<|im_end|>` and **unconditionally** appends the generation prompt after the message loop when `add_generation_prompt` is true:

```jinja
{% if add_generation_prompt %}
<|im_start|>assistant
{% if enable_thinking %}<think>
{% endif %}
{% endif %}
```

This fires regardless of the last message type. There is no special-casing for tool calls vs content — every complete conversation gets the generation prompt.

### When This Breaks

This bug manifests in agentic tool-calling flows:

1. **Cancellation/timeout:** The agent sends an assistant message with tool calls, then the user cancels or a timeout occurs before the tool response arrives. The conversation now ends with an assistant tool call message.
2. **Error recovery:** The tool execution fails, and the framework re-prompts the model to try again. The last message in the re-prompt conversation is the original assistant tool call message.
3. **Manual intervention:** The user sends a follow-up message while tool calls are pending, and the framework must construct a prompt that includes the prior assistant tool call turn.

In all cases, the model receives a prompt that ends mid-turn — the assistant message's `<|im_end|>` is missing and no `<|im_start|>assistant\n<think>\n` generation prompt follows. The model has no open turn to generate into.

### The Fix

Change `prefill` to exclude assistant messages with tool calls:

```go
prefill := lastMessage && message.Role == "assistant" && len(message.ToolCalls) == 0
```

This is a one-line change in each renderer. With this fix, only assistant messages without tool calls are treated as text continuations (prefills). Assistant messages with tool calls are treated as complete turns: closed with `<|im_end|>` and followed by the generation prompt, matching the HuggingFace Jinja2 template behavior.

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

The four sampling profiles from the [Qwen 3.5 27B model card](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices) are documented in [Section 3](#3-recommended-inference-parameters) above. For complete, copy-paste-ready Ollama API request bodies with all necessary overrides (`repeat_penalty`, `presence_penalty`, `num_ctx`, `num_predict`), see [`solution.md`](solution.md#all-four-parameter-profiles).

**Key notes for upstream Ollama (without the fork fixes):** The parameter profiles in Section 3 include `presence_penalty` values (1.5 for general, 0.0 for precise coding, 2.0 for hard reasoning). These are **silently ignored** by upstream Ollama's Go-based runner — see [Bug 1](#7-ollama-critical-bug-missing-penalty-sampling). The [fork](https://github.com/BigBIueWhale/ollama) fixes this.

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

Multiple open bugs exist: XML parsing failures, streaming conflicts, infinite "!" streams. The `qwen3_coder` parser had a [security vulnerability (GHSA-79j6-g2m3-jgfw)](https://github.com/vllm-project/vllm/security/advisories/GHSA-79j6-g2m3-jgfw) where it used Python's `eval()` for unknown parameter types — **fixed in vLLM v0.10.1.1** ([PR #21396](https://github.com/vllm-project/vllm/pull/21396)). The current code at the pinned commit (`a1f53ad`) uses `ast.literal_eval()` ([`qwen3coder_tool_parser.py:230`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py#L230)), which only evaluates Python literals — no arbitrary code execution.

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
| Model loads | **Nightly only** | Stable vLLM v0.16.0 fails; nightly required ([#35391](https://github.com/vllm-project/vllm/issues/35391)) |
| Chat template | **CORRECT** | Uses HF Jinja2 template verbatim via `tokenizer.apply_chat_template()` — no overrides ([Section 4](#4-vllm-support-status)) |
| `enable_thinking` | **CORRECT** | Passed through as Jinja variable; prompt detection via `is_reasoning_end()` for disabled case ([Section 4](#4-vllm-support-status)) |
| Reasoning + tool parser handoff | **CORRECT** | Two-phase protocol: reasoning fully consumed before tool parsing begins ([Section 4](#4-vllm-support-status)) |
| temperature | **Must set manually** | No defaults injected |
| top_k | **Must set manually** | No defaults injected |
| top_p | **Must set manually** | No defaults injected |
| presence_penalty | **Works if set** | vLLM inference server implements it correctly; not defaulted (model card recommends 1.5) |
| Tool calling format | **CORRECT** | `qwen3_coder` parser matches model's trained XML format |
| Generation prompt | **CORRECT** | HuggingFace Jinja2 template unconditionally appends `<|im_start|>assistant\n<think>\n` after message loop |
| Tool calling stability | **Buggy** | Multiple open issues ([#21711](https://github.com/vllm-project/vllm/issues/21711), [#20611](https://github.com/vllm-project/vllm/issues/20611), [#19051](https://github.com/vllm-project/vllm/issues/19051), [#29192](https://github.com/vllm-project/vllm/issues/29192)) |
| Reasoning parser | **Known bug** | Truncated reasoning misclassified as content in non-streaming mode only ([#35221](https://github.com/vllm-project/vllm/issues/35221)) |
| `eval()` security vuln | **FIXED** | Was `eval()`, now `ast.literal_eval()` since v0.10.1.1 ([PR #21396](https://github.com/vllm-project/vllm/pull/21396)) |
| DeltaNet implementation | **Complete** | Triton kernels, dedicated attention backend |
| MTP speculative decoding | **Supported** | `Qwen3_5MTP` registered |
| RTX 5090 Blackwell | **Supported (Docker)** | Docker `-cu130` images (CUDA 13.0.1) include sm_120 kernels — confirmed working in [#35303](https://github.com/vllm-project/vllm/issues/35303); `pip install` requires source build; FA2 only (FA3 incompatible, FA4 not released for sm_120); for Qwen 3.5: use `vllm/vllm-openai:cu130-nightly` or `qwen3_5-cu130` |

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
| Generation prompt after tool calls | **BROKEN** | `prefill` variable treats last assistant messages with tool calls as text continuations, omitting `<|im_end|>` and generation prompt ([Section 9.5](#95-ollama-critical-bug-missing-generation-prompt-after-tool-call-turns)) | **YES** |
| Stop tokens | **Adequate** | Params blob has no string-based `stop` sequences, but EOS token IDs in the GGUF file (`<\|im_end\|>` 248046, `<\|endoftext\|>` 248044) handle turn-boundary stopping correctly ([Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m)) | No |
| Non-tool prompt format | **CORRECT** | Matches HuggingFace ground truth template byte-for-byte for simple chat (no tools) | — |
| Thinking prefill | **CORRECT** | Both enabled (`<think>\n`) and disabled (`<think>\n\n</think>\n\n`) match HuggingFace ground truth | — |
| Historical thinking stripping | **PARTIAL** | `Qwen3VLRenderer` implements `lastQueryIndex` correctly ([`qwen3vl.go:61-74`](https://github.com/ollama/ollama/blob/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/model/renderers/qwen3vl.go#L61-L74)), but `Qwen3CoderRenderer` (needed for Qwen 3.5) does not — it preserves thinking from all rounds. HuggingFace template strips thinking from turns before `last_query_index`. | **YES** |
| think=false | **BROKEN (OK)** | Does not disable thinking; this is desirable for agent use | No |
| Context length default | **CORRECT** | 32,768 tokens for 32 GB VRAM | — |
| DeltaNet numerics | **CORRECT** | FP32 unary ops, no overflow possible | — |
| DeltaNet implementation | **CORRECT** | Full chunked + autoregressive in Go and C++ | — |
| Parallel requests | **Limited** | Forced to 1 (expected architectural limitation of hybrid recurrent state) | No |
| GGUF sampling metadata | **IGNORED** | Ollama inference framework reads only the registry params blob, ignores GGUF `general.sampling.*` keys | — |

### Bottom Line

**For agent coding use cases, the Ollama inference framework v0.17.4 has four layers of critical bugs that make Qwen 3.5 27B tool calling non-functional:**

> These four bugs are the verified, source-linked findings that belong in a bug report to the Ollama team. See the [Bug Report Summary](#bug-report-summary) at the top of this document for a self-contained summary with all links.

1. **BUG REPORT — Missing penalty sampling** (Ollama Go-based runner limitation) — The Qwen 3.5 model's recommended `presence_penalty=1.5` is silently accepted by the Ollama API and silently ignored by the Go-based runner's sampler. Extended thinking sequences may enter repetition loops with no API-level workaround available. See [Section 7](#7-ollama-critical-bug-missing-penalty-sampling).

2. **BUG REPORT — Wrong tool calling format** (wiring bug in Ollama source code + Ollama model registry config blob) — The registry config blob sets `renderer: "qwen3.5"` and `parser: "qwen3.5"`, which map to the Qwen 3 Hermes-style JSON pipeline. The Qwen 3.5 model was trained on the Qwen3-Coder XML format. The correct `Qwen3CoderParser` + `Qwen3CoderRenderer` already exist in the Ollama codebase but lack thinking support, requiring non-trivial extension before they can be rewired to `"qwen3.5"`. See [Section 8](#8-ollama-critical-bug-tool-calling-format-mismatch).

3. **BUG REPORT — Unclosed `</think>` tag in multi-turn tool call history** (renderer bug in Ollama source code) — The `Qwen3VLRenderer` fails to close `</think>` tags before tool calls when the assistant message has empty content, placing `<tool_call>` blocks inside unclosed thinking blocks and corrupting the conversation structure the model sees. See [Section 9](#9-ollama-registry-model-validation-qwen3527b-q4_k_m).

4. **BUG REPORT — Missing generation prompt after assistant tool call turns** (renderer bug in Ollama source code) — Both `Qwen3VLRenderer` and `Qwen3CoderRenderer` treat all last-position assistant messages as text continuations (prefills), including those with tool calls. The HuggingFace Jinja2 chat template unconditionally appends the generation prompt after the message loop. Without this, the model has no open assistant turn to generate into when the conversation ends with an assistant tool call — a scenario that arises during agentic cancellation, timeout, and error recovery. See [Section 9.5](#95-ollama-critical-bug-missing-generation-prompt-after-tool-call-turns).

**For simple chat without tools, the Ollama inference framework works correctly.** The prompt format matches the HuggingFace ground truth template byte-for-byte, the sampling parameters (temperature, top_k, top_p) are correct, and thinking mode works. All four bugs are specific to tool calling and penalty sampling — areas that only matter for agentic use cases.

**The vLLM inference server is the only viable path for agent/tool use** despite requiring a nightly build. It has working penalty sampling, the correct tool call parser (`qwen3_coder` matching the model's XML format), proper thinking mode, and — critically — **uses the HuggingFace Jinja2 chat template verbatim** ([Section 4](#4-vllm-support-status)), guaranteeing template correctness for tool definitions, system message ordering, format instructions, and the `<IMPORTANT>` compliance block. Official Docker images with sm_120 (Blackwell) kernels are available — no source build required. The Ollama inference framework compiles its own Go renderers that must be manually kept in sync — and for Qwen 3.5, they are not. vLLM's approach means the prompt format is always correct by construction. Tool calling has implementation bugs but the format is right — bugs are fixable, wrong format is architectural. **Quantization caveat:** The official FP8 checkpoint is the only well-tested quantization for 32 GB VRAM (~28 GB model, ~4 GB KV cache, ~64K tokens). Community 4-bit checkpoints exist but are unverified on single-GPU deployments — see [quantization deep dive](#deep-dive-quantization-on-32-gb-vram).

**Neither library auto-applies the Qwen 3.5 model's recommended defaults** — you must set parameters explicitly in both cases. The model card recommends `presence_penalty=1.5` for 3 of 4 usage profiles. In vLLM, you can set this per-request or via server defaults. In the Ollama inference framework, it is silently ignored.

**Thinking mode always-on is correct for agent use.** The Qwen 3.5 model's agentic training (1 million reinforcement learning environments, Terminal-Bench 52.5) assumes thinking mode. Disabling it removes the model's strongest capability for multi-step tool use.

---

## 16. Source Code Versions Used

All source code analysis in this report was performed against locally cloned repositories at the commits listed below. Each commit is linked to its exact tree on GitHub for reproducibility.

| Repository | Version | Commit | Date | Link |
|------|---------|--------|------|------|
| vLLM inference server | nightly (post-v0.16.1rc0) | `a1f53addb` | Feb 26, 2026 UTC | [vllm-project/vllm@a1f53ad](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329) |
| Ollama inference framework (original analysis) | v0.17.1 | `9bf41969f` | Feb 24, 2026 UTC | [ollama/ollama@9bf4196](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a) |
| Ollama inference framework (re-verification) | post-v0.17.4 | `79917cf` | Feb 26, 2026 UTC | [ollama/ollama@79917cf](https://github.com/ollama/ollama/tree/79917cf80bf74538a4ae694e6b61adb908b0f8df) |
| Ollama inference framework (fork base for fixes) | v0.17.4 | `cc90a035` | Feb 26, 2026 UTC | [ollama/ollama@cc90a03](https://github.com/ollama/ollama/tree/cc90a035a0cc3ae9bd0c1dc95d42b620e8dcb0e2) |
| llama.cpp C++ inference library (upstream) | post-b5136 | `723c71064` | Feb 26, 2026 UTC | [ggml-org/llama.cpp@723c710](https://github.com/ggml-org/llama.cpp/tree/723c71064da0908c19683f8c344715fbf6d986fd) |
| llama.cpp C++ inference library (Ollama-pinned) | — | `ec98e2002` + [34 Ollama patches](https://github.com/ollama/ollama/tree/9bf41969f0c23d2ee980d7f092f5f80ea4521d2a/llama/patches) | — | [ggml-org/llama.cpp@ec98e2002](https://github.com/ggml-org/llama.cpp/tree/ec98e2002) |

Note: The upstream llama.cpp commit (`723c710`) is NOT the same as the Ollama-pinned version (`ec98e2002`). The Ollama inference framework vendors llama.cpp at a specific commit and applies 34 patches on top (for IMRoPE, Metal GPU backend fixes, etc.).

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

### vLLM Inference Server (pinned to commit [`a1f53ad`](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329), nightly post-v0.16.1rc0)
- [vLLM GitHub repository](https://github.com/vllm-project/vllm)
- [vLLM GitHub Releases](https://github.com/vllm-project/vllm/releases)
- [vLLM Qwen3.5 Usage Guide / Recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM Tool Calling Documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [Qwen vLLM Deployment Guide](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
- [Issue #35391 — Architecture Not Supported in v0.16.0](https://github.com/vllm-project/vllm/issues/35391) (confirmed by maintainer @Isotr0py)
- [Issue #35289 — FP8 Shard ID Weight Loading Bug (fixed in nightly)](https://github.com/vllm-project/vllm/issues/35289)
- [Issue #35221 — Reasoning Parser Truncation Bug](https://github.com/vllm-project/vllm/issues/35221)
- [Issue #35154 — Optimized Deployment Recipe for Qwen3.5 (backlog)](https://github.com/vllm-project/vllm/issues/35154)
- [Issue #28234 — Dynamic FP8 Slower Than BF16 on RTX 5090](https://github.com/vllm-project/vllm/issues/28234)
- [Issue #21711 — Tool Call XML Parsing](https://github.com/vllm-project/vllm/issues/21711)
- [Issue #20611 — Streaming Tool Calls](https://github.com/vllm-project/vllm/issues/20611)
- [Issue #19051 — Reasoning + Tool Calling Conflict](https://github.com/vllm-project/vllm/issues/19051)
- [Issue #29192 — Parser Infinite Stream](https://github.com/vllm-project/vllm/issues/29192)
- [Security Advisory GHSA-79j6-g2m3-jgfw — `eval()` in qwen3_coder parser (fixed in v0.10.1.1)](https://github.com/vllm-project/vllm/security/advisories/GHSA-79j6-g2m3-jgfw)
- [PR #21396 — Fix `eval()` to `ast.literal_eval()` in qwen3_coder parser](https://github.com/vllm-project/vllm/pull/21396)
- [vLLM on RTX 5090 Forum Thread](https://discuss.vllm.ai/t/vllm-on-rtx5090-working-gpu-setup-with-torch-2-9-0-cu128/1492)

### vLLM Key Source Files (at commit [`a1f53ad`](https://github.com/vllm-project/vllm/tree/a1f53addb132f75704710184f4c1cc4780343329))
- [`vllm/renderers/hf.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/renderers/hf.py) — Chat template resolution chain (`resolve_chat_template`, lines 102-149) and application (`safe_apply_chat_template`, lines 442-478)
- [`vllm/transformers_utils/chat_templates/registry.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/transformers_utils/chat_templates/registry.py) — Fallback template registry (never used for Qwen 3.5)
- [`vllm/reasoning/qwen3_reasoning_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/reasoning/qwen3_reasoning_parser.py) — Qwen3/3.5 reasoning parser
- [`vllm/reasoning/basic_parsers.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/reasoning/basic_parsers.py) — Base class with `is_reasoning_end()` (lines 69-78) and streaming extraction
- [`vllm/tool_parsers/qwen3coder_tool_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3coder_tool_parser.py) — Qwen3-Coder XML tool call parser (recommended for Qwen 3.5)
- [`vllm/tool_parsers/qwen3xml_tool_parser.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/tool_parsers/qwen3xml_tool_parser.py) — Older XML tool call parser (same format, different implementation)
- [`vllm/entrypoints/openai/chat_completion/serving.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/chat_completion/serving.py) — Streaming handler with two-phase reasoning+tool protocol (lines 1036-1101)
- [`vllm/entrypoints/openai/cli_args.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/entrypoints/openai/cli_args.py) — `--default-chat-template-kwargs` (lines 96-101), `--reasoning-parser`, `--tool-call-parser`
- [`vllm/model_executor/models/qwen3_5.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/models/qwen3_5.py) — Main model implementation (dense + MoE, 880 lines)
- [`vllm/model_executor/models/qwen3_5_mtp.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/models/qwen3_5_mtp.py) — MTP speculative decoding support
- [`vllm/model_executor/models/registry.py`](https://github.com/vllm-project/vllm/blob/a1f53addb132f75704710184f4c1cc4780343329/vllm/model_executor/models/registry.py) — Model registry (`Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5MTP`, `Qwen3_5MoeMTP`)

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
- [Upstream commit `723c710` (Feb 26, 2026 UTC, post-b5136)](https://github.com/ggml-org/llama.cpp/tree/723c71064da0908c19683f8c344715fbf6d986fd)
- [llama.cpp PR #19468 — Qwen 3.5 Support (Merged)](https://github.com/ggml-org/llama.cpp/pull/19468)
- [Issue #19860 — CUDA Multi-GPU Crash (Fixed)](https://github.com/ggml-org/llama.cpp/issues/19860)
- [PR #19866 — CUDA Multi-GPU Crash Fix](https://github.com/ggml-org/llama.cpp/pull/19866)
- [Issue #19869 — Chat Parsing Crash](https://github.com/ggml-org/llama.cpp/issues/19869)
- [Issue #19872 — Template `arguments|items` Filter Failure](https://github.com/ggml-org/llama.cpp/issues/19872)
- [PR #19635 — Template Detection Fix](https://github.com/ggml-org/llama.cpp/pull/19635)

### Unsloth GGUF Quantization (per-tensor analysis, March 1, 2026 UTC)
- [unsloth/Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) — full file downloaded (16,740,812,160 bytes, 851 tensors) for per-tensor comparison
- [Unsloth Qwen3.5 GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) — KL divergence per-tensor analysis, MXFP4 retirement
- [Unsloth Dynamic 2.0 Documentation](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) — per-layer quantization methodology
- [Unsloth Dynamic v2.0 Blog Post](https://unsloth.ai/blog/dynamic-v2) — calibration dataset, optimization goals
- [Unsloth Qwen3.5 How-to Guide](https://unsloth.ai/docs/models/qwen3.5) — tool-calling template fix announcement, "Re-download 122B, 27B once they're updated"
- [HuggingFace discussion: Qwen3.5-35B-A3B Feb 27 GGUF update + template diff](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/10)
- [HuggingFace discussion: Upstream Qwen tool calling template is broken](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4) — `|items` filter bug, NOT fixed upstream as of March 1, 2026 UTC
- [HuggingFace discussion: 27B update ETA inquiry (unanswered)](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/discussions/13)
- [HuggingFace discussion: Bug in UD-Q4_K_XL recipe using MXFP4 for attn tensors](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/5)
- [llama.cpp `--tensor-type` per-tensor override discussion](https://github.com/ggml-org/llama.cpp/discussions/12741)
- [Ollama GitHub Issue #10222 — Support Jinja chat templates](https://github.com/ollama/ollama/issues/10222) — OPEN since April 2025, no PRs, no timeline

### News Coverage
- [MarkTechPost — Qwen 3.5 Medium Series](https://www.marktechpost.com/2026/02/24/alibaba-qwen-team-releases-qwen-3-5-medium-model-series-a-production-powerhouse-proving-that-smaller-ai-models-are-smarter/)
- [VentureBeat — Qwen 3.5](https://venturebeat.com/technology/alibabas-qwen-3-5-397b-a17-beats-its-larger-trillion-parameter-model-at-a)
- [DataCamp — Qwen 3.5 Guide](https://www.datacamp.com/blog/qwen3-5)
