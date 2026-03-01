# Qwen 3.5 27B: Tool calling completely non-functional and repetition penalties silently ignored

Qwen 3.5 27B is the first consumer-GPU-capable model to match GPT-5 mini on SWE-bench (72.4), with native multimodal support, 262K context, and agentic tool calling trained across 1 million RL environments. It fits on a single RTX 5090 at Q4_K_M. There is currently no comparable alternative at this size.

Three bugs in Ollama make its agentic capabilities — the model's primary differentiator — completely non-functional. All three are verified against source code (v0.17.1 through master `79917cf`) and the [HuggingFace ground truth template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json). None are fixable by end users.

Full source-level analysis with exact file paths, line numbers, and diffs: [Inference Report](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md#bug-report-summary)

---

## Bug 1: Repetition penalties are silently ignored

The Go runner's sampler has **zero implementation of penalty sampling**. `repeat_penalty`, `presence_penalty`, and `frequency_penalty` are accepted by the API without error and silently discarded. The model card explicitly recommends `presence_penalty=1.5` to prevent repetition loops during thinking. Setting it via the API has no effect whatsoever.

The C++ runner (`llamarunner`) implements penalties correctly — but Qwen 3.5 is forced onto the Go runner via `OllamaEngineRequired()` and cannot use it.

This affects **all models on the Go runner**, not just Qwen 3.5.

**Full evidence:** [Missing Penalty Sampling](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md#7-ollama-critical-bug-missing-penalty-sampling)

---

## Bug 2: Tool calling uses the completely wrong format

The registry config blob sets `renderer: "qwen3.5"` / `parser: "qwen3.5"`, which maps to the **Qwen 3 Hermes-style JSON** tool calling pipeline (`Qwen3VLRenderer` + `Qwen3Parser`).

Qwen 3.5 was not trained on this format. It was trained on the **Qwen3-Coder XML** format (`<function=name><parameter=key>value</parameter></function>`), as confirmed by the [HuggingFace chat template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json).

The correct pipeline (`Qwen3CoderRenderer` + `Qwen3CoderParser`) already exists in the codebase — it's just wired to `"qwen3-coder"` instead of `"qwen3.5"`.

The system prompt, the format instruction, the tool call rendering in conversation history, and the output parser are all wrong. There are [6 concrete mismatches](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md#prompt-template-diff-huggingface-ground-truth-vs-ollama-renderer-output) between what Ollama sends and what the model was trained on.

**Full evidence:** [Tool Calling Format Mismatch](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md#8-ollama-critical-bug-tool-calling-format-mismatch)

---

## Bug 3: Unclosed `</think>` tag corrupts multi-turn tool calling prompts

When an assistant message has thinking + tool calls but no text content (the standard "think then call a tool" pattern), the renderer never emits `</think>`. The tool call is rendered **inside an unclosed `<think>` block**, corrupting every subsequent turn the model sees.

The **parser side** of this was fixed in v0.17.3 ([`d98dda4`](https://github.com/ollama/ollama/commit/d98dda4676d44a3882fd38492cc00db257f35974), PR [#14477](https://github.com/ollama/ollama/pull/14477)) — the parser now correctly handles model output that has `<tool_call>` before `</think>`. But the **renderer side** remains broken: multi-turn prompts sent to the model still contain unclosed `<think>` tags. The tool-call thinking tests in `qwen3vl_thinking_test.go` (lines 119–323) are still commented out.

**Full evidence:** [Bug 3 details](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md#bug-3-unclosed-think-tag-before-tool-calls-in-multi-turn-conversation-history)

---

## Status across releases

| Bug | v0.17.1 | v0.17.2 | v0.17.3 | v0.17.4 | master |
|-----|---------|---------|---------|---------|--------|
| Penalty sampling silently ignored | Present | Present | Present | Present | **Present** |
| Wrong tool call format | Present | Present | Present | Present | **Present** |
| Unclosed `</think>` (renderer) | Present | Present | Present | Present | **Present** |
| Unclosed `</think>` (parser) | Present | Present | **Fixed** | Fixed | Fixed |

## Environment

- Ollama v0.17.1 through master (`79917cf`, Feb 26, 2026 UTC)
- Model: `qwen3.5:27b-q4_K_M`
- Full report with source-level verification: [qwen3.5_27b_inference_report.md](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md)

---

**Edited (Feb 27, 2026 UTC):** Found two additional issues while implementing fixes for the above three bugs in a [fork based on v0.17.4](https://github.com/ollama/ollama/tree/cc90a035) (`cc90a035`):

**Bug 2 expanded — Coder pipeline has zero thinking support:** Simply rewiring `"qwen3.5"` to `Qwen3CoderRenderer`/`Qwen3CoderParser` is insufficient. Both have **zero thinking support** — no `<think>`/`</think>` handling in the renderer, no thinking state machine in the parser. Qwen 3.5 requires thinking support for agentic use (the model card recommends `enable_thinking=true`), so the Coder pipeline needs to be extended with full thinking support before the rewiring is useful.

**Bug 4 — Missing generation prompt after tool call turns:** When the last message is an assistant message with tool calls, the renderer treats it as a prefill (incomplete turn to be continued) and never emits `<|im_end|>` or the `<|im_start|>assistant\n` generation prompt. Root cause is the `prefill` variable in both [`qwen3coder.go:148`](https://github.com/ollama/ollama/blob/cc90a035/model/renderers/qwen3coder.go#L148) and [`qwen3vl.go:82`](https://github.com/ollama/ollama/blob/cc90a035/model/renderers/qwen3vl.go#L82) — it fires for any last assistant message including ones with tool calls. This breaks the entire tool call round-trip loop. Also affects `qwen3-vl-instruct` and `qwen3-vl-thinking`. [Full evidence](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md#95-ollama-critical-bug-missing-generation-prompt-after-tool-call-turns).

Updated status table:

| Bug | v0.17.1 | v0.17.2 | v0.17.3 | v0.17.4 | master |
|-----|---------|---------|---------|---------|--------|
| Penalty sampling silently ignored | Present | Present | Present | Present | **Present** |
| Wrong tool call format (+ missing thinking support) | Present | Present | Present | Present | **Present** |
| Unclosed `</think>` (renderer) | Present | Present | Present | Present | **Present** |
| Unclosed `</think>` (parser) | Present | Present | **Fixed** | Fixed | Fixed |
| Missing generation prompt after tool calls | Present | Present | Present | Present | **Present** |

---

## Note: Vendored llama.cpp version is not a concern

Ollama v0.17.4 vendors llama.cpp at commit `ec98e2002`, which corresponds to **llama.cpp b7437** (December 16, 2025) — two months before Qwen 3.5 was even announced (February 16, 2026). Community recommendations say llama.cpp **b8149+** (February 25, 2026) is required for Qwen 3.5 GGUF support. This sounds alarming but is **not actually a problem** for Ollama. Here's why:

### What llama.cpp b8149 fixes (and why Ollama doesn't need them)

The fixes that landed between b7437 and b8149 in upstream llama.cpp are all in llama.cpp's **own model-level C++ code** — its GGUF loader, architecture registry, and compute graph builder. Ollama **does not use any of these**:

| llama.cpp fix | What it fixes | Why Ollama is unaffected |
|---|---|---|
| [PR #19468](https://github.com/ggml-org/llama.cpp/pull/19468): `qwen35`/`qwen35moe` architecture | llama.cpp's C++ model registry didn't recognize the GGUF `general.architecture` field | Ollama registers architectures in Go: `model.Register("qwen35", New)` in [`model/models/qwen3next/model.go:622-626`](https://github.com/ollama/ollama/blob/cc90a035/model/models/qwen3next/model.go#L622) |
| [PR #19870](https://github.com/ggml-org/llama.cpp/pull/19870): `ftell`/`fseek` overflow on >2GB GGUFs | llama.cpp's C GGUF loader used 32-bit file offsets on Windows | Ollama loads GGUFs entirely in Go via its own `fs/ggml` package |
| [PR #19866](https://github.com/ggml-org/llama.cpp/pull/19866): multi-GPU graph split ordering | llama.cpp's C++ graph scheduler ordered nodes incorrectly for Qwen 3.5 | Ollama has its own Go graph scheduler |
| [PR #19324](https://github.com/ggml-org/llama.cpp/pull/19324): `key_gdiff` vectorized calculation | llama.cpp's C++ Delta Net code was missing a `reshape_4d` before multiplying `k * g_diff_exp`, causing wrong broadcasting and degraded output quality past ~5000 tokens | Ollama's Go implementation at [`deltanet.go:434-436`](https://github.com/ollama/ollama/blob/cc90a035/model/models/qwen3next/deltanet.go#L434) independently includes the correct reshape: `gDiffExp.Reshape(ctx, 1, chunkSize, nChunks, ...)` before the multiply |

### What Ollama actually uses from llama.cpp

Ollama only uses the **GGML tensor computation backends** (CUDA, Vulkan, Metal, CPU kernels) from the vendored llama.cpp. These are the low-level math kernels — matrix multiply, softmax, RoPE, etc. Ollama builds the entire compute graph in Go and hands it to GGML for execution.

All of the GGML ops required by Qwen 3.5's Gated Delta Net architecture (`SSM_CONV`, `SOLVE_TRI`, `CUMSUM`, `TRI`, `SOFTPLUS`, `L2_NORM`, `DIAG`, `FILL`) already have CUDA implementations in the vendored b7437 code. These ops were added in [PR #17063](https://github.com/ggml-org/llama.cpp/pull/17063) (merged November 13, 2025) and [PR #17584](https://github.com/ggml-org/llama.cpp/pull/17584) (merged December 4, 2025) — both before b7437.

### What could still be a concern

Generic GGML kernel-level fixes that affect all models (not Qwen 3.5 specifically) have landed after b7437, including flash attention numerical stability improvements and a cuBLAS FP16 overflow fix ([PR #19959](https://github.com/ggml-org/llama.cpp/pull/19959)) that caused degenerate output on V100 and Blackwell GPUs. These are not Qwen 3.5 architecture issues — they would affect any model on those GPU architectures.
