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

- Ollama v0.17.1 through master (`79917cf`, Feb 26 2026)
- Model: `qwen3.5:27b-q4_K_M`
- Full report with source-level verification: [qwen3.5_27b_inference_report.md](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md)
