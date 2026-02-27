# Qwen 3.5 27B on Ollama: Client-Side Usage Guide

> **Fork:** [`BigBIueWhale/ollama`](https://github.com/BigBIueWhale/ollama) at commit [`ab234955`](https://github.com/BigBIueWhale/ollama/commit/ab234955) (Feb 28, 2026 UTC)
> **Base:** Ollama v0.17.4 ([`cc90a035`](https://github.com/ollama/ollama/commit/cc90a035a0cc3ae9bd0c1dc95d42b620e8dcb0e2))
> **Model:** `qwen3.5:27b` (Q4_K_M, ~16.2 GB on disk)
> **Hardware:** NVIDIA RTX 5090 (32 GB VRAM)

This document describes how to use the Qwen 3.5 27B dense model for agentic tool calling with thinking via our Ollama fork. The fork addresses two independent sets of issues in upstream Ollama v0.17.4:

1. **Four critical Qwen 3.5 bugs** that make tool calling completely non-functional — documented in the [Qwen 3.5 27B inference report](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md) and filed as [ollama/ollama#14493](https://github.com/ollama/ollama/issues/14493).

2. **Tokenizer performance bugs** that cause 13+ seconds of CPU burn before GPU inference on long conversations — documented in the [Ollama performance deep dive](https://github.com/BigBIueWhale/ollama_perf_bug_report/blob/master/PERFORMANCE_DEEP_DIVE.md) and [benchmark results](https://github.com/BigBIueWhale/ollama_perf_bug_report/blob/master/BENCHMARK_RESULTS.md).

---

## All Changes in This Fork

Every change in this fork has been [verified against the HuggingFace ground truth templates](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) for correctness. Performance changes are output-identical to upstream — they produce the same results faster. All changes have passing tests.

| Change | Affects | Category | Risk |
|--------|---------|----------|------|
| [Penalty sampling](https://github.com/BigBIueWhale/ollama/blob/ab234955/sample/transforms.go#L120) now implemented in Go runner | All Go-runner models | Qwen 3.5 fix | **High** — default `repeat_penalty=1.1` was silently ignored, now applied |
| [Renderer/parser rewired](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/renderer.go#L59) from Qwen3VL (JSON) to Qwen3Coder (XML) | `qwen3.5` only | Qwen 3.5 fix | Low |
| [Thinking support](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L58) added to Coder renderer/parser | `qwen3.5` only | Qwen 3.5 fix | Low |
| [Prefill excludes tool calls](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L148) — generation prompt emitted after tool call turns | `qwen3.5`, `qwen3-coder`, `qwen3-vl-*` | Qwen 3.5 fix | Medium — also fixes `qwen3-coder` and `qwen3-vl` models |
| [Tool call whitespace](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L163) matches HuggingFace template | `qwen3.5`, `qwen3-coder` | Qwen 3.5 fix | Low — verified against both [Qwen 3.5](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) and [Qwen3-Coder](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/tokenizer_config.json) templates |
| [`</think>` always closed](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3vl.go#L98) in VL renderer | `qwen3-vl-thinking` | Qwen 3.5 fix | Low |
| [Binary search truncation](https://github.com/BigBIueWhale/ollama/blob/ab234955/server/prompt.go#L33) — O(log N) instead of O(N) tokenize calls | All models | [Perf fix](https://github.com/BigBIueWhale/ollama_perf_bug_report/blob/master/PERFORMANCE_DEEP_DIVE.md#bug-1-the-on-truncation-loop-that-nobody-else-uses) | Low — output-identical |
| [`strings.Contains` early-out](https://github.com/BigBIueWhale/ollama/blob/ab234955/tokenizer/special.go#L28) + [`slices.Replace`](https://github.com/BigBIueWhale/ollama/blob/ab234955/tokenizer/special.go#L53) for special tokens | All models | [Perf fix](https://github.com/BigBIueWhale/ollama_perf_bug_report/blob/master/PERFORMANCE_DEEP_DIVE.md#bug-2-osf-special-token-scanning--1000-sequential-string-searches) | Low — output-identical, eliminates GC pressure |
| [Stack-allocated `Merge()` key](https://github.com/BigBIueWhale/ollama/blob/ab234955/tokenizer/vocabulary.go#L107) avoids heap allocation per BPE lookup | All models | [Perf fix](https://github.com/BigBIueWhale/ollama_perf_bug_report/blob/master/PERFORMANCE_DEEP_DIVE.md#bug-5-merge-creates-a-new-string-on-every-bpe-lookup) | Low — output-identical |

**High-risk note:** The penalty sampling fix is the only change that alters observable output for models other than Qwen 3.5. Upstream Ollama's Go runner silently discarded `repeat_penalty`, `frequency_penalty`, and `presence_penalty` — accepting them via the API but applying no effect. This fork makes them work. The default `repeat_penalty=1.1` (from `DefaultOptions()`) will now be applied to all Go-runner models, matching what the C++ runner (llamarunner) already does.

---

## Quick Start

Pull the model, then call the API. Thinking is enabled by default for Qwen 3.5 — no flag needed.

```bash
ollama pull qwen3.5:27b
```

### Recommended: Thinking Mode for General Agentic Use

This is what you should use for most tasks. Parameters match the [Qwen 3.5 27B model card "Best Practices"](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices) table for **Thinking — General tasks**, with `num_predict` set to 81,920 (the model card's recommendation for complex problems).

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:27b",
  "messages": [
    {"role": "user", "content": "Solve: Find all integers n such that n^2 + 2n + 4 is divisible by 7."}
  ],
  "stream": false,
  "options": {
    "num_ctx": 32768,
    "num_predict": 81920,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0,
    "seed": -1
  }
}'
```

### Recommended: Thinking Mode with Tool Calling

Same parameters, with tools added. The model thinks, then calls tools using the Qwen3-Coder XML format (`<function=name><parameter=key>value</parameter></function>`).

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:27b",
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "stream": false,
  "options": {
    "num_ctx": 32768,
    "num_predict": 81920,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0
  }
}'
```

When the model calls a tool, send the result back as a `tool` message and continue the conversation:

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:27b",
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo?"},
    {"role": "assistant", "thinking": "I should check the weather.", "tool_calls": [
      {"type": "function", "function": {"index": 0, "name": "get_weather", "arguments": {"city": "Tokyo"}}}
    ]},
    {"role": "tool", "tool_name": "get_weather", "content": "{\"temperature\": 22, \"condition\": \"partly cloudy\"}"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "stream": false,
  "options": {
    "num_ctx": 32768,
    "num_predict": 81920,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "presence_penalty": 1.5,
    "repeat_penalty": 1.0
  }
}'
```

---

## All Four Parameter Profiles

The [Qwen 3.5 27B model card](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices) defines four sampling profiles. We recommend always using thinking mode (the model's primary differentiator), but all four are listed here for completeness.

### Profile 1: Thinking — General Tasks (Recommended Default)

For agentic tool calling, math, reasoning, creative writing, and general use.

```json
"options": {
  "num_ctx": 32768,
  "num_predict": 81920,
  "temperature": 1.0,
  "top_k": 20,
  "top_p": 0.95,
  "min_p": 0.0,
  "presence_penalty": 1.5,
  "repeat_penalty": 1.0,
  "seed": -1
}
```

The model card's [code example](https://huggingface.co/Qwen/Qwen3.5-27B#text-generation) for thinking mode uses exactly `temperature=1.0, top_p=0.95, presence_penalty=1.5, top_k=20, max_tokens=81920`.

### Profile 2: Thinking — Precise Coding

For web development, code generation where determinism matters more than exploration.

```json
"options": {
  "num_ctx": 32768,
  "num_predict": 81920,
  "temperature": 0.6,
  "top_k": 20,
  "top_p": 0.95,
  "min_p": 0.0,
  "presence_penalty": 0.0,
  "repeat_penalty": 1.0,
  "seed": -1
}
```

### Profile 3: Non-Thinking — General Tasks

Disables thinking for faster, shorter responses. Must explicitly set `"think": false`.

```json
"think": false,
"options": {
  "num_ctx": 32768,
  "num_predict": 32768,
  "temperature": 0.7,
  "top_k": 20,
  "top_p": 0.8,
  "min_p": 0.0,
  "presence_penalty": 1.5,
  "repeat_penalty": 1.0,
  "seed": -1
}
```

### Profile 4: Non-Thinking — Hard Reasoning

For chain-of-thought reasoning without the `<think>` block. Uses aggressive sampling to encourage diverse reasoning paths.

```json
"think": false,
"options": {
  "num_ctx": 32768,
  "num_predict": 32768,
  "temperature": 1.0,
  "top_k": 40,
  "top_p": 1.0,
  "min_p": 0.0,
  "presence_penalty": 2.0,
  "repeat_penalty": 1.0,
  "seed": -1
}
```

---

## What the Defaults Are and Where They Come From

When you `ollama pull qwen3.5:27b` and call the API with no `options`, here is what actually happens. Three layers merge in order: Ollama code defaults, then model params blob from the registry, then your API request (highest priority).

### Layer 1: Ollama Code Defaults

Hardcoded in [`api/types.go:1054` (`DefaultOptions()`)](https://github.com/BigBIueWhale/ollama/blob/ab234955/api/types.go#L1054):

| Parameter | Code Default |
|---|---|
| `temperature` | 0.8 |
| `top_k` | 40 |
| `top_p` | 0.9 |
| `min_p` | 0.0 |
| `repeat_penalty` | 1.1 |
| `presence_penalty` | 0.0 |
| `frequency_penalty` | 0.0 |
| `repeat_last_n` | 64 |
| `num_predict` | -1 (unlimited) |
| `num_keep` | 4 |
| `seed` | -1 (random) |
| `num_ctx` | 0 (triggers VRAM-based selection) |
| `num_batch` | 512 |

### Layer 2: Model Params Blob from the Ollama Library

The `qwen3.5:27b` (Q4_K_M) manifest references a 42-byte params blob containing exactly three parameters:

```json
{"temperature":1,"top_k":20,"top_p":0.95}
```

These three values overwrite the code defaults. Everything else (penalties, context length, max tokens, stop tokens) falls through to Layer 1.

### Layer 3: VRAM-Based Context Length

Since the params blob does not set `num_ctx`, and `DefaultOptions()` sets it to 0 (meaning "auto"), the VRAM-based tier system in [`server/routes.go:1758`](https://github.com/BigBIueWhale/ollama/blob/ab234955/server/routes.go#L1758) applies:

| Total VRAM | Default `num_ctx` |
|---|---|
| >= 47 GiB | 262,144 (256K) |
| >= 23 GiB | **32,768 (32K)** |
| < 23 GiB | 4,096 (4K) |

An RTX 5090 with 32 GB VRAM hits the middle tier: **32,768 tokens of context**.

The model's native context length (`max_position_embeddings` in [`config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json)) is 262,144 tokens (256K). The GGUF file's `qwen3.context_length` metadata caps `num_ctx` at this value — you cannot exceed 262,144 even if you try.

### Effective Defaults (No User Overrides)

Merging all three layers, here is what you get if you call the API with zero options on an RTX 5090:

| Parameter | **Effective Value** | Source | Model Card Recommends |
|---|---|---|---|
| `temperature` | **1.0** | params blob | 1.0 (general thinking) |
| `top_k` | **20** | params blob | 20 |
| `top_p` | **0.95** | params blob | 0.95 |
| `min_p` | **0.0** | code default | 0.0 |
| `repeat_penalty` | **1.1** | code default | **1.0** (disabled) |
| `presence_penalty` | **0.0** | code default | **1.5** |
| `frequency_penalty` | **0.0** | code default | 0.0 |
| `repeat_last_n` | **64** | code default | — |
| `num_predict` | **-1** (unlimited) | code default | 81,920 (complex) |
| `num_ctx` | **32,768** | VRAM tier | — |
| `seed` | **-1** (random) | code default | — |
| `think` | **true** | auto-enabled | true |

### What You Must Override

Two parameters are wrong by default:

1. **`repeat_penalty`**: Ollama defaults to 1.1 (enabled). The model card says 1.0 (disabled) across all four profiles. Set `"repeat_penalty": 1.0` in every request.

2. **`presence_penalty`**: Ollama defaults to 0.0 (disabled). The model card recommends 1.5 for general thinking and non-thinking modes. Set `"presence_penalty": 1.5` in every request (except precise coding, where 0.0 is correct).

Everything else happens to align between the Ollama defaults and the model card for the general thinking profile — `temperature: 1.0`, `top_k: 20`, `top_p: 0.95` all come from the params blob and match the model card.

### What `num_predict` Should Be

The model card recommends two tiers of maximum output length:

| Task Complexity | `num_predict` | Source |
|---|---|---|
| Standard queries | 32,768 | [Model card](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices) |
| Math/programming competitions | **81,920** | [Model card](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices) |

For context: the [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388) used 38,912 tokens for AIME benchmarks with Qwen 3, but Qwen 3.5 bumped the recommendation to 81,920 — a 2.1x increase reflecting longer thinking chains in the newer model.

The Ollama default of `num_predict: -1` (unlimited) is safe — the model will hit its EOS token naturally. Setting `81920` explicitly caps runaway generation while providing enough budget for AIME-class problems. We recommend 81,920 as the default for all agentic use.

---

## What the Fork Fixes

Upstream Ollama v0.17.1 through v0.17.4 has four bugs that make Qwen 3.5 27B agentic tool calling completely non-functional. Our fork fixes all four. See the [full issue](https://github.com/ollama/ollama/issues/14493) and [inference report](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md) for source-level evidence.

### Bug 1: Penalty Sampling Silently Ignored

The Go runner's `NewSampler()` in [`sample/samplers.go`](https://github.com/ollama/ollama/blob/cc90a035/sample/samplers.go#L130) accepted 6 parameters (temperature, topK, topP, minP, seed, grammar) and silently discarded `repeat_penalty`, `frequency_penalty`, `presence_penalty`, and `repeat_last_n`. The API accepted these values without error but they had zero effect.

**Fix:** Extended `Sampler` struct with penalty fields, added [`applyPenalties()`](https://github.com/BigBIueWhale/ollama/blob/ab234955/sample/transforms.go#L120) in `sample/transforms.go`, wired all four parameters through [`NewSampler()`](https://github.com/BigBIueWhale/ollama/blob/ab234955/runner/ollamarunner/runner.go#L890). This affects all models on the Go runner, not just Qwen 3.5.

### Bug 2: Wrong Tool Calling Format + Missing Thinking Support

The `"qwen3.5"` renderer/parser name mapped to `Qwen3VLRenderer`/`Qwen3Parser` — the **Qwen 3 Hermes-style JSON** tool calling pipeline. Qwen 3.5 was trained on the **Qwen3-Coder XML** format, as confirmed by the [HuggingFace chat template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json). The correct `Qwen3CoderRenderer`/`Qwen3CoderParser` existed but had zero thinking support.

**Fix:** Extended `Qwen3CoderRenderer` with `<think>`/`</think>` emission and `Qwen3CoderParser` with a full streaming state machine for thinking states. Rewired `"qwen3.5"` to the extended Coder pipeline in [`renderer.go`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/renderer.go#L59) and [`parsers.go`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/parsers/parsers.go#L52). Tool call whitespace [matches the HuggingFace Jinja2 ground truth](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L163).

### Bug 3: Unclosed `</think>` Tag

When an assistant message had thinking + tool calls but no text content, `Qwen3VLRenderer` never emitted `</think>`. The tool call was rendered inside an unclosed `<think>` block.

**Fix:** Always emit `\n</think>\n\n` after thinking content in [`qwen3vl.go`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3vl.go#L98), regardless of whether `content` is empty. (This also affects `qwen3-vl-instruct` and `qwen3-vl-thinking` models.)

### Bug 4: Missing Generation Prompt After Tool Calls

The `prefill` variable in both renderers fired for any last assistant message, including ones with tool calls. This suppressed `<|im_end|>` and the `<|im_start|>assistant\n` generation prompt.

**Fix:** Changed `prefill` to exclude tool call messages in both [`qwen3coder.go`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L148) and [`qwen3vl.go`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3vl.go#L82): `prefill := lastMessage && message.Role == "assistant" && len(message.ToolCalls) == 0`.

---

## The Prompt Template is Hardcoded in Go, Not From the Ollama Library

Unlike older Ollama models that ship a Go template blob (e.g. Qwen 3 ships a 1,723-byte template in the `application/vnd.ollama.image.template` manifest layer), **Qwen 3.5 has no template blob in its registry manifest**. The manifest contains only three layers: model weights, license, and the 42-byte params blob.

The entire chat prompt — system message formatting, tool definition rendering, tool call rendering, tool response formatting, thinking block handling, and generation prompt — is hardcoded in Go:

- **Renderer:** [`Qwen3CoderRenderer`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L58) (our fork; upstream incorrectly uses `Qwen3VLRenderer`)
- **Parser:** [`Qwen3CoderParser`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/parsers/qwen3coder.go#L31) (our fork; upstream incorrectly uses `Qwen3Parser`)

The mapping from the registry's `renderer: "qwen3.5"` / `parser: "qwen3.5"` to these Go structs lives in [`renderer.go:59`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/renderer.go#L59) and [`parsers.go:52`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/parsers/parsers.go#L52).

This means: **you cannot customize the chat template via a Modelfile `TEMPLATE` directive.** The `TEMPLATE` directive only works for models that use the Go template engine (like Devstral, Llama, etc.). Qwen 3.5 uses a builtin renderer, so the template is whatever the Go code says it is. The only way to change the prompt format is to modify the Go source.

---

## Thinking Behavior

Thinking is enabled by default for Qwen 3.5. When the model responds, the API returns both `thinking` and `content` fields:

```json
{
  "message": {
    "role": "assistant",
    "thinking": "Let me work through this step by step...",
    "content": "The answer is 42."
  }
}
```

To disable thinking, pass `"think": false` at the top level of the request. The Qwen 3.5 model card notes that the `/think` and `/nothink` soft switch from Qwen 3 is **not supported** in Qwen 3.5 — thinking must be controlled via the `think` parameter (which maps to the template's `enable_thinking` variable).

When thinking is disabled (`think: false`), the renderer emits an empty think block (`<think>\n\n</think>\n\n`) before the generation prompt. This is the `emitEmptyThinkOnNoThink` behavior, which matches the HuggingFace template's handling of `enable_thinking=False`.

There is **no automatic parameter adjustment** when thinking is toggled. Temperature, penalties, and all other sampling parameters remain exactly as you set them regardless of whether thinking is on or off. If you switch from thinking to non-thinking mode, you should also change your sampling parameters to match the appropriate profile (see the four profiles above).

---

## Context Length Considerations

The model supports up to 262,144 tokens (256K) natively. The model card states a minimum of 128K tokens is recommended "to preserve thinking capabilities."

On an RTX 5090 with 32 GB VRAM and Q4_K_M quantization (~16.2 GB weights), the remaining ~15.8 GB is available for KV cache. The default `num_ctx: 32768` (32K) is conservative and fits comfortably.

To request more context, set `num_ctx` in the `options` object or via the `OLLAMA_CONTEXT_LENGTH` environment variable. The GGUF metadata caps it at 262,144 regardless of what you request.

| `num_ctx` | VRAM for KV Cache (approx) | Fits RTX 5090? |
|---|---|---|
| 32,768 (32K, default) | ~4 GB | Yes |
| 65,536 (64K) | ~8 GB | Yes |
| 131,072 (128K) | ~16 GB | Tight, may OOM |
| 262,144 (256K) | ~32 GB | No (exceeds total VRAM) |

The safe maximum for single-user inference on RTX 5090 at Q4_K_M is approximately 65,536 (64K). Going higher requires monitoring actual VRAM usage.

---

## OpenAI-Compatible API

Ollama exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1/`. This works with the OpenAI Python client and any tool that speaks the OpenAI chat completions format.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",  # required by client library, ignored by Ollama
)

response = client.chat.completions.create(
    model="qwen3.5:27b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    max_tokens=81920,
)
```

**Limitations of the OpenAI-compatible endpoint:**

- `num_ctx` cannot be set — use a custom Modelfile or the native `/api/chat` endpoint.
- `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n` cannot be set — these are Ollama-specific parameters not in the OpenAI spec.
- `max_tokens` maps to Ollama's `num_predict`.
- `tool_choice` is not supported.

If you need full control over all parameters, use the native `/api/chat` endpoint.

---

## Sources

- [Qwen 3.5 27B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-27B) — parameter profiles, output length recommendations, deployment examples
- [Qwen 3.5 27B `generation_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/generation_config.json) — shipped defaults: temperature=0.6, top_k=20, top_p=0.95
- [Qwen 3.5 27B `config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json) — max_position_embeddings=262144
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388) — AIME benchmark methodology: 38,912 max tokens, 64 samples/question, temperature=0.6
- [Ollama Issue #14493](https://github.com/ollama/ollama/issues/14493) — the four bugs documented with source-level evidence
- [Inference Report](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md) — full source-level analysis
- [HuggingFace Chat Template (ground truth)](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) — Qwen3-Coder XML tool calling format
