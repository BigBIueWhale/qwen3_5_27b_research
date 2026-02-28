# Qwen 3.5 27B on Ollama — Working Fork

> **Fork:** [`BigBIueWhale/ollama`](https://github.com/BigBIueWhale/ollama) at commit [`ab234955`](https://github.com/BigBIueWhale/ollama/commit/ab234955) (Feb 28, 2026 UTC)
> **Base:** Ollama v0.17.4 ([`cc90a035`](https://github.com/ollama/ollama/commit/cc90a035a0cc3ae9bd0c1dc95d42b620e8dcb0e2))
> **Model:** `qwen3.5:27b` (Q4_K_M, ~16.2 GB on disk)

Upstream Ollama v0.17.4 ships Qwen 3.5 27B with broken agentic tool calling — wrong prompt format, silently ignored penalties, and malformed thinking blocks. This fork fixes all four bugs and includes three tokenizer performance improvements from a [separate investigation](https://github.com/BigBIueWhale/ollama_perf_bug_report). All changes have passing tests and have been [verified against the HuggingFace ground truth template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json).

---

## What This Fork Changes

| Change | Affects | Category | Risk |
|--------|---------|----------|------|
| [Penalty sampling](https://github.com/BigBIueWhale/ollama/blob/ab234955/sample/transforms.go#L120) now implemented in Go runner | All Go-runner models | Qwen 3.5 fix | **High** — default `repeat_penalty=1.1` was silently ignored, now applied |
| [Renderer/parser rewired](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/renderer.go#L59) from Qwen3VL (JSON) to Qwen3Coder (XML) | `qwen3.5` only | Qwen 3.5 fix | Low |
| [Thinking support](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L58) added to Coder renderer/parser | `qwen3.5` only | Qwen 3.5 fix | Low |
| [Prefill excludes tool calls](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L148) — generation prompt emitted after tool call turns | `qwen3.5`, `qwen3-coder`, `qwen3-vl-*` | Qwen 3.5 fix | Medium — also fixes `qwen3-coder` and `qwen3-vl` models |
| [Tool call whitespace](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L163) matches HuggingFace template | `qwen3.5`, `qwen3-coder` | Qwen 3.5 fix | Low — verified against both [Qwen 3.5](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) and [Qwen3-Coder](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/tokenizer_config.json) templates |
| [`</think>` always closed](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3vl.go#L98) in VL renderer | `qwen3-vl-thinking` | Qwen 3.5 fix | Low |
| [Binary search truncation](https://github.com/BigBIueWhale/ollama/blob/ab234955/server/prompt.go#L33) — O(log N) instead of O(N) tokenize calls | All models | [Perf fix](https://github.com/BigBIueWhale/ollama_perf_bug_report) | Low — output-identical |
| [`strings.Contains` early-out](https://github.com/BigBIueWhale/ollama/blob/ab234955/tokenizer/special.go#L28) + [`slices.Replace`](https://github.com/BigBIueWhale/ollama/blob/ab234955/tokenizer/special.go#L53) for special tokens | All models | [Perf fix](https://github.com/BigBIueWhale/ollama_perf_bug_report) | Low — output-identical |
| [Stack-allocated `Merge()` key](https://github.com/BigBIueWhale/ollama/blob/ab234955/tokenizer/vocabulary.go#L107) avoids heap allocation per BPE lookup | All models | [Perf fix](https://github.com/BigBIueWhale/ollama_perf_bug_report) | Low — output-identical |

**High-risk note:** Penalty sampling is the only change that alters observable output for models other than Qwen 3.5. Upstream Ollama's Go runner silently discarded `repeat_penalty`, `frequency_penalty`, and `presence_penalty` — accepting them via the API but applying no effect. This fork makes them work. The default `repeat_penalty=1.1` (from `DefaultOptions()`) will now be applied to all Go-runner models, matching what the C++ runner (llamarunner) already does.

**Performance note:** The three tokenizer fixes reduce per-call tokenizer overhead by [approximately 4x](https://github.com/BigBIueWhale/ollama_perf_bug_report/blob/master/BENCHMARK_RESULTS.md) (0.56s → 0.13s TTFB on a 151-message, 65 KB test payload). These bugs are still present in upstream Ollama v0.17.4. See the [full performance investigation](https://github.com/BigBIueWhale/ollama_perf_bug_report) for profiling data and benchmark methodology.

---

## Quick Start

Pull the model, then call the API. Thinking is enabled by default for Qwen 3.5 — no flag needed.

```bash
ollama pull qwen3.5:27b-q4_K_M
```

**VRAM note:** At Q4_K_M with `num_ctx=131072` (128K), the model uses **~32.5 GiB VRAM** — fitting entirely on an RTX 5090 (32 GiB) with zero CPU offload.

### Thinking Mode with Tool Calling

Parameters match the [Qwen 3.5 27B model card](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices) "Thinking — General tasks" profile. The model thinks, then calls tools using the Qwen3-Coder XML format.

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:27b-q4_K_M",
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
    "num_ctx": 131072,
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
  "model": "qwen3.5:27b-q4_K_M",
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
    "num_ctx": 131072,
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

### Thinking Mode without Tools

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5:27b-q4_K_M",
  "messages": [
    {"role": "user", "content": "Solve: Find all integers n such that n^2 + 2n + 4 is divisible by 7."}
  ],
  "stream": false,
  "options": {
    "num_ctx": 131072,
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

---

## Parameters You Must Override

Two parameters are wrong by default and must be set in every request:

1. **`repeat_penalty: 1.0`** — Ollama defaults to 1.1. The model card says 1.0 (disabled) across all four profiles.
2. **`presence_penalty: 1.5`** — Ollama defaults to 0.0. The model card recommends 1.5 for general thinking and non-thinking modes (0.0 only for precise coding).

Everything else aligns between the Ollama defaults and the model card for the general thinking profile — `temperature: 1.0`, `top_k: 20`, `top_p: 0.95` all come from the model's registry params blob.

---

## All Four Parameter Profiles

The [Qwen 3.5 27B model card](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices) defines four sampling profiles.

### Profile 1: Thinking — General Tasks (Recommended Default)

For agentic tool calling, math, reasoning, creative writing, and general use.

```json
"options": {
  "num_ctx": 131072,
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

### Profile 2: Thinking — Precise Coding (e.g. WebDev)

For precise coding tasks where determinism matters more than exploration.

```json
"options": {
  "num_ctx": 131072,
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
  "num_ctx": 131072,
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
  "num_ctx": 131072,
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

## Technical Details

### How Defaults Are Determined

Three layers merge in order: Ollama code defaults → model params blob → your API request (highest priority).

**Layer 1 — Ollama code defaults** ([`api/types.go:1054`](https://github.com/BigBIueWhale/ollama/blob/ab234955/api/types.go#L1054)): `temperature=0.8`, `top_k=40`, `top_p=0.9`, `repeat_penalty=1.1`, `presence_penalty=0.0`, `num_predict=-1` (unlimited), `num_ctx=0` (auto).

**Layer 2 — Model params blob** (42 bytes in registry manifest): `{"temperature":1,"top_k":20,"top_p":0.95}`. These three override Layer 1. Everything else falls through.

**Layer 3 — VRAM-based context length** ([`server/routes.go:1758`](https://github.com/BigBIueWhale/ollama/blob/ab234955/server/routes.go#L1758)): Since `num_ctx` is 0, the VRAM tier system applies. RTX 5090 (32 GB) gets 32,768 tokens (32K). The model supports up to 262,144 (256K) natively.

**Effective defaults with no overrides on RTX 5090:**

| Parameter | Effective Value | Source | Model Card Recommends |
|---|---|---|---|
| `temperature` | 1.0 | params blob | 1.0 |
| `top_k` | 20 | params blob | 20 |
| `top_p` | 0.95 | params blob | 0.95 |
| `repeat_penalty` | **1.1** | code default | **1.0** |
| `presence_penalty` | **0.0** | code default | **1.5** |
| `num_predict` | -1 (unlimited) | code default | 81,920 |
| `num_ctx` | 32,768 | VRAM tier | — |

### `num_predict` Recommendations

The model card recommends 32,768 for standard queries and **81,920** for math/programming competitions. The Ollama default of -1 (unlimited) is safe — the model hits EOS naturally. Setting 81,920 caps runaway generation while providing enough budget for complex problems.

### Thinking Behavior

Thinking is enabled by default. The API returns both `thinking` and `content` fields:

```json
{
  "message": {
    "role": "assistant",
    "thinking": "Let me work through this step by step...",
    "content": "The answer is 42."
  }
}
```

To disable, pass `"think": false` at the request top level. The Qwen 3.5 model card notes that `/think` and `/nothink` soft switches from Qwen 3 are **not supported** — use the `think` parameter.

When thinking is disabled, the renderer emits an empty think block (`<think>\n\n</think>\n\n`) before the generation prompt, matching the HuggingFace template's `enable_thinking=False` behavior. There is no automatic parameter adjustment — change sampling parameters to match the appropriate profile when switching modes.

### Prompt Template

The Qwen 3.5 prompt template is **hardcoded in Go**, not loaded from the Ollama registry. The manifest contains no template blob — only weights, license, and the 42-byte params blob. The chat prompt is built by:

- **Renderer:** [`Qwen3CoderRenderer`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/renderers/qwen3coder.go#L58) (our fork; upstream incorrectly uses `Qwen3VLRenderer`)
- **Parser:** [`Qwen3CoderParser`](https://github.com/BigBIueWhale/ollama/blob/ab234955/model/parsers/qwen3coder.go#L31) (our fork; upstream incorrectly uses `Qwen3Parser`)

This means you **cannot customize the chat template via a Modelfile `TEMPLATE` directive**. The `TEMPLATE` directive only works for models that use the Go template engine. The only way to change the prompt format is to modify the Go source.

### Context Length on RTX 5090

| `num_ctx` | Total VRAM (model + KV cache) | Fits RTX 5090? |
|---|---|---|
| 32,768 (32K, default) | ~24.8 GiB | Yes |
| 65,536 (64K) | ~27.4 GiB | Yes |
| 131,072 (128K) | ~32.5 GiB | **Yes** — confirmed, zero CPU offload, ~118 MiB free |
| 140,000 | ~34.0 GiB | No — 4.7 GiB spills to CPU |
| 262,144 (256K) | ~48 GiB | No (far exceeds total VRAM) |

The maximum context that fits 100% on GPU at Q4_K_M on RTX 5090 is **131,072 (128K)**. The model card recommends a minimum of 128K "to preserve thinking capabilities."

### OpenAI-Compatible API

Ollama exposes an OpenAI-compatible endpoint at `http://localhost:11434/v1/`:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",  # required by client library, ignored by Ollama
)

response = client.chat.completions.create(
    model="qwen3.5:27b-q4_K_M",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    max_tokens=81920,
)
```

**Limitations:** `num_ctx`, `top_k`, `min_p`, `repeat_penalty`, `repeat_last_n` cannot be set via the OpenAI endpoint — use the native `/api/chat` endpoint for full control.

---

## Sources

- [Qwen 3.5 27B Model Card](https://huggingface.co/Qwen/Qwen3.5-27B) — parameter profiles, output length recommendations
- [Qwen 3.5 27B Chat Template (HuggingFace)](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) — ground truth for Qwen3-Coder XML tool calling format
- [Ollama Issue #14493](https://github.com/ollama/ollama/issues/14493) — the four bugs with source-level evidence
- [Qwen 3.5 Inference Report](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md) — full source-level analysis
- [Tokenizer Performance Investigation](https://github.com/BigBIueWhale/ollama_perf_bug_report) — profiling, benchmarks, and the three per-call tokenizer bugs
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388) — AIME benchmark methodology
