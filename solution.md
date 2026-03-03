# Qwen 3.5 27B on RTX 5090 — Working Setup

> **Fork:** [`BigBIueWhale/ollama`](https://github.com/BigBIueWhale/ollama) (`main` branch)
> **Base:** Ollama v0.17.4 ([`cc90a035`](https://github.com/ollama/ollama/commit/cc90a035a0cc3ae9bd0c1dc95d42b620e8dcb0e2))
> **Hardware:** NVIDIA RTX 5090 (32 GB VRAM)

Upstream Ollama v0.17.4 ships Qwen 3.5 27B with broken agentic tool calling — wrong prompt format, silently ignored penalties, and malformed thinking blocks. This fork fixes all four bugs and includes three tokenizer performance improvements from a [separate investigation](https://github.com/BigBIueWhale/ollama_perf_bug_report). All changes have passing tests and have been [verified against the HuggingFace ground truth template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json).

For full source-level evidence of each bug, see the [inference report](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md).

---

## Use the Unsloth UD-Q4_K_XL GGUF

**Use [`Qwen3.5-27B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) from Unsloth (March 2, 2026 re-upload, iteration 2).** This is the best available GGUF for Qwen 3.5 27B. Unsloth Dynamic 2.0 uses per-tensor quantization that protects the novel GDN/SSM layers at dramatically higher precision than any other option:

| Tensor (48 GDN blocks) | Ollama registry | Unsloth Q4_K_M | **Unsloth UD-Q4_K_XL** |
|---|---|---|---|
| `ssm_alpha` | Q4_K (4.5 bpw) | Q8_0 (8.5 bpw) | **F16 (16.0 bpw)** |
| `ssm_beta` | Q4_K (4.5 bpw) | Q8_0 (8.5 bpw) | **F16 (16.0 bpw)** |
| `ssm_out` | Q4_K (4.5 bpw) | Q5_K (5.5 bpw) | **Q8_0 (8.5 bpw)** |
| `attn_gate` | Q4_K (4.5 bpw) | Q4_K (4.5 bpw) | **Q5_K (5.5 bpw)** |
| `attn_qkv` | Q4_K (4.5 bpw) | Q5_K (5.5 bpw) | **Q5_K (5.5 bpw)** |

The UD-Q4_K_XL (17.6 GB) is ~0.9 GB larger than Q4_K_M (16.7 GB). That extra budget buys `ssm_alpha`/`ssm_beta` at full F16 (4× the precision of Q4_K in the registry, 2× the Q8_0 in Q4_K_M), and `ssm_out` at Q8_0 (nearly double Q5_K). These are the exact tensors that control the SSM's temporal memory dynamics — the novel architecture that makes Qwen 3.5 different from a standard transformer.

The March 2 re-upload (iteration 2) fixes the [MXFP4 bug](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/5) that affected the Feb 24 UD GGUFs. Zero MXFP4 tensors remain. The extra bit budget is funded by selective IQ4_XS on 6 of 64 FFN gate/up blocks — generic transformer layers where quality impact is minimal.

### Setup

```bash
# Download the text-only GGUF (skips the OCI vision projector that breaks ollama):
wget -O /tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf \
  https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-UD-Q4_K_XL.gguf

cat > Modelfile << 'EOF'
FROM /tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf

# Required for non-registry GGUFs: Ollama's built-in Qwen 3.5 chat template.
TEMPLATE {{ .Prompt }}
RENDERER qwen3.5
PARSER qwen3.5

PARAMETER num_ctx 131072
PARAMETER num_predict 81920
PARAMETER temperature 1.0
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.0
PARAMETER presence_penalty 1.5
EOF
ollama create qwen3.5-custom -f Modelfile
```

**VRAM note:** At 17.6 GB, the UD-Q4_K_XL is ~0.9 GB larger than Q4_K_M. On RTX 5090 (32 GB VRAM), 131K context was already at the edge with Q4_K_M (~32.5 GiB total, ~118 MiB free). The UD-Q4_K_XL will push 131K over the limit. Reduce `num_ctx` to 65536 if you hit OOM, or use Q4_K_M if you need the full 128K context.

**`TEMPLATE`, `RENDERER`, and `PARSER` are mandatory** for non-registry GGUFs. Ollama registry models (`ollama pull qwen3.5:...`) include these in their config blob automatically, but local GGUFs do not — without them, the model gets a bare `{{ .Prompt }}` template and tool calling silently breaks.

### Why Not the Ollama Registry Model?

The Ollama registry GGUF (`ollama pull qwen3.5:27b`) uses standard llama.cpp Q4_K_M quantization with no GDN/SSM layer awareness. Every GDN tensor — including `ssm_alpha`, `ssm_beta`, and `ssm_out` — gets the same Q4_K as generic FFN layers. The [GGUF Deep Dive](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md#gguf-deep-dive-ollama-registry-vs-unsloth-use-unsloth) in the inference report has the full per-tensor comparison.

The registry model does have simpler setup (no `wget`, no manual `TEMPLATE`/`RENDERER`/`PARSER`) and its bundled 450 MB vision encoder + 80 MB MTP head are unused dead weight. If setup friction matters more than model quality, use the registry model with the same parameter overrides.

### Unsloth GGUF Setup Friction

- **`FROM hf.co/...` is broken** — Unsloth's OCI manifest bundles a 927 MB CLIP vision projector alongside the text GGUF, causing a fatal `"split vision models aren't supported"` error in Ollama's Go engine. Must use `wget` to download the text GGUF directly.
- **Missing GGUF metadata** — third-party GGUFs lack `ssm.v_head_reordered` and store `head_count_kv` as a scalar instead of a per-layer array. Requires [this fork's compatibility fixes](third_party_gguf_compatibility.md) or the model produces garbage output.
- **No chat template** — must manually add `TEMPLATE {{ .Prompt }}`, `RENDERER qwen3.5`, `PARSER qwen3.5` or tool calling silently breaks.

For the full per-tensor comparison, see the [GGUF Deep Dive](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md#gguf-deep-dive-ollama-registry-vs-unsloth-use-unsloth) in the inference report.

---

## What This Fork Fixes

| Change | Affects | Risk |
|--------|---------|------|
| [Penalty sampling](https://github.com/BigBIueWhale/ollama/blob/main/sample/transforms.go#L120) implemented in Go runner | All Go-runner models | **High** — `repeat_penalty=1.1` was silently ignored, now applied |
| [Renderer/parser rewired](https://github.com/BigBIueWhale/ollama/blob/main/model/renderers/renderer.go#L59) from Qwen3VL (JSON) to Qwen3Coder (XML) | `qwen3.5` only | Low |
| [Thinking support](https://github.com/BigBIueWhale/ollama/blob/main/model/renderers/qwen3coder.go#L58) added to Coder renderer/parser | `qwen3.5` only | Low |
| [Historical thinking stripped](https://github.com/BigBIueWhale/ollama/blob/main/model/renderers/qwen3coder.go#L148) across rounds via `lastQueryIndex` scan | `qwen3.5` only | Low |
| [Prefill excludes tool calls](https://github.com/BigBIueWhale/ollama/blob/main/model/renderers/qwen3coder.go#L148) — generation prompt emitted after tool call turns | `qwen3.5`, `qwen3-coder`, `qwen3-vl-*` | Medium |
| [Tool call whitespace](https://github.com/BigBIueWhale/ollama/blob/main/model/renderers/qwen3coder.go#L163) matches HuggingFace template | `qwen3.5`, `qwen3-coder` | Low |
| [`</think>` always closed](https://github.com/BigBIueWhale/ollama/blob/main/model/renderers/qwen3vl.go#L98) in VL renderer | `qwen3-vl-thinking` | Low |
| [Third-party GGUF compatibility](third_party_gguf_compatibility.md) — architecture-based defaults for `ssm.v_head_reordered`, `rope.mrope_interleaved`, and `isRecurrent` computation | `qwen3.5`, `qwen35moe`, `qwen3next` | **High** — wrong defaults caused garbage output and wrong RoPE on all third-party GGUFs |
| [`ssm_dt.bias` tensor name](https://github.com/BigBIueWhale/ollama/blob/main/model/models/qwen3next/deltanet.go#L44) supported as alias for `ssm_dt` | `qwen3.5`, `qwen3next` | Low |
| [Binary search truncation](https://github.com/BigBIueWhale/ollama/blob/main/server/prompt.go#L33) — O(log N) instead of O(N) tokenize calls | All models | Low |
| [`strings.Contains` early-out](https://github.com/BigBIueWhale/ollama/blob/main/tokenizer/special.go#L28) + [`slices.Replace`](https://github.com/BigBIueWhale/ollama/blob/main/tokenizer/special.go#L53) | All models | Low |
| [Stack-allocated `Merge()` key](https://github.com/BigBIueWhale/ollama/blob/main/tokenizer/vocabulary.go#L107) avoids heap allocation per BPE lookup | All models | Low |

**Penalty note:** Penalty sampling is the only change that alters output for models other than Qwen 3.5. Upstream Ollama's Go runner silently discarded `repeat_penalty`, `frequency_penalty`, and `presence_penalty`. This fork makes them work. The default `repeat_penalty=1.1` will now apply to all Go-runner models.

**Performance note:** The three tokenizer fixes reduce per-call overhead by [~4x](https://github.com/BigBIueWhale/ollama_perf_bug_report/blob/master/BENCHMARK_RESULTS.md) (0.56s -> 0.13s TTFB on a 151-message, 65 KB payload).

---

## Parameters You Must Override

Two parameters are wrong by default and must be set in every request:

1. **`repeat_penalty: 1.0`** — Ollama defaults to 1.1. Model card says 1.0 (disabled) for all four profiles.
2. **`presence_penalty: 1.5`** — Ollama defaults to 0.0. Model card says 1.5 for general use (0.0 only for precise coding).

Everything else (`temperature: 1.0`, `top_k: 20`, `top_p: 0.95`) comes from the model's params blob and is already correct for the general thinking profile.

---

## Thinking Mode with Tool Calling

Uses the **OpenAI-compatible endpoint** (`/v1/chat/completions`). The model thinks, then calls tools using the Qwen3-Coder XML format.

```python
import json, urllib.request

API = "http://localhost:11434/v1/chat/completions"
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer ollama"}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"],
                              "description": "Temperature units"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results to return"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate"}
                },
                "required": ["expression"]
            }
        }
    }
]

def call_api(messages):
    payload = json.dumps({
        "model": "qwen3.5-custom",
        "messages": messages,
        "tools": TOOLS,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 81920,
        "stream": False
    }).encode()
    req = urllib.request.Request(API, data=payload, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())

# Simulated tool implementations
def simulate_tool(name, args):
    if name == "get_weather":
        return json.dumps({"city": args.get("city", "?"), "temperature": 14,
                           "condition": "overcast", "humidity": 72, "wind_kph": 18})
    elif name == "search_web":
        return json.dumps({"results": [{"title": "Tel Aviv Marathon 2026",
            "snippet": "The 2026 Tel Aviv Marathon took place Feb 27. "
                       "Winner: Eliud Kosgei, time 2:04:31. Over 40,000 participants."}]})
    elif name == "calculate":
        return json.dumps({"result": str(eval(args.get("expression", "0")))})
    return "{}"

# --- Turn 1: User asks a multi-part question ---
messages = [{"role": "user", "content":
    "I'm planning to go running in Tel Aviv today. What's the weather like, "
    "when was the last Tel Aviv marathon and who won, "
    "and what's 42.195 km in miles?"}]

r1 = call_api(messages)
choice1 = r1["choices"][0]["message"]
# finish_reason == "tool_calls", model issued 3 parallel tool calls:
#   get_weather({"city":"Tel Aviv","units":"celsius"})
#   search_web({"query":"last Tel Aviv marathon winner date","num_results":3})
#   calculate({"expression":"42.195 * 0.621371"})

# --- Turn 2: Send simulated tool results back ---
messages.append(choice1)
for tc in choice1["tool_calls"]:
    fn = tc["function"]
    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    messages.append({
        "role": "tool",
        "tool_call_id": tc["id"],
        "content": simulate_tool(fn["name"], args)
    })

r2 = call_api(messages)
# finish_reason == "stop", model synthesizes final answer from all 3 tool results
print(r2["choices"][0]["message"]["content"])
```

## Thinking Mode without Tools

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3.5-custom",
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

## All Four Parameter Profiles

From the [Qwen 3.5 27B model card](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices):

### Profile 1: Thinking — General (Recommended Default)

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

### Profile 2: Thinking — Precise Coding

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

### Profile 3: Non-Thinking — General

Disables thinking for faster, shorter responses. Must set `"think": false`.

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

Chain-of-thought reasoning without `<think>` blocks. Aggressive sampling encourages diverse reasoning paths.

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

## Context Length on RTX 5090

With **UD-Q4_K_XL** (17.6 GB model weight):

| `num_ctx` | Total VRAM (model + KV cache) | Fits? |
|---|---|---|
| 32,768 (32K, Ollama default) | ~25.7 GiB | Yes |
| 65,536 (64K) | ~28.3 GiB | Yes |
| **131,072 (128K)** | **~33.4 GiB** | **Tight — may spill ~1.4 GiB to CPU** |
| 262,144 (256K, model max) | ~49 GiB | No |

**Use `num_ctx: 131072`.** The model card recommends a minimum of 128K "to preserve thinking capabilities." At 17.6 GB the UD-Q4_K_XL is ~0.9 GB larger than Q4_K_M — 131K context may trigger minimal CPU offload depending on Ollama's VRAM allocation overhead. In practice this is a small fraction of KV and the quality gain from F16 SSM layers far outweighs the marginal KV spill.

Ollama defaults to 32K (VRAM tier system). You must override `num_ctx` explicitly.

---

## Thinking Behavior

Thinking is enabled by default. The API returns both `thinking` and `content` fields. To disable, pass `"think": false` at the request top level. The `/think` and `/nothink` soft switches from Qwen 3 are **not supported** in Qwen 3.5.

In multi-turn conversations, thinking traces from previous rounds are automatically stripped from the prompt — only the current round's reasoning is preserved. This matches the model's training distribution.

---

## OpenAI-Compatible API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama",
)

response = client.chat.completions.create(
    model="qwen3.5-custom",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=1.0,
    top_p=0.95,
    presence_penalty=1.5,
    max_tokens=81920,
)
```

**Limitation:** `num_ctx`, `top_k`, `min_p`, `repeat_penalty` cannot be set via the OpenAI endpoint — use the native `/api/chat` endpoint for full parameter control.

---

## Sources

- [Qwen 3.5 27B Model Card](https://huggingface.co/Qwen/Qwen3.5-27B) — parameter profiles, output length recommendations
- [Qwen 3.5 27B Chat Template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) — ground truth Qwen3-Coder XML tool calling format
- [unsloth/Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF) — **recommended GGUF**: UD-Q4_K_XL (March 2 re-upload) with F16 on ssm_alpha/beta, Q8_0 on ssm_out
- [Unsloth Qwen3.5 GGUF Benchmarks](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks) — per-tensor KL divergence analysis
- [Inference Report — GGUF Deep Dive](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md#gguf-deep-dive-ollama-registry-vs-unsloth-use-unsloth) — full per-tensor comparison from downloaded files
- [Inference Report — Full Bug Analysis](https://github.com/BigBIueWhale/qwen3_5_27b_research/blob/master/qwen3.5_27b_inference_report.md) — source-level evidence for all four bugs
- [Ollama Issue #14493](https://github.com/ollama/ollama/issues/14493) — the four bugs filed upstream
- [Tokenizer Performance Investigation](https://github.com/BigBIueWhale/ollama_perf_bug_report) — profiling, benchmarks, and the three tokenizer bugs
