# Ollama Fork vs Upstream: Precise Analysis

Comparing fork (`BigBIueWhale/ollama`, 7 commits atop `v0.17.4` merge) against upstream `ollama/ollama` master.
Official reference: Qwen 3.5 Jinja2 template from `Qwen/Qwen3.5-35B-A3B` (publicly accessible on HuggingFace).

---

## Part 1: What the Fork Should Fix

### 1.1 CRITICAL: `SetInplace` in deltanet.go — breaks Apple Silicon & Vulkan

**File**: `model/models/qwen3next/deltanet.go:466-473`

The fork uses `v.SetInplace()` to assemble chunk outputs in `deltaNetChunked`. This maps to `GGML_OP_SET`. Backend support:

| Backend | `GGML_OP_SET` | `GGML_OP_CONCAT` |
|---------|---------------|-------------------|
| CPU     | Yes           | Yes               |
| CUDA    | Yes           | Yes               |
| Metal   | **No**        | Yes               |
| Vulkan  | **No**        | Yes               |

`SetInplace` is the **only call to GGML_OP_SET in the entire codebase** (both fork and upstream). No other model uses it. The upstream deliberately avoided it with an explicit comment: `"Avoids SET on buffer-less intermediates under partial offload."` and uses a balanced binary concat tree instead.

The qwen3next model registers normally for all backends — there is no GPU restriction. On Apple Silicon (the primary Mac inference path), recurrent layers offloaded to Metal will crash.

**Fix**: Replace with upstream's balanced concat tree pattern (O(log N) depth, universally supported ops).

### 1.2 MEDIUM-HIGH: Tool definition format — Qwen3.5 was trained on JSON, not XML

**File**: `model/renderers/qwen3coder.go:90-133`

There are **two different official templates** from Qwen:

| Template | Tool definitions | System+tools order |
|----------|-----------------|-------------------|
| **Qwen3-Coder** | XML `<function><name>...` | System first, tools second |
| **Qwen3.5**     | JSON `{{ tool \| tojson }}` | Tools first, system appended after |

The fork's `Qwen3CoderRenderer` uses the Qwen3-Coder XML format for **both** models. This is correct for `qwen3-coder` but wrong for `qwen3.5`. The model sees a completely different token sequence than what it was trained on:

```
Official:  {"type": "function", "function": {"name": "get_weather", ...}}
Fork:      <function>\n<name>get_weather</name>\n<description>...
```

Note: the **tool call format** (`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`) and the **instruction text** are character-for-character identical between both official templates and the fork. Only the tool definition rendering differs.

**Fix**: For `qwen3.5`, render tools as raw JSON (`tool | tojson`) inside `<tools>` tags, matching the official Qwen3.5 template. Keep XML for `qwen3-coder`.

### 1.3 MEDIUM-HIGH: System+tools ordering reversed for Qwen3.5

**File**: `model/renderers/qwen3coder.go:80-138`

The official Qwen 3.5 template renders:
```
<|im_start|>system
# Tools
...tools...
<IMPORTANT>...</IMPORTANT>

[user's system message]
<|im_end|>
```

The fork renders system message first, tools second. The Qwen3-Coder template does it the fork's way (system first). So this is correct for qwen3-coder but wrong for qwen3.5.

**Fix**: For `qwen3.5`, emit tools block first, append system message content after `</IMPORTANT>`.

### 1.4 MEDIUM: Missing `repeatPenalty <= 0` guard

**File**: `sample/samplers.go`

The API (`api/types.go` `FromMap()`) accepts any float64 for `repeat_penalty` with no range validation. With `repeatPenalty = 0`: division by zero produces `+Inf` logits. With negative: penalty inverts, boosting repeated tokens.

**Fix**: `if repeatPenalty <= 0 { repeatPenalty = 1.0 }`

### 1.5 MEDIUM: Missing nil checks + Validate() for third-party GGUFs

**Files**: `model/models/qwen3next/deltanet.go`, `model/models/qwen3next/model.go`

The fork explicitly targets third-party GGUF compatibility (3 commits fixing Unsloth/llama.cpp GGUFs). But it lacks defensive validation against GGUFs with **missing tensors**. The ollama converter always produces complete GGUFs, so these checks only matter for corrupted/truncated third-party files. Still, given the fork's third-party focus, the contradiction should be resolved.

Upstream has both inline nil checks in `Forward()` and a `Validate()` method (called via the `model.Validator` interface in `model.New()` after tensor loading).

**Fix**: Add `Validate()` method and nil checks. These are defense-in-depth — won't trigger for well-formed GGUFs but provide clear errors instead of panics for broken ones.

### 1.6 LOW: No image/vision support in Qwen3CoderRenderer

**File**: `model/renderers/qwen3coder.go`

The official Qwen 3.5 template supports `<|vision_start|><|image_pad|><|vision_end|>` tags. The fork's renderer writes `message.Content` as a plain string. If Qwen 3.5 27B is text-only in practice (likely for the agentic CLI use case), this is a non-issue.

### 1.7 LOW: Default system message injected when not provided

**File**: `model/renderers/qwen3coder.go:83-85`

The fork injects `"You are Qwen, a helpful AI assistant..."` when tools are present but no system message given. The official template does not. Minor but unnecessary divergence.

---

## Part 2: What Upstream Should Fix

### 2.1 CRITICAL: Wrong parser/renderer for qwen3.5

**Files**: `model/parsers/parsers.go:53`, `model/renderers/renderer.go:59-60`

Upstream maps `"qwen3.5"` to `Qwen3Parser` (JSON tool calls) + `Qwen3VLRenderer` (JSON tool definitions). The official Qwen 3.5 template uses XML-style tool calls (`<function=name><parameter=key>value</parameter></function>`).

**Tool calling is completely broken for Qwen 3.5 on upstream.** The model outputs XML-style calls but the parser expects JSON. The fork correctly routes through `Qwen3CoderParser`/`Qwen3CoderRenderer`.

### 2.2 CRITICAL: `</think>` not closed when content is empty

**File**: `model/renderers/qwen3vl.go:95-99`

Upstream only emits `\n</think>\n\n` when `content != ""`. For thinking-only assistant messages, the `<think>` tag is left **unclosed**. The official template always closes it. This corrupts the model's view of the conversation — it sees an open `<think>` block and doesn't know thinking has ended, producing degraded continuation.

### 2.3 HIGH: Prefill triggers on assistant messages with tool calls

**Files**: `model/renderers/qwen3coder.go:140`, `model/renderers/qwen3vl.go:82`

Upstream: `prefill := lastMessage && message.Role == "assistant"` — any last assistant message is a prefill, even one with tool calls. This means after tool calls, the model doesn't get a fresh `<|im_start|>assistant\n<think>\n` generation prompt. It tries to continue the previous assistant message instead of responding to tool results.

Fork correctly adds `&& len(message.ToolCalls) == 0`.

### 2.4 HIGH: No thinking support in Qwen3CoderParser/Renderer

Upstream's `Qwen3CoderParser.HasThinkingSupport()` returns `false`. `Qwen3CoderRenderer` ignores `ThinkValue` entirely (parameter named `_`). Even if upstream fixed the parser/renderer routing, the coder variants can't handle Qwen 3.5's thinking mode. The fork added full thinking support to both.

### 2.5 HIGH: No lastQueryIndex in Qwen3CoderRenderer

Upstream's `Qwen3CoderRenderer` has zero `lastQueryIndex` logic. Historical thinking traces from previous conversation rounds are included in the prompt, wasting context window and potentially confusing the model. The official template explicitly strips thinking from messages before the last real user query. The fork implements this.

### 2.6 MEDIUM: Tool call whitespace wrong

Upstream always emits `\n<tool_call>`. Official template has three-way logic. Fork matches exactly.

### 2.7 MEDIUM: `repeat_last_n` API parameter silently ignored

**File**: `sample/samplers.go`

Upstream's `NewSampler` does not accept `repeatLastN`. The constant `DefaultPenaltyLookback = 64` is hardcoded. A user setting `repeat_last_n: 128` in the API has **zero effect** — the value flows through `api/types.go` into `req.Options.RepeatLastN` but is never read by the sampler. Silently broken API contract.

The fork's `NewSampler` accepts `repeatLastN` and sizes the ring buffer accordingly.

### 2.8 MEDIUM: `mropeInterleaved` defaults to `false` — wrong for third-party qwen35 GGUFs

**File**: `model/models/qwen3next/model.go:641`

Third-party GGUFs (llama.cpp converter) don't write the ollama-internal `rope.mrope_interleaved` key. For qwen35/qwen35moe, the correct value is `true` (Interleaved MRoPE — llama.cpp hardcodes `LLAMA_ROPE_TYPE_IMROPE`). Upstream defaults to `false`, producing wrong position embeddings across all 16 full-attention layers.

The fork correctly defaults to `c.Architecture() != "qwen3next"`.

### 2.9 LOW: Upstream's `repeat_penalty` defaults to 1.0 — entire penalty system is dead code

With `repeatPenalty = 1.0` (identity), `frequencyPenalty = 0.0`, `presencePenalty = 0.0`: the penalty math is a complete no-op. `logit / 1.0 = logit`, `logit - 0.0*count = logit`, `logit - 0.0 = logit`. The entire `Accept()`/`Reset()` machinery in runner.go runs but has **zero effect on output**. The sampler maintains a history that is never meaningfully consulted.

The fork sets `RepeatPenalty: 1.1`, making penalties actually functional — a ~9% logit reduction for repeated tokens.

---

## Part 3: Things That Look Neutral But Are Actually Critical

### 3.1 Prompt tokens excluded from penalty — CORRECT for agentic use, not a design choice

**Why the fork removed `sampler.Accept()` from runner.go:**

Upstream feeds every prompt token into the penalty history via `Accept()`. With `repeatPenalty > 1.0`, this means the model is **penalized for generating any token that appeared in the user's prompt**. In an agentic CLI context:

- User prompt contains tool names (`get_weather`, `run_command`), function names, parameter names, code constructs
- The model MUST reproduce these exactly in tool calls
- Upstream's approach actively discourages this — the model is penalized for generating the exact tokens it needs

The fork's decision to record only generated tokens via `recordToken()` inside `Sample()` is not a "design choice" — it is a **correctness requirement** for agentic tool-calling models. The model should be penalized for repeating *its own output* but never for using terms from the prompt.

This also explains why upstream defaults `repeatPenalty` to 1.0 (making the whole system a no-op) — with prompt tokens in the history, any non-trivial penalty would degrade tool calling. The fork can safely default to 1.1 because only generated tokens are penalized.

### 3.2 Ring buffer vs Accept/Reset — not interchangeable architectures

The fork's self-contained ring buffer in `Sample()` is not just "a different approach." It enables the key property above (prompt token exclusion) without requiring the runner to distinguish between prompt tokens and generated tokens. The upstream's `Accept()` pattern fundamentally cannot achieve this because the runner calls `Accept()` on ALL inputs indiscriminately.

The fork's approach also eliminates an entire class of bugs: the runner never needs to "remember" to call Accept/Reset at the right times. There are 3 separate Accept/Reset call sites in upstream's runner.go (cache load, pending inputs, cache shift), each of which must be kept in sync. The fork has zero such call sites.

### 3.3 Default RepeatPenalty 1.1 vs 1.0 — enables the penalty system to actually work

This is not a policy preference. With upstream's 1.0 default:
- The penalty math is identity (no effect)
- The Accept/Reset bookkeeping is wasted work
- The `repeat_last_n` API parameter is doubly useless (already hardcoded to 64, AND the penalty is 1.0)

With the fork's 1.1 default:
- Repeated tokens get ~9% logit reduction after softmax
- The ring buffer actually serves a purpose
- The `repeat_last_n` parameter controls something real

### 3.4 Unconditional KV emission — prevents a real default-mismatch bug

The fork always writes `ssm.v_head_reordered` and `rope.mrope_interleaved` (even when `false`). This is not "extra metadata" — it prevents a concrete bug:

Third-party GGUFs don't contain these ollama-internal keys. When upstream omits them for legacy qwen3next (converted as `general.architecture=qwen35moe`), the model loader hits `defaultVHeadReordered("qwen35moe")` → `true`, which is **wrong** for legacy qwen3next. The fork's explicit `false` overrides the default, producing correct behavior.

### 3.5 Recurrent layer inference — fork matches llama.cpp exactly

The fork's 4-line interval computation matches llama.cpp's approach. The upstream's `inferRecurrentLayers()` is more elaborate but produces identical results for all real-world GGUFs. The fork's approach is strictly simpler and never consults `head_count_kv` for layer type detection, avoiding the scalar-broadcast edge case entirely (which the fork already fixed separately in commit bebddccd, then simplified away in 9ec17fc1).

### 3.6 BOS/EOS bug at vocabulary.go:57 — real bug in both codebases

```go
// Should be v.EOS, not v.BOS
if len(ids) > 0 && slices.Contains(v.BOS, ids[len(ids)-1]) {
    slog.Warn("adding eos token to prompt which already has it", "id", v.EOS)
}
```

Only affects warning/logging, not tokenization correctness. Both codebases have it.

---

## Part 4: Tokenizer Performance (All Validated Correct)

1. **Binary search in prompt.go**: O(log N) tokenize calls. Monotonicity assumption valid (dropping messages can only reduce token count).
2. **Special token splitting in special.go**: Equivalent behavior with `strings.Contains` early-out. 13 test cases.
3. **Stack buffer in vocabulary.go Merge()**: 128-byte stack buffer eliminates heap allocations in BPE merge hot path. Leverages Go 1.12+ map-lookup compiler optimization.

---

## Priority Action Items for the Fork

| Priority | Item | Effort | Section |
|----------|------|--------|---------|
| **P0** | Replace `SetInplace` with balanced concat tree | Small | 1.1 |
| **P1** | Use JSON tool definitions for qwen3.5 (keep XML for qwen3-coder) | Medium | 1.2 |
| **P1** | Swap system+tools ordering for qwen3.5 | Small | 1.3 |
| **P2** | Add `repeatPenalty <= 0` guard | Trivial | 1.4 |
| **P2** | Add nil checks + `Validate()` to qwen3next | Small | 1.5 |
| **P3** | Check Qwen 3.5 image support needs | Research | 1.6 |
| **P3** | Remove default system message injection with tools | Trivial | 1.7 |

## What Upstream Should Fix (for reference)

| Priority | Item |
|----------|------|
| **P0** | Route qwen3.5 through Qwen3CoderParser/Renderer (XML tool calls) |
| **P0** | Add thinking support to Qwen3CoderParser/Renderer |
| **P0** | Always close `</think>` in Qwen3VLRenderer |
| **P1** | Fix prefill detection to exclude tool-call messages |
| **P1** | Add lastQueryIndex to Qwen3CoderRenderer |
| **P1** | Fix mropeInterleaved default for qwen35/qwen35moe |
| **P2** | Actually honor `repeat_last_n` API parameter |
| **P2** | Fix tool call whitespace to match official template |
| **P3** | Consider making repeat_penalty default non-trivial (1.1) |
