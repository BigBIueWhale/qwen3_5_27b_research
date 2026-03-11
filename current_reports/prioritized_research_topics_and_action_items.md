# Prioritized Research Topics and Action Items for the BigBIueWhale/ollama Fork (Qwen 3.5 27B Focus)

**Date:** 2026-03-08
**Fork:** `BigBIueWhale/ollama` @ commit `4044b63f` (11 commits atop the Ollama `v0.17.4` merge base `cc90a035`)
**Model:** Qwen 3.5 27B (Alibaba Qwen), running via Unsloth Dynamic 2.0 `UD-Q4_K_XL` GGUF quantization (17.6 GB, 851 tensors, text-only, at `/tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf`)
**Inference engine:** Ollama (Go-based inference engine that vendors the GGML tensor math library from llama.cpp but reimplements everything else — chat template rendering, sampling, model runner — in Go)
**GGUF template note:** The Unsloth GGUF embeds a **modified** Jinja2 chat template — not byte-identical to the official `Qwen/Qwen3.5-27B` HuggingFace template. Unsloth changed the tool call arguments section: `tool_call.arguments is defined` → `tool_call.arguments is mapping`, and replaced `tool_call.arguments|items` tuple unpacking with key iteration + bracket access (`for args_name in tool_call.arguments` / `tool_call.arguments[args_name]`). This is a defensive workaround (likely for older llama.cpp Jinja engine versions where `|items` didn't return proper tuples). Functionally equivalent for normal use because llama.cpp's `func_args_not_string` workaround (at `common/chat.cpp:2947-2965`) ensures arguments are always parsed objects before template execution. The `enable_thinking` handling and all other template logic is identical to the official template. llama.cpp's converter (`convert_hf_to_gguf.py`) does NOT modify templates — this is a Unsloth-specific patch.

---

## Prioritized Action Items

| Priority | Item | Effort | Potential Impact |
|----------|------|--------|-----------------|
| ~~**High**~~ **DONE** | Grammar-constrained tool call generation — see Section A.3 and dedicated plan [`grammar_constrained_tool_calls_plan.md`](grammar_constrained_tool_calls_plan.md). **ALL 5 STEPS DONE** (commit `4044b63f`). Full pipeline: C bridge `grammar_init_lazy()` + `tool_call_grammar_from_json()`, Go wrappers `NewGrammarLazy()`/`ToolCallGrammarFromJSON()`, server wiring with mode-dependent trigger patterns, dead-end detection. 8 files, ~212 lines. | Medium (~265-415 est. → ~212 actual) | Structurally robust tool calling for agentic workflows — the model cannot hallucinate function names, produce type-mismatched parameters, or emit malformed XML |
| **CRITICAL** | JSON serialization mismatch fix (remaining sub-bug: key ordering) — see Section A.1. **Update 2026-03-11:** Sub-bug A is now DONE in the fork: the full tool-definition serialization path now emits literal `<>&` end-to-end, including nested tool-type `MarshalJSON` boundaries. **Update 2026-03-10:** Sub-bug C is also DONE: `formatToolCallArgument` for Qwen-family tool call arguments emits spaced JSON with literal `<>&`. Remaining open scope is `ToolFunctionParameters` key ordering. | **Medium** (remaining work is shared ordering research / renderer-local ordering strategy) | KV cache round-trip correctness (performance) + tool calling accuracy (quality) for multi-turn agentic workflows |
| **High** | CUDA asynchronous copy and reduced synchronization vendor update from llama.cpp — see Section B.3 | Medium | 5-15% throughput improvement on NVIDIA GPU hardware |
| ~~**High**~~ **DONE** | History `<think>` block rendering fix (removed incorrect `isThinking` gate in two places) — see Section A.2. Fixed in commit `4044b63f`. | Tiny (2 lines changed) | Correct multi-turn thinking/non-thinking conversations |
| **Medium** | `num_batch 2048` Modelfile configuration change — see Section B.4 | Zero (configuration-only, no code change) | Approximately 10% faster prompt evaluation speed |
| **Medium** | Multi-Resolution Rotary Position Embedding (M-RoPE) `can_shift()` guard — see Section C.5 | Tiny (3 lines of code) | Defensive correctness fix preventing corrupt positional encoding |
| **Medium** | Parallelism investigation (lifting the `numParallel = 1` scheduler restriction) — see Section D.9 | High | Enables serving concurrent requests instead of queuing them |
| **Low** | Qwen 3.5 model family size diversity verification — see Section C.6 | Small | Ensures all Qwen 3.5 model sizes (0.8B through 397B) work correctly |
| **Low** | Qwen3-VL-Thinking renderer missing `emitEmptyThinkOnNoThink` field — see Section E.12 | Research required | Correct `think: false` behavior for Qwen3 Vision-Language Thinking models |
| **Low** | Vulkan flash attention fix for AMD Radeon RDNA2 GPUs on Microsoft Windows — see Section G.14 | Small | Fixes broken flash attention for AMD GPU users on older Windows drivers |
| **Informational** | Speculative decoding confirmed impossible for hybrid recurrent architectures — see Section D.8 | Not applicable | Confirmed dead end — do not pursue |
| **Informational** | Multi-modal prompt caching (vision image re-encoding avoidance) — see Section D.10 | High | Future vision performance improvement (2-10x for repeated-image workflows) |
| **Informational** | Enabling vision capabilities with Unsloth third-party GGUF files — see Section F.13 | Very High | Third-party GGUF vision capability (currently only official Ollama unified GGUFs support vision) |

---

## Section A: Bugs In the Fork That Are Still Open

### A.1 — JSON Serialization Mismatches in Tool Definitions and Tool Call Arguments (3 Sub-Bugs)

The fork (and upstream Ollama) serialize tool definitions into the model's prompt differently from how the Qwen 3.5 model was originally trained by Alibaba. The official Qwen 3.5 Jinja2 template (at `/tmp/qwen35_template.jinja2`, fetched from `Qwen/Qwen3.5-27B` on HuggingFace) uses the `tojson` Jinja2 filter at line 50 for tool definitions and at line 122 for tool call arguments. In HuggingFace Transformers, this filter is overridden (at `chat_template_utils.py:463-466` in version 5.2.0) to use `json.dumps` with default separators. In llama.cpp's vendored Jinja engine, the equivalent is at `common/jinja/value.cpp:179-180`, which defaults to `", "` and `": "` separators and does not HTML-escape. Ollama reimplements this serialization in Go, producing different output in three ways.

> **CRITICAL IMPLEMENTATION NOTE — READ BEFORE ATTEMPTING ANY FIX:**
>
> These three sub-bugs range from "straightforward but requires care" (Sub-bug C) to "deceptively dangerous shared infrastructure change" (Sub-bug B). They should NOT be attempted simultaneously, and each requires prerequisite research described below. The core danger is that much of the serialization code is **shared infrastructure** used by many renderers (DeepSeek, GLM, Cogito, Olmo3, Nemotron, etc.), not just the Qwen family. Changing shared code without verifying every consumer can silently break other models — and the breakage may manifest not as obvious errors but as subtle quality degradation or catastrophic KV cache performance collapse (see "KV Cache Round-Trip Danger" below).

**Sub-bug A — HTML character escaping (high impact for coding and API tools) — DONE 2026-03-11:**
The `marshalWithSpaces` function at `model/renderers/json.go:9` calls Go's `json.Marshal` as its first step, then post-processes the output to add spaces after `:` and `,`. The spacing post-processing is correct. However, `json.Marshal` also HTML-escapes `<`, `>`, and `&` to `\u003c`, `\u003e`, `\u0026` — a Go design decision for safe HTML embedding that has no relevance here. The official template's `tojson` outputs these characters literally. A tool description like `"Returns temperature in <fahrenheit> & <celsius>"` becomes completely different tokens when HTML-escaped. The originally proposed fix was to replace the `json.Marshal` call in `marshalWithSpaces` with `json.NewEncoder` using `SetEscapeHTML(false)`. This pattern already exists in the codebase at `model/renderers/lfm2.go:55-56`.

*Shared infrastructure concern:* `marshalWithSpaces` is called by the Olmo3, Qwen 3.5, and Qwen3VL renderers. Changing its internals propagates to all of them. This verification has now been completed for the current callers: official HuggingFace templates downloaded into `/tmp/olmo3-check/` and the existing Qwen template validation show that `qwen3.5`, `qwen3-vl`, and `olmo3` all use `tools | tojson` for tool definitions, so disabling HTML escaping is correct for those paths. The key implementation correction discovered on 2026-03-11 was also applied: this was **not** just a `marshalWithSpaces` patch. The fork now fixes the **full tool-definition serialization path** by switching the top-level helper and the nested tool-type `MarshalJSON` boundaries in `api/types.go` / `internal/orderedmap` onto a no-HTML-escape JSON path. End result: top-level and nested tool-definition strings now emit literal `<`, `>`, and `&`, matching the reference convention. Full `go test ./...` passed after the change.

**Sub-bug B — Key ordering at the `ToolFunctionParameters` level (HIGH COMPLEXITY — shared struct change):**
Go's `json.Marshal` outputs struct fields in declaration order. In `api/types.go:486-491`, `Required` is declared before `Properties`, so serialized JSON always outputs `"required"` before `"properties"`. The official Qwen 3.5 template's `tojson` preserves Python dict insertion order, which places `"properties"` before `"required"` for Qwen 3.5 specifically.

*This is NOT a simple "swap two struct fields" fix.* `ToolFunctionParameters` is a shared type in `api/types.go` used by **every single renderer** in Ollama that supports tool calling — DeepSeek, GLM-4.7, GLM-OCR, Cogito, Olmo3, Nemotron3Nano, LFM2, and all Qwen variants. Swapping the field order changes the JSON output for ALL of them simultaneously. The critical subtlety: Python's `tojson` (which is `json.dumps` with `sort_keys=False`) preserves **dict insertion order**, which means the key ordering is an artifact of how each model's specific Jinja2 template constructs its Python dicts. Different model families have different templates that may construct dicts in different orders. The fact that Qwen 3.5's template produces `"properties"` before `"required"` says nothing about what DeepSeek's or GLM's or Cogito's templates produce. Changing the shared struct to match Qwen 3.5's ordering could make the serialization WORSE for other models.

The correct approach is one of:
1. **Renderer-local custom marshaling** — Have the Qwen renderers control field ordering when they serialize tool definitions, leaving the shared struct untouched. This is safer but more code.
2. **Universal verification** — Verify the HuggingFace template for every model family that uses `ToolFunctionParameters`, confirm they all produce the same ordering, and only then change the shared struct. This is more work upfront but cleaner if the ordering is in fact universal.
3. **Per-renderer struct wrapper** — Define a renderer-specific wrapper type with the desired field order and marshal that instead of the shared type.

Until the per-model verification is done, the shared struct MUST NOT be changed.

**Sub-bug C — Compact separators in `formatToolCallArgument` (HIGH impact, safest to fix) — DONE 2026-03-10:**
The fork now fixes this locally in `qwen3coder.go`: `formatToolCallArgument` no longer uses raw `json.Marshal` for map/slice/array values inside XML `<parameter>` tags. It now emits spaced JSON (`{"key": "value"}`) with HTML escaping disabled, matching the Qwen-family `tojson` convention for tool call arguments. This change is scoped to the Qwen renderers (`qwen3-coder`, `qwen3.5`, `qwen3-vl`) and avoids touching shared serializer infrastructure before the remaining cross-model verification is done.

#### Why These Bugs Matter — Two Distinct Failure Modes

**Failure mode 1 — Training distribution shift (affects all three sub-bugs):**
Every time the model uses tools, it sees a prompt format that differs from its training data. This creates a distribution shift — the model still works, but tool calling reliability is degraded. The model was trained on millions of examples with `{"key": "value"}` (spaced), `<` (literal), and `"properties"` before `"required"`. Feeding it `{"key":"value"}` (compact), `\u003c` (escaped), and `"required"` before `"properties"` means the model is operating slightly out-of-distribution on every single tool-using turn.

**Failure mode 2 — KV cache round-trip mismatch (affects Sub-bugs A and C specifically, catastrophic for performance):**
This is the more dangerous failure mode and is non-obvious. In a multi-turn conversation with tool calls, Ollama must re-render the entire conversation history on each turn. When re-rendering an assistant message that contained tool calls, the renderer must re-serialize the tool call arguments from the parsed Go types back into the prompt. If the re-serialized bytes don't exactly match what the model originally generated, the KV cache cannot be reused for that portion of the prompt — the engine must reprocess everything from the point of divergence.

Here is the round-trip path:
1. Model generates: `<parameter=city>\n{"name": "San Francisco", "state": "CA"}\n</parameter>` (spaced JSON, literal `<>&`)
2. Parser (`qwen3coder.go:parseValue`) parses `{"name": "San Francisco", "state": "CA"}` via `json.Unmarshal` into a `map[string]any` — original formatting is lost
3. On the next turn, the renderer calls `formatToolCallArgument(value)` on the stored map
4. `formatToolCallArgument` calls `json.Marshal`, producing `{"name":"San Francisco","state":"CA"}` — compact, no spaces
5. The re-rendered prompt now has **different bytes** at this position than what the model originally generated
6. KV cache lookup fails at the divergence point — everything after it must be reprocessed from scratch

#### Empirically verified round-trip mismatches (2026-03-08)

The full `parseValue()` → Go type → `formatToolCallArgument()` round-trip was tested for every parameter type. Results:

| Parameter type | Stored Go type | Round-trip match? | Notes |
|---|---|---|---|
| `string` | `string` | **YES** | Returned as-is, no `json.Marshal` |
| `integer` | `int` | **YES** | `fmt.Sprintf("%v")` produces identical text |
| `number` (float) | `float64` | **YES** | `fmt.Sprintf("%v")` produces identical text |
| `number` (int-like, e.g. `100`) | `int` | **YES** | `parseValue` special-cases `Trunc(f)==f` |
| `boolean` | `bool` | **YES** | `fmt.Sprintf("%v")` produces identical text |
| `null` | `nil` | **YES** | Hard-coded `"null"` string |
| No type declared | `string` (raw) | **YES** | No `json.Unmarshal`, preserved verbatim |
| **`object`** | `map[string]any` | **NO — 3 simultaneous corruptions** | See below |
| **`array`** | `[]any` | **NO — 2 simultaneous corruptions** | See below |

**The three corruptions for `object` parameters** (all verified with Go test code):

1. **Spacing removed**: Model generates `{"city": "Paris"}` (spaced, matching `tojson` training data). `json.Marshal` produces `{"city":"Paris"}` (compact). Every `:` and `,` loses its trailing space.

2. **Key order alphabetized**: Model generates `{"zebra": 1, "apple": 2}` (arbitrary order). `json.Unmarshal` into `map[string]any` loses insertion order. `json.Marshal` produces `{"apple":2,"zebra":1}` (alphabetical — Go spec guarantees this). The model was trained on Python `json.dumps(sort_keys=False)` which preserves insertion order.

3. **HTML escaping injected**: Model generates `{"query": "x < 5 & y > 3"}`. `json.Marshal` produces `{"query":"x \u003c 5 \u0026 y \u003e 3"}`. This is Go's HTML-safety escaping — completely wrong for model prompts. **In coding/agentic contexts, `<`, `>`, `&` appear constantly** (HTML, XML, comparison operators, generic types, logical operators, template syntax). This corruption hits a large fraction of real tool call arguments in coding workflows.

**For `array` parameters**: corruptions 1 (spacing) and 3 (HTML escaping) apply. Key ordering is not applicable.

**Combined effect**: A single object parameter like `{"file": "src/App.tsx", "content": "if (x < 5 && y > 3) { ... }"}` suffers ALL THREE corruptions simultaneously. The re-rendered prompt is unrecognizable compared to what the model generated. The KV cache misses at that point and reprocesses everything after it.

This is not just a minor performance regression. It is a **scaling catastrophe for agentic coding workflows**. Consider:
- An agentic coding assistant with a 20-turn conversation, each turn involving 2-3 tool calls with structured arguments
- Each tool call argument with any nested map/array triggers a KV cache miss at the point of the argument
- The miss forces reprocessing of everything from that point to the end of the conversation
- On turn 20, this could mean reprocessing 15,000+ tokens that were already in the cache
- The user experiences progressively slower response times with each turn, with no visible cause — the first few turns are fast, and by turn 15-20, each response takes dramatically longer
- This is invisible in testing because it only manifests at conversation scale, and the model still produces correct output — it's purely a latency problem

This fix has now been implemented in the fork's Qwen-local serializer path. The renderer re-serializes structured tool call arguments as spaced JSON with literal `<>&`, matching the model's output format more closely and improving KV cache round-tripping for Qwen-family tool calls. The remaining mismatch at this layer is object key ordering, which still requires a separate solution.

llama.cpp avoids all three sub-bugs because it executes the Jinja2 template directly — its `tojson` implementation at `common/jinja/value.cpp:179-180` defaults to `", "` and `": "` separators, preserves dict insertion order (at line 1330: `as_ordered_object()` with comment "IMPORTANT: need to keep exact order"), and writes `<>&` literally without HTML escaping (lines 1291-1312). These JSON serialization bugs exist identically in upstream Ollama.

However, llama.cpp has its **own** set of bugs unrelated to JSON serialization — see `fork_vs_latest_upstream_and_llama_cpp.md` Section 6.4 for the full list (non-thinking prefill token sequence, penalty sampler prompt contamination, no model-specific sampler defaults, `tojson` float precision). Also note that the Unsloth GGUF embeds a modified template (see header note) — the modifications are in the tool call arguments section only and are functionally equivalent, but the template is not byte-identical to the official HuggingFace version.

Beyond JSON serialization, there is a deeper **structural** difference between Ollama and llama.cpp in how tool calls are validated during generation. Ollama uses a free-form streaming parser (`parsers/qwen3coder.go`) that accepts whatever the model generates and tries to extract tool calls after the fact — it trusts the model. llama.cpp does not trust the model: it builds a GBNF grammar from the declared tool schemas (`chat.cpp:1555-1616`) and masks model logits during generation so only syntactically valid tool calls with declared function names and schema-conforming parameter values can be produced. Malformed tool calls are structurally impossible. The fork will implement the same grammar-constrained approach for Qwen 3.5 to achieve structurally robust tool calling for agentic workflows. The vendored llama.cpp C++ grammar engine already supports this (including lazy grammar activation with trigger patterns); the gap is only in the Go↔C bridge layer (~30 lines) and a Qwen-specific GBNF grammar builder (~200-400 lines). See Section A.3 for the full implementation plan and `fork_vs_latest_upstream_and_llama_cpp.md` Section 6.6 for the detailed analysis of the problem.

**llama.cpp's `tojson` has one minor flaw:** float serialization at `value.cpp:1290` uses C++ `ostringstream` default formatting (6 significant digits), while Python's `json.dumps` preserves full IEEE 754 double precision (~17 digits). Additionally, `1.0` becomes `1` (no decimal point) in C++ but stays `1.0` in Python. This is very unlikely to affect model behavior in practice (only matters for tool call arguments with high-precision float values), but is a technical divergence from HuggingFace's `tojson` behavior.

#### Prerequisite Research Before Implementation

Each sub-bug requires specific research that must be completed before writing code:

**Research topic 1 — Shared infrastructure dependency map:**
Build a complete call graph of which renderers call which shared functions (`marshalWithSpaces`, `jsonMarshalNoEscape` if created, `formatToolCallArgument`, `renderAdditionalKeys`, `formatToolDefinitionType`). For each call site, document whether the caller is model-family-specific or shared across families. Map which renderers serialize `ToolFunctionParameters` and how (direct `json.Marshal`, `json.MarshalIndent`, `marshalWithSpaces`, or custom). This determines the blast radius of each change.

**Research topic 2 — Per-model template verification:**
For every model family whose renderer touches shared serialization code (at minimum: DeepSeek, GLM-4.7, GLM-OCR, Cogito, Olmo3, Nemotron3Nano, LFM2, and all Qwen variants), download and inspect the official HuggingFace Jinja2 template. For each, determine: (a) What JSON serialization function the template uses for tool definitions (is it `tojson`? `tojson(indent=4)`? Custom?). (b) What key ordering the template produces for the `parameters` object (is it `properties` before `required`? The reverse? Something else?). (c) Whether the template expects literal `<>&` or HTML-escaped versions. (d) Whether tool call arguments use spaced or compact JSON. Only after this verification is complete can we know which changes are safe to make globally vs. which must be renderer-local.

**Research topic 3 — KV cache round-trip proof per renderer:**
For each renderer that handles tool calls (Qwen3Coder, Qwen 3.5, Qwen3VL, DeepSeek, GLM, Cogito, Olmo3, Nemotron), trace the complete round-trip: model output format → parser extraction → Go type storage → renderer re-serialization → prompt bytes. Prove that after the proposed fix, the re-serialized bytes exactly match what the model was trained to generate. A mismatch in ANY leg of this chain creates a KV cache miss. Document edge cases: nested objects, arrays, strings containing `<>&"`, unicode, empty objects, null values.

**Research topic 4 — Scoping and sequencing strategy:**
Decide upfront: fix only Qwen 3.5 (safest, narrowest blast radius), fix all Qwen family models (medium, since they share `tojson` conventions), or fix globally (maximum impact but requires completing Research topic 2 for all models). The sequencing guidance was mostly correct: Sub-bug C first (Qwen-only code, improves KV cache round-trip) is DONE in the fork; Sub-bug A second is now also DONE via the full shared tool-definition serialization-path fix; Sub-bug B last (shared struct, requires universal verification or renderer-local approach) remains open.

**Research topic 5 — Test infrastructure strategy:**
The existing renderer tests encode exact byte strings as golden expectations. Changing shared types or functions breaks tests across many renderer test files simultaneously, even if the runtime behavior is correct. Before implementing, decide: update all affected test expectations (requires understanding whether the new output is correct for each model), or restructure tests to be more resilient to serialization changes (e.g., parse the JSON output and compare structurally rather than byte-for-byte).

### A.2 — History `<think>` Blocks Incorrectly Gated on the `isThinking` Boolean — **FIXED** (commit `4044b63f`)

**Status: DONE.** The fork's `Qwen35Renderer` now unconditionally wraps recent assistant messages in `<think>...</think>`, matching the official Jinja2 template. Upstream Ollama still has this bug.

**The fix** (two lines in `model/renderers/qwen35.go`):
1. Line 65: `splitQwen35ReasoningContent(content, messageThinking string)` — removed `isThinking` parameter; now always uses `messageThinking` when available
2. Line 143: `if i > lastQueryIndex` — removed `isThinking &&` prefix; assistant messages after `lastQueryIndex` always get `<think>` blocks

Both the fork (before this fix) and upstream Ollama gate the rendering of `<think>` blocks in assistant message history on `isThinking` (controlled by the API's `think: true/false` parameter). This is a two-part bug — the gate appears in both the rendering condition and the reasoning extraction function.

Notably, llama.cpp has an **analogous but different** generation-prompt bug: its `common_chat_params_init_qwen3_coder` handler (at `common/chat.cpp:1527`) calls `apply(tmpl, inputs)` without passing `enable_thinking` in the template context. The `apply()` function at `chat.cpp:804-854` builds the Jinja context from messages, tools, `extra_context`, and `add_generation_prompt` — but never includes `enable_thinking`. Only the Hermes 2 Pro handler (at `chat.cpp:2392-2397`) passes `enable_thinking` via `extra_context`; the Qwen 3.5 handler does not. As a result, `enable_thinking` is always `undefined` in the Jinja context, and the template's `enable_thinking is defined and enable_thinking is false` check at line 148 never triggers — the template always produces the thinking-mode ending `<think>\n`. The handler then attempts to compensate with post-processing at `chat.cpp:1533-1539`: when `!inputs.enable_thinking`, it appends `</think>` to the prompt, producing `<think>\n</think>` (3 tokens). But the official non-thinking prefill is `<think>\n\n</think>\n\n` (6 tokens) — with an empty line inside the think block and a double-newline after it. The 3-token sequence `<think>\n</think>` was never seen during training. The model encounters a novel prefix it must interpret on the fly, which is a real distribution shift that can degrade output quality for all non-thinking mode usage. This is a llama.cpp-specific bug — it does NOT affect history rendering (which is always correct in llama.cpp because the template handles it), only the generation prompt for the current turn. The fork and upstream Ollama handle the generation prompt correctly via `emitEmptyThinkOnNoThink` (Section 9.4 of `fork_vs_latest_upstream_and_llama_cpp.md`), but have the history rendering bug described below.

**Part 1 — Rendering condition at `qwen35.go:143`:**
```go
if isThinking && i > lastQueryIndex {
    // render with <think> block
} else {
    // render without <think> block — reasoning content silently discarded
}
```

The official Qwen 3.5 Jinja2 template at line 100 (`/tmp/qwen35_template.jinja2`) uses only:
```jinja2
{%- if loop.index0 > ns.last_query_index %}
```
No `enable_thinking` check. Assistant messages after `last_query_index` ALWAYS include `<think>` blocks regardless of whether thinking is enabled for the current turn.

**Part 2 — Reasoning extraction at `qwen35.go:65`:**
```go
func splitQwen35ReasoningContent(content, messageThinking string, isThinking bool) (reasoning string, remaining string) {
    if isThinking && messageThinking != "" {
        return strings.TrimSpace(messageThinking), content
    }
    // fallback: parse <think> tags from content
}
```

The official template at lines 91-92 always uses `reasoning_content` when it exists as a string — no `enable_thinking` check:
```jinja2
{%- if message.reasoning_content is string %}
    {%- set reasoning_content = message.reasoning_content %}
```

When `isThinking=false` and `messageThinking` is non-empty (the normal case when a previous turn produced reasoning), the fork ignores the stored thinking field and falls through to content parsing, which typically finds nothing (since the thinking was stored separately, not embedded in content). The reasoning is silently lost.

**When this bug triggers:** Multi-turn conversations where thinking mode switches between turns — for example, an agentic framework that uses `think: true` for the initial reasoning step and `think: false` for a follow-up. Previous assistant responses had thinking content stored in `message.Thinking`. With `think: false`, both the extraction and the rendering gate fail, and the model sees its own previous responses with reasoning stripped out — a prompt format it was never trained on.

**The fix is two-part:**
1. Remove `isThinking &&` from the rendering condition at `qwen35.go:143`
2. Remove `isThinking &&` from `splitQwen35ReasoningContent` at `qwen35.go:65`, so it always uses `messageThinking` when available

**The parser (`qwen35.go` in `model/parsers/`) is unaffected by this bug.** The parser correctly handles both thinking-enabled and thinking-disabled modes: when thinking is enabled, it collects content until `</think>` then delegates tool call parsing to `Qwen3CoderParser`; when thinking is disabled, it starts directly in content collection mode. The parser also correctly ignores tool call XML that appears inside thinking blocks, handles streaming with partial tag boundaries, and strips optional leading `<think>` tags that some checkpoints re-emit. All parser behavior was verified against the official template's expected output format and is correct.

### A.3 — Grammar-Constrained Tool Call Generation (Matching What llama.cpp Does Correctly) — **FULLY DONE** (commit `4044b63f`)

**Full plan:** See dedicated report [`grammar_constrained_tool_calls_plan.md`](grammar_constrained_tool_calls_plan.md).

**Status: ALL 5 STEPS DONE.** The full pipeline is implemented and committed at `4044b63f` (8 files, ~212 lines added):

| Step | What | Files |
|------|------|-------|
| 1 | C bridge: `grammar_init_lazy()` for dormant grammars with regex triggers | `llama/sampling_ext.cpp`, `.h` |
| 2 | Go wrappers: `NewGrammarLazy()`, `ToolCallGrammarFromJSON()` | `llama/llama.go` |
| 3 | GBNF grammar builder: `tool_call_grammar_from_json()` (done earlier, ~230 lines C++) | `llama/sampling_ext.cpp`, `.h` |
| 4 | Server wiring: `ChatHandler` builds grammar for `qwen3.5` with mode-dependent triggers | `server/routes.go`, `llm/server.go`, `runner/ollamarunner/runner.go` |
| 5 | Dead-end detection: clear error when grammar rejects all tokens | `sample/samplers.go` |

**Key design decisions:**
- **Mode-dependent trigger patterns**: Thinking mode requires `</think>` before `<tool_call>` in the regex (`[\s\S]*</think>[\s\S]*?(<tool_call>[\s\S]*)`) to prevent grammar activation on hallucinated tool calls inside thinking text. Non-thinking mode triggers on first `<tool_call>` (`[\s\S]*?(<tool_call>[\s\S]*)`).
- **Scope limited to `qwen3.5` parser**: Gated on `m.Config.Parser == "qwen3.5"` — no other models affected.
- **Format→Grammar guard**: When both tools and Format are present, tool grammar takes precedence via `if req.Grammar == "" && len(req.Format) > 0`.
- **Code consolidation**: Public functions (`NewGrammar`/`NewGrammarLazy`, `NewGrammarSampler`/`NewGrammarSamplerLazy`) delegate to private shared implementations to eliminate duplication.

**Bonus fix**: Renderer `isThinking` gate bug (Section A.2) was discovered and fixed during this implementation — the grammar trigger patterns depend on correct `</think>` placement, which led to verifying the renderer against the official template.

Ollama's tool call parser trusts the model to produce well-formed output. It should not. The model's output is untrusted — it can hallucinate function names, produce type-mismatched parameter values, emit malformed XML, or leave `<tool_call>` tags unclosed. llama.cpp handles this correctly via grammar-constrained generation (`chat.cpp:1555-1616` at reference commit `a0ed91a`). The fork is now the first Ollama model to implement grammar-constrained tool calls.

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
