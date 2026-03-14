# Qwen 3.5 Renderer and Parser Test Gap Analysis

**Date:** 2026-03-13
**Fork:** `BigBIueWhale/ollama` at `/home/user/shared_vm/ollama/`, commit `c90b3cbe`, 11 commits atop the Ollama `v0.17.4` merge base `cc90a035`
**Model:** Qwen 3.5 27B (Alibaba Qwen), a hybrid recurrent architecture (`qwen3next`). Currently the most promising open-source large language model that can run on consumer-grade hardware (17.6 GB at Q4_K_XL quantization).
**Scope:** This document covers only the test gaps that need to be closed for the Qwen 3.5 prompt template renderer (`model/renderers/qwen35.go`) and parser (`model/parsers/qwen35.go`, which delegates tool call parsing to `model/parsers/qwen3coder.go`), and the grammar-constrained tool call generation pipeline. It does not cover model architecture, GGUF conversion, penalty sampling, or performance — those are covered in `consolidated_report.md`.

---

## Why Exact Template Fidelity Matters

Qwen 3.5 27B was trained on billions of tokens where every chat turn follows the exact byte sequence produced by the official Jinja2 template from `Qwen/Qwen3.5-27B` on HuggingFace. Ollama does not execute Jinja2 templates — it reimplements each model's template as a hand-written Go renderer. Any byte-level deviation between the Go renderer's output and what the official Jinja2 template produces means the model receives a prompt format it was never trained on. This causes two distinct failure modes:

**Failure mode 1 — training distribution shift.** The model's learned attention patterns were optimized for the exact token sequence the official template produces. A missing space, a swapped JSON key, or an HTML-escaped angle bracket changes the token IDs the model sees, pushing the input out of the training distribution. The model still runs and produces output, but its tool-calling accuracy, reasoning quality, and instruction-following reliability all degrade. The degradation is silent — the model doesn't error, it just gets worse. Users blame the model ("Qwen 3.5 is unreliable") rather than the inference engine.

**Failure mode 2 — KV cache invalidation in multi-turn agentic conversations.** Ollama re-renders the entire conversation history on every turn. In a 20-turn agentic coding session with tool calls, the engine checks whether the new prompt shares a byte-identical prefix with the previous prompt. If it does, the KV cache (the expensive intermediate computation from processing all prior tokens) is reused. If any byte differs — a space removed, a key reordered, an `&` escaped to `\u0026` — the cache is invalidated from that point forward, and everything after must be recomputed. By turn 15-20, each response takes dramatically longer. The model still produces correct output, so the problem is invisible in functional testing.

Both failure modes are especially damaging for this specific model because Qwen 3.5 27B is positioned as an agentic coding assistant. Its primary use case involves long multi-turn conversations with many tool calls — exactly the scenario where template infidelity compounds.

**How the official template handles thinking modes:** The official Qwen 3.5 Jinja2 template (from `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.5-35B-A3B` on HuggingFace — both have identical templates) receives `enable_thinking` as a per-request template parameter, not a per-message field. It is checked in exactly one place: the `add_generation_prompt` block at the end of the template. It has zero effect on how historical assistant messages are rendered. Historical assistant messages receive `<think>` block wrapping based solely on their position relative to `last_query_index` — if a message appears after the last real user query, it gets `<think>\nreasoning\n</think>\n\n` wrapping unconditionally, even if `enable_thinking` is `false`, and even if the reasoning content is empty (producing `<think>\n\n</think>\n\n`). The `enable_thinking` parameter only controls whether the generation prompt at the very end opens an active thinking block (`<think>\n` when true or undefined) or a pre-closed empty one (`<think>\n\n</think>\n\n` when explicitly false). Reasoning content itself is extracted from each message via two paths: an explicit `reasoning_content` field on the message (preferred), or by parsing inline `<think>...</think>` tags from the message content (fallback). The `<|im_end|>` tag that closes each assistant message is emitted unconditionally — it appears outside any conditional block in the template and runs for every assistant message regardless of thinking mode, tool calls, or message position. These rules are the ground truth that every renderer test must hold the Go implementation accountable to.

---

## What Tests Exist Today

### Renderer tests: `model/renderers/qwen35_test.go` (601 lines, 8 test functions)

**`TestQwen35RendererUsesXMLToolCallingFormat`** (lines 10-75): Verifies that tool calls are rendered in XML format (`<tool_call><function=get_weather>...`), that tool definitions appear as JSON in the system prompt, and that the tools section appears before the system message content. Uses `strings.Contains` checks — not byte-exact. Does not verify JSON spacing or field ordering within tool definitions. Does not verify the `<|im_end|>` tag placement on the assistant message (the message with tool calls is not the last message, so the prefill bug is not exercised).

**`TestQwen35RendererNoThinkPrefill`** (lines 77-91): Verifies that when thinking is disabled via `think.Value = false`, the renderer emits the official empty thinking block prefill `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Uses `strings.HasSuffix` — checks the exact trailing bytes. The renderer is constructed with `isThinking: true`, and the `ThinkValue{Value: false}` override is passed to `Render()`. This tests the `emitEmptyThinkOnNoThink` field but does NOT test the prefill bug fix because the last message is a user message, not an assistant message.

**`TestQwen35RendererBackToBackToolCallsAndResponses`** (lines 93-176): Verifies two tool calls (`add` and `multiply`) in one assistant message, grouped tool responses, and that historical thinking content ("Need to call add and multiply.") is NOT rendered because the assistant message is at index 2 and `lastQueryIndex` is 1 (the user message at index 1 is the last non-tool-response user message). The last message is `{Role: "user", Content: "Summarize the results."}`, so the prefill bug is not exercised. All renderer tests construct the renderer with `isThinking: true`.

**`TestQwen35RendererStructuredToolArgumentsUseSpacedJSON`** (lines 178-212): Verifies that when a tool call argument is a Go `map[string]any` containing `{"content": "if (x < 5 && y > 3) {}"}`, the rendered output uses spaced JSON separators (`: ` after colons, `, ` after commas) and preserves literal HTML characters (`<`, `>`, `&`) without escaping them to `\u003c`, `\u003e`, `\u0026`. Uses `strings.Contains` with the exact expected substring `<parameter=payload>\n{"content": "if (x < 5 && y > 3) {}"}\n</parameter>`.

**`TestQwen35RendererToolDefinitionsUseLiteralHTMLChars`** (lines 214-263): Verifies that tool definitions in the system prompt preserve literal HTML characters in tool descriptions (`"Returns temperature in <fahrenheit> & <celsius>"`), parameter descriptions (`"City name with <tag> & symbol"`), and nested `items` descriptions (`"Use < and > literally & keep order"`). Checks both negative (no `\u003c`, `\u003e`, `\u0026` present) and positive (exact JSON substring match). The expected JSON substring at line 259 hardcodes the field ordering as `"required": ["location"], "properties": {...}` — this is the Go struct declaration order, which differs from the HuggingFace/OpenAI convention of `"properties": {...}, "required": ["location"]`. This means the test locks in the wrong field ordering as the expected behavior.

**`TestQwen35RendererInterleavedThinkingAndTools`** (lines 265-449, 2 subtests): Verifies two consecutive assistant turns, each with thinking content, visible content, and a tool call, separated by a tool response. The `think=true` subtest checks exact multi-line substrings including the `<think>` block, the `</think>` close, the blank line separator, the visible content, and the `<tool_call>` XML, plus the think=true generation prompt suffix. The `think=false` subtest is the primary regression detector for the fork's unconditional thinking block fix (Gap 2): it renders the same messages with `ThinkValue{Value: false}` and verifies that both historical assistant messages retain their `<think>` blocks with full reasoning text, and that the generation prompt uses the non-thinking form (`<think>\n\n</think>\n\n`). This catches both regression vectors: re-adding `isThinking` to `splitQwen35ReasoningContent` or re-adding `isThinking &&` to line 143.

**`TestQwen35RendererAssistantPrefillWithThinking`** (lines 351-378): The only byte-exact test in the suite — uses `got != want` to compare the entire rendered output. Verifies that when the last message is an assistant message with `Thinking: "Keep it short."` and `Content: "Hello world"` but NO tool calls, the output is a prefill: no `<|im_end|>` after the assistant content, and no generation prompt appended. This is the correct prefill behavior for an assistant message without tool calls (the `len(message.ToolCalls) == 0` condition is true, so `prefill` is true).

**Think mode coverage across renderer tests:** All renderer tests construct the `Qwen35Renderer` with `isThinking: true`. Three tests exercise `think: false` at runtime: `TestQwen35RendererNoThinkPrefill` (generation prompt only, no assistant messages), `TestQwen35RendererAssistantToolCallIsNotPrefill/think=false` (generation prompt after tool-calling assistant, no historical thinking blocks), and `TestQwen35RendererInterleavedThinkingAndTools/think=false` (historical assistant messages with thinking blocks + generation prompt — the Gap 2 regression detector). The last of these is the only test that exercises historical assistant message rendering under think=false, verifying that `<think>` blocks are preserved unconditionally. As documented in "Why Exact Template Fidelity Matters," the official template checks `enable_thinking` in exactly one place — the generation prompt. Historical `<think>` block rendering at line 143 is unconditional — it does not check `isThinking` (this is the fork's fix, now protected by Gap 2's test). Content formatting, tool call XML, tool definition JSON, and `<|im_end|>` placement are all independent of think mode. `TestQwen35RendererBackToBackToolCallsAndResponses` still verifies only the think=true suffix — adding a think=false variant would cover the simpler case where historical assistant messages are at or before `lastQueryIndex` (so no `<think>` wrapping regardless) but the generation prompt suffix still needs to be correct. Tests that verify content formatting via `strings.Contains` (`TestQwen35RendererStructuredToolArgumentsUseSpacedJSON`, `TestQwen35RendererToolDefinitionsUseLiteralHTMLChars`) do not need think=false variants because the substrings they check are identical regardless of think mode. `TestQwen35RendererAssistantPrefillWithThinking` also does not need a think=false variant because prefill behavior is independent of think mode.

### Parser tests: `model/parsers/qwen35_test.go` (383 lines, 11 test functions)

**`TestQwen35ParserXMLToolCall`** (lines 9-61): Verifies basic XML tool call parsing with two parameters (`location` as string, `days` as integer). Checks exact argument values and types via the `.Get()` method on the parsed `ToolCallFunctionArguments`. Does not involve thinking blocks.

**`TestQwen35ParserThinkingWithExplicitOpeningTag`** (lines 63-84): Verifies that `<think>Let me think...</think>Answer.` produces `thinking = "Let me think..."` and `content = "Answer."`.

**`TestQwen35ParserAssistantPrefillStartsInContent`** (lines 86-109): Verifies that when the parser is initialized with `think: false` (the `CollectingContent` initial state), incoming text goes directly to content without thinking extraction. Simulates an assistant prefill continuation.

**`TestQwen35ParserToolCallEmittedInThinkingIsNotParsed`** (lines 111-153): Verifies that `<tool_call>` XML appearing inside a `<think>` block is treated as literal thinking text, not parsed as a tool call. The thinking output contains the raw XML tags. Zero tool calls are returned. This is a critical edge case because the model sometimes "rehearses" tool calls in its thinking.

**`TestQwen35ParserToolCallAfterThinkingCloseIsParsed`** (lines 155-202): Verifies that `<tool_call>` XML appearing after `</think>` is correctly parsed as a tool call. Thinking content is extracted separately.

**`TestQwen35ParserThinkingDisabledPassesContentThrough`** (lines 204-225): Verifies that with `think: false`, the entire input passes through as content without any thinking extraction.

**`TestQwen35ParserThinkingDisabledWithCloseTagTreatsAsContent`** (lines 227-248): Verifies that with `think: false`, a `</think>` tag in the input is treated as literal content, not a thinking boundary marker.

**`TestQwen35ParserLeadingThinkCloseProducesContent`** (lines 250-271): Verifies that when thinking is enabled but the input starts with `</think>`, the text after the close tag becomes content (no thinking extracted).

**`TestQwen35ParserStreamingSplitThinkCloseTag`** (lines 273-308): Verifies correct behavior when the `</think>` tag is split across two streaming chunks (`"Reasoning text</thi"` then `"nk>The final answer."`). Thinking is extracted from the first chunk, content from the second.

**`TestQwen35ParserStreamingEatsWhitespaceAfterThinkClose`** (lines 310-359): Verifies that whitespace (newlines, spaces, tabs) immediately after `</think>` is consumed and does not appear in the content output. Uses three streaming chunks.

**`TestQwen35ParserThinkingTruncatedWithoutCloseTag`** (lines 361-382): Verifies that when the input ends without a `</think>` close tag, all accumulated text is treated as thinking content.

### Shared parser tests: `model/parsers/qwen3coder_test.go` (1380 lines)

The `Qwen35Parser` delegates tool call parsing to an embedded `Qwen3CoderParser`. The `qwen3coder_test.go` file contains 105 `parseValue()` type coercion test cases (string, integer, float, boolean, null, array, object, union types), 18 streaming scenarios (character-by-character splitting, partial tag detection, Unicode including Arabic, Chinese, and emoji), 8 `TestQwenToolParser` cases (names with spaces, names with quotes, ampersands and angle brackets in values, Unicode in function names), tool call indexing tests (sequential, streaming, reset on `Init()`), and thinking-specific parser tests (thinking with tool calls, tool call interrupting thinking, split-chunk thinking, leading `<think>` tag stripping, thinking disabled).

### KV cache round-trip tests: `server/qwen35_kvcache_roundtrip_test.go` (809 lines, 22 test cases)

These tests simulate a full agentic loop: render a prompt with tool definitions → append a model-generated suffix (simulating what the model would output) → parse the suffix to extract tool calls → reconstruct the conversation with parsed tool calls + tool results → render the next prompt → compare byte-level prefix overlap between the two rendered prompts to determine KV cache reuse.

The 22 test cases cover: scalar string arguments with `think=true` and `think=false`, prose before tool calls, structured JSON object arguments, JSON client round-trips (marshal to JSON and unmarshal before sending back), `<tool_response>`-wrapped user-style tool results, compact JSON spacing canonicalization (`{"a":1}` → `{"a": 1}`), extra JSON spacing canonicalization (`{  "a" : 1 }` → `{"a": 1}`), non-canonical key ordering preservation (`{"z": 1, "a": 2}` stays in z-then-a order), nested non-canonical key ordering, number lexical form preservation (`1.0` and `1e3`), literal HTML character preservation (`<`, `>`, `&`), fresh-user continuation with boundary shifts, and four cancellation scenarios (incomplete tool call, incomplete thinking, complete thinking but incomplete tool call, prose before incomplete tool call).

### Grammar tests

**`sample/samplers_test.go:TestGrammar`** (lines 93-150): Tests `NewGrammarSampler()` with a generic JSON GBNF grammar. Verifies that after `grammar.Apply(tokens)`, some tokens are masked to `-inf` and some remain. Does not test lazy grammar activation, trigger patterns, or tool-call-specific grammars.

**`llama/llama_test.go`**: Contains `TestSchemaToGrammar()` and `TestIssue7978()` which test the generic JSON Schema to GBNF converter. Does not test `ToolCallGrammarFromJSON()` or `NewGrammarLazy()`.

**C++ embedded validation**: The `tool_call_grammar_from_json()` function in `llama/sampling_ext.cpp` has 12 defensive validation checks (null pointers, zero-length buffers, invalid JSON, non-array JSON, empty arrays, missing fields, empty names, null bytes, forbidden characters, duplicate names, required-not-in-properties, output truncation). These are validation guards, not standalone tests — they are only exercised if the Go caller passes invalid input.

---

## Gap 1: ~~No Test for the Prefill Bug Fix~~ DONE — `TestQwen35RendererAssistantToolCallIsNotPrefill` (the Fork's Most Important Correctness Fix)

**What the fix does:** At `model/renderers/qwen35.go` line 136, the fork adds `&& len(message.ToolCalls) == 0` to the prefill condition:
```go
prefill := lastMessage && message.Role == "assistant" && len(message.ToolCalls) == 0
```
Without this guard, when the last message in the conversation is an assistant message that contains tool calls, the renderer treats it as a prefill: it omits the `<|im_end|>` closing tag (line 169: `if !prefill`) and skips the generation prompt (line 183: `if lastMessage && !prefill`). The model's view of the conversation is corrupted — it sees an unclosed assistant turn ending with `</tool_call>` and no `<|im_start|>assistant\n<think>\n` to begin generating.

**Why the official template doesn't have this problem:** The official Jinja2 template emits `<|im_end|>\n` for every assistant message unconditionally — the line `{{- '<|im_end|>\n' }}` appears outside any conditional block and runs for every assistant message regardless of whether it has tool calls, is the last message, or anything else. The "prefill" concept does not exist in the official template; it has a separate `add_generation_prompt` flag that controls whether the generation prompt is appended after all messages. Ollama's Go renderer merges these two concerns into a single `prefill` heuristic: "if the last message is an assistant message, the client must be providing partial text for the model to continue, so suppress `<|im_end|>` and the generation prompt." This heuristic is correct for assistant messages without tool calls (the client is providing a partial response to continue from), but incorrect for assistant messages with tool calls (those are complete turns that must be closed). The `len(message.ToolCalls) == 0` guard makes the Go renderer match the official template's behavior for this case.

**When does this trigger in real use:** An agentic framework sends back the full conversation history including the assistant's tool-call message as the last message, with the tool result arriving in a separate subsequent request. Or a client replays a saved conversation that ended at a tool-call boundary.

**Why no existing test catches a regression:** Every test that includes an assistant message with tool calls follows it with at least one more message (a tool response or a user message). The assistant+toolcalls message is never the final message in any test's message array. If someone removed the `&& len(message.ToolCalls) == 0` guard, all 7 renderer tests and all 22 KV cache round-trip tests would still pass.

**What the test needs to verify:** Create a message array where the last message is `{Role: "assistant", ToolCalls: [...]}`. Verify that:
1. The `<|im_end|>` tag IS present after the tool call XML (the message is properly closed, not treated as a prefill).
2. The generation prompt (`<|im_start|>assistant\n<think>\n`) IS present at the end of the rendered output (the model is prompted to generate).
3. The assistant's visible content and tool call XML are fully rendered (not truncated).

**Status: DONE.** Implemented as `TestQwen35RendererAssistantToolCallIsNotPrefill` in `model/renderers/qwen35_test.go` with two subtests (`think=true` and `think=false`). Each subtest has three layers of assertions: (1) targeted `<|im_end|>` presence check after tool call XML, (2) targeted generation prompt suffix check, and (3) byte-exact full output comparison. All three assertion layers include descriptive failure messages explaining the official template's behavior, the real-world trigger scenario (agentic frameworks sending tool-calling history), and the consequence of the regression (undefined model behavior from training-absent prompt shapes). Verified that removing the `&& len(message.ToolCalls) == 0` guard causes both subtests to fail with all three error messages firing.

**Think mode variants:** This test must exist for both `think=true` and `think=false` modes. The `<|im_end|>` emission after the tool call is independent of think mode — the official Qwen 3.5 Jinja2 template closes every assistant message unconditionally — so both modes must verify its presence. The generation prompt, however, differs: think=true appends `<|im_start|>assistant\n<think>\n`, while think=false with `emitEmptyThinkOnNoThink` appends `<|im_start|>assistant\n<think>\n\n</think>\n\n`. Testing only one mode would leave the other's generation prompt unverified, and a regression could silently produce the wrong prompt shape for one thinking configuration while the other continues to work. This is particularly dangerous because users and agentic frameworks frequently switch thinking modes between requests within the same conversation — a user might enable thinking for complex tool-planning turns and disable it for simple follow-ups.

---

## Gap 2: ~~No Test for Unconditional Thinking Block Rendering When Thinking Mode Is Disabled~~ DONE — `TestQwen35RendererInterleavedThinkingAndTools/think=false`

**What the fix does:** The fork removed `isThinking` gating from two locations in `model/renderers/qwen35.go` — both are part of the same conceptual fix, and both must be protected by the same regression test.

**Part 1 — Reasoning extraction at line 65:** The fork changed the signature of `splitQwen35ReasoningContent` from `(content, messageThinking string, isThinking bool)` to `(content, messageThinking string)`, removing the `isThinking` parameter entirely. The upstream version (commit `9896e36`) gates reasoning extraction on `if isThinking && messageThinking != ""` — when `isThinking` is false, the function ignores the stored `messageThinking` field and falls through to content parsing, which typically finds nothing. The reasoning is silently lost before it even reaches the rendering code. The fork's version uses `if messageThinking != ""` — no `isThinking` check — matching the official template's `message.reasoning_content is string` guard, which never checks `enable_thinking`. (See Gap 3 for unit-level testing of this function's code paths; the connection is that Gap 3 tests the function in isolation, while this gap tests the renderer-level consequence of the signature change.)

**Part 2 — Rendering condition at line 143:** The fork uses `if i > lastQueryIndex` without any `isThinking` gate:
```go
if i > lastQueryIndex {
    sb.WriteString(imStartTag + message.Role + "\n<think>\n" + contentReasoning + "\n</think>\n\n" + content)
}
```

As documented in "Why Exact Template Fidelity Matters" above, the official Qwen 3.5 Jinja2 template checks `enable_thinking` in exactly one place — the generation prompt — and never in the message rendering loop. Historical assistant messages receive `<think>` wrapping based solely on position relative to `last_query_index`. The upstream Ollama renderer at commit `9896e36` gates this on `isThinking`, which means that when a user switches from `think: true` to `think: false` between conversation turns, all previous assistant thinking blocks are silently stripped from the rendered prompt. The model then sees its own previous responses with reasoning removed — a prompt shape it was never trained on. This is not a subtle formatting difference — the entire `<think>...\n</think>\n\n` wrapper disappears, shifting every subsequent token ID and invalidating the KV cache from the first affected assistant message onward.

**Why both parts matter together:** If only Part 2 is re-introduced (the `isThinking &&` gate at line 143), the thinking blocks disappear from the rendered output even though the reasoning was correctly extracted. If only Part 1 is re-introduced (the `isThinking` parameter to `splitQwen35ReasoningContent`), the extraction itself silently returns empty reasoning, so line 143 wraps an empty string — producing `<think>\n\n</think>\n\n` instead of `<think>\nactual reasoning\n</think>\n\n`. Both regressions corrupt the prompt relative to the official template's output. A test that renders with `think=false` and checks for the presence of historical thinking content in the output catches both regressions: Part 1 would cause the reasoning text to be missing inside the `<think>` tags, and Part 2 would cause the `<think>` tags themselves to be missing.

**Why no existing test catches a regression:** All 7 renderer tests in `qwen35_test.go` construct the `Qwen35Renderer` with `isThinking: true`. No test creates a renderer with `isThinking: false` (or passes `ThinkValue{Value: false}` to `Render()`) while also including historical assistant messages that have non-empty `Thinking` fields. If someone added back an `isThinking &&` gate at line 143, all tests would still pass because `isThinking` is always `true`.

**What the test needs to verify:** Create a multi-turn conversation where:
- Turn 1 used `think: true` — the assistant message has `Thinking: "some reasoning"` and `Content: "some answer"`.
- Turn 2 uses `think: false` — the user sends a follow-up.

Render with `ThinkValue{Value: false}`. Verify that:
1. The historical assistant message at turn 1 still renders with `<think>\nsome reasoning\n</think>\n\n` wrapping — the thinking block is NOT stripped despite `think: false` on the current turn.
2. The generation prompt at the end uses the non-thinking prefill (`<think>\n\n</think>\n\n`) — only the current generation is affected by `think: false`, not the history.

This should also be tested in reverse: turn 1 used `think: false` (assistant message has `Thinking: ""` and `Content: "answer"`), turn 2 uses `think: true`. If the historical assistant message is positioned after `lastQueryIndex`, it should render WITH an empty thinking block — `<think>\n\n</think>\n\nanswer` — because the renderer wraps every post-`lastQueryIndex` assistant message in `<think>` tags unconditionally, even when reasoning is empty. This matches the official template, which produces the same empty `<think>` block when `reasoning_content` is an empty string. If the historical assistant message is positioned at or before `lastQueryIndex` (for example, because a new user query follows it), it should render as plain `answer` with no `<think>` wrapping — and this is because of its position, not because reasoning is empty. The generation prompt should use `<think>\n` (thinking enabled for the current turn).

**Status: DONE.** Implemented as a `think=false` subtest on the existing `TestQwen35RendererInterleavedThinkingAndTools` test (commit `af78cfb2`). The test reuses the same message array (two consecutive assistant messages with non-empty `Thinking` fields positioned after `lastQueryIndex`, separated by tool responses), renders with `ThinkValue{Value: false}`, and verifies: (1) both historical assistant messages retain their `<think>` blocks with full reasoning text despite think=false on the current turn, and (2) the generation prompt at the end uses the non-thinking form (`<think>\n\n</think>\n\n`). Both regression vectors were verified: temporarily re-adding `isThinking &&` to line 143 causes the think=false subtest to fail (tags vanish) while think=true passes; temporarily re-adding `isThinking` to `splitQwen35ReasoningContent`'s signature causes the think=false subtest to fail (reasoning text vanishes inside empty tags) while think=true passes.

---

## Gap 3: No Unit Tests for `splitQwen35ReasoningContent`

### What the function does and why it exists

Every historical assistant message the model sees during a multi-turn conversation is rendered as a specific byte sequence. For messages positioned after `lastQueryIndex` (the last real user query), the official Qwen 3.5 Jinja2 template produces:

```
<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}...tool calls...<|im_end|>\n
```

The model was trained on billions of tokens in this exact format. The `{reasoning}` and `{content}` values are not stored directly — they are extracted from the message object at render time, because different clients store them differently (see below). The function `splitQwen35ReasoningContent` at `model/renderers/qwen35.go` lines 64-79 performs this extraction. Its two return values (`reasoning` and `remaining`) are placed into the byte sequence at line 143-144 of the renderer:

```go
sb.WriteString(imStartTag + message.Role + "\n<think>\n" + contentReasoning + "\n</think>\n\n" + content)
```

If this function returns wrong values — wrong reasoning, wrong remaining content, extra characters like a duplicate `<think>` tag, a leftover `\n` — the rendered prompt deviates from what the model was trained on. The model does not error. It silently produces worse output: lower tool-calling accuracy, weaker reasoning, more hallucinations. The user blames the model, not the inference engine. Additionally, because Ollama reuses the KV cache (expensive intermediate computation) by comparing the new prompt against the previous one byte-by-byte, any byte difference in the historical portion invalidates the cache from that point forward, making every subsequent response in a long agentic session progressively slower to generate.

### The official template's extraction algorithm and the fork's equivalent

The official Jinja2 template (from `Qwen/Qwen3.5-27B` on HuggingFace, in the `chat_template` field of `tokenizer_config.json`) extracts reasoning from assistant messages using two paths. This is the verbatim Jinja2 code (with escaped newlines expanded for readability):

```jinja2
{%- set reasoning_content = '' %}
{%- if message.reasoning_content is string %}
    {%- set reasoning_content = message.reasoning_content %}
{%- else %}
    {%- if '</think>' in content %}
        {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
        {%- set content = content.split('</think>')[-1].lstrip('\n') %}
    {%- endif %}
{%- endif %}
{%- set reasoning_content = reasoning_content|trim %}
```

The fork's equivalent at `model/renderers/qwen35.go` lines 64-79:

```go
func splitQwen35ReasoningContent(content, messageThinking string) (reasoning string, remaining string) {
    if messageThinking != "" {
        return strings.TrimSpace(messageThinking), content
    }
    if idx := strings.Index(content, qwen35ThinkCloseTag); idx != -1 {
        before := content[:idx]
        if open := strings.LastIndex(before, qwen35ThinkOpenTag); open != -1 {
            reasoning = before[open+len(qwen35ThinkOpenTag):]
        } else {
            reasoning = before
        }
        content = strings.TrimLeft(content[idx+len(qwen35ThinkCloseTag):], "\n")
    }
    return strings.TrimSpace(reasoning), content
}
```

**How these correspond, step by step:**

| Official template operation | Fork equivalent | Notes |
|---|---|---|
| `if message.reasoning_content is string` → use it directly | `if messageThinking != ""` → return it directly | See "Known divergence" below for the empty-string difference |
| `content.split('</think>')[0]` — everything before the first `</think>` | `content[:idx]` where `idx = strings.Index(content, "</think>")` — same thing | Both find the **first** `</think>`. Equivalent for any number of `</think>` tags. |
| `.rstrip('\n')` — strip trailing newlines from the text before `</think>` | No explicit equivalent; the final `strings.TrimSpace(reasoning)` strips all trailing whitespace | `TrimSpace` is a superset of `rstrip('\n')`, but since the final `\|trim` (below) also strips all whitespace, the end result is identical |
| `.split('<think>')[-1]` — everything after the **last** `<think>` tag | `strings.LastIndex(before, "<think>")` — finds the **last** `<think>` and takes everything after it | Both discard any text before the last `<think>` tag. When there is no `<think>` tag, the template's `split` returns a one-element list and `[-1]` returns the whole string; the fork's `LastIndex` returns -1 and the `else` branch at line 74 takes the whole `before` string. Equivalent. |
| `.lstrip('\n')` — strip leading newlines from the extracted reasoning | No explicit equivalent; the final `strings.TrimSpace(reasoning)` strips all leading whitespace | Same reasoning as above — end result identical after final trim |
| `content.split('</think>')[-1].lstrip('\n')` — remaining content is everything after the **last** `</think>`, with leading newlines stripped | `strings.TrimLeft(content[idx+len("</think>"):], "\n")` — remaining content is everything after the **first** `</think>`, with leading newlines stripped | **Divergence when content has multiple `</think>` tags**: the template takes after the last, the fork takes after the first. In practice the model never produces multiple `</think>` tags — the grammar-constrained generation pipeline and the parser both enforce a single thinking block. This divergence is documented but not tested. |
| `reasoning_content\|trim` — Jinja2's `trim` filter strips all whitespace from both ends | `strings.TrimSpace(reasoning)` — Go's `TrimSpace` strips all Unicode whitespace from both ends | Equivalent |

**The extraction-then-wrapping cycle:** Both the template and the fork extract reasoning and content, then re-wrap them in `<think>\n{reasoning}\n</think>\n\n{content}`. This means the model's raw output — which includes `\n` after `<think>` and `\n` before `</think>` — goes through a strip-then-re-add cycle. The raw output `<think>\nLet me check\n</think>\n\nParis is 18°C.` is extracted to reasoning=`"Let me check"` and content=`"Paris is 18°C."` (newlines stripped), then the renderer re-wraps as `<think>\nLet me check\n</think>\n\nParis is 18°C.` (newlines re-added). The final bytes are identical to the raw output. This is the same cycle the official template performs. The model sees the same bytes regardless of whether this is the first render or a re-render on a subsequent turn.

### Why two extraction paths exist — different clients encode the same model output differently

The function has two main extraction paths because different client applications store the same model output in different ways, and the renderer must produce the correct byte sequence regardless of which client sent the conversation history.

**Path 1 — Explicit `Thinking` field (line 65-66).** When a client application (Claude Code, a custom agentic framework, etc.) receives the model's streaming response, Ollama's parser separates the thinking text from the visible content and delivers them as two separate fields: `Thinking` (the reasoning inside `<think>...</think>`) and `Content` (everything after). A well-behaved client stores these as separate fields in its conversation history. On the next turn, the client sends both fields back. Path 1 handles this: if the `message.Thinking` field (called `messageThinking` in the function parameter) is non-empty, use it as the reasoning and return the `Content` field unchanged.

**Path 2 — Inline `<think>` tags in content (lines 69-76).** Not all clients handle the `Thinking` field. Some third-party clients (Open WebUI, LangChain integrations, custom tools, older client libraries) either don't recognize the `Thinking` field and drop it during JSON serialization, or store the raw model output as a single string with the `<think>` tags still embedded. When these clients send the conversation history back, the `Thinking` field is empty, but the `Content` field contains the raw model output with tags.

Path 2 has a sub-case that is not an edge case but the most common inline format: **Path 2b — close tag without opening tag (line 73-74).** When thinking is enabled, the renderer prefills `<|im_start|>assistant\n<think>\n` before the model starts generating. The model then produces `reasoning text\n</think>\n\nvisible answer`. The `<think>` open tag is part of the renderer's prefill, not part of the model's output. A client that captures only the model's generated tokens (not the prefill) stores `reasoning text\n</think>\n\nvisible answer` — a close tag without an opening tag. This is the realistic Path 2 input format. Path 2a (with both tags) occurs when a client captures the full output including the prefill.

**Path 3 — No thinking at all (line 79).** Neither the `Thinking` field nor inline tags are present. The function returns empty reasoning and the content unchanged.

### The critical property: all paths must produce the same `(reasoning, remaining)` for equivalent input

Three different clients can encode the exact same assistant turn three different ways:

- **Client A** (well-behaved, stores fields separately): `{Thinking: "Let me check the weather", Content: "Paris is 18°C."}`
- **Client B** (stores raw output with prefill): `{Content: "<think>\nLet me check the weather\n</think>\n\nParis is 18°C."}`
- **Client C** (stores raw output without prefill): `{Content: "Let me check the weather\n</think>\n\nParis is 18°C."}`

All three represent the same model turn. All three must produce the same `(reasoning, remaining)` tuple: `("Let me check the weather", "Paris is 18°C.")`. The renderer then wraps both into the same byte sequence: `<|im_start|>assistant\n<think>\nLet me check the weather\n</think>\n\nParis is 18°C.`. The model sees the same prompt regardless of which client sent the history.

If any path produces different values — a leftover newline, a duplicate `<think>` tag, a missing whitespace strip — the model sees a prompt it was never trained on, and the KV cache is invalidated from that point forward.

### Relationship to Gap 2 — same fix, different test level

This function's signature was changed as part of the same fix described in Gap 2. The upstream Ollama version (commit `9896e36`) has the signature `splitQwen35ReasoningContent(content, messageThinking string, isThinking bool)` and gates Path 1 on `if isThinking && messageThinking != ""`. The fork removed the `isThinking` parameter entirely, changing the guard to `if messageThinking != ""`.

What this means: in the upstream version, when a user switches from `think: true` to `think: false` between conversation turns (a common pattern — enable thinking for complex planning, disable it for quick follow-ups), the upstream function receives `isThinking=false`. The guard evaluates to `false && true` → `false`, skipping Path 1. The function falls through to Path 2, which searches the content for inline tags. If the content has no inline tags (because the client properly stored thinking in the `Thinking` field), Path 2 finds nothing, and the function returns empty reasoning. The historical assistant message's thinking is silently erased. The model sees a conversation where it previously answered without thinking, even though it actually did think. This is a prompt shape the model was never trained on. The official template never does this — its `message.reasoning_content is string` check has no `enable_thinking` gate.

Gap 2 tests the renderer-level consequence of this fix. This gap tests the extraction function's internal logic in isolation. Both are needed: Gap 2 catches someone re-adding the `isThinking` gate; this gap catches bugs in the extraction logic itself (wrong tag stripping, wrong fallback priority, wrong whitespace handling).

### Known divergence from the official template

The fork's Path 1 guard is `messageThinking != ""`. In Go, the `Thinking` field on `api.Message` is a plain `string` with zero value `""`. Go's JSON unmarshaler sets it to `""` both when the field is absent from JSON and when it is explicitly `"thinking": ""`. These are indistinguishable.

The official template's guard is `message.reasoning_content is string`. In Python/Jinja2, `None` (field absent) is NOT a string, but `""` (explicitly empty) IS a string. The official template distinguishes "absent" (fall through to Path 2) from "empty string" (use it, skip Path 2). The fork cannot make this distinction.

Practical consequence: if a message has `Thinking: ""` (indistinguishable from absent in Go) AND `Content` contains `</think>`, the fork falls through to Path 2 and parses the content. The official template would take Path 1 with the empty string. In practice, a model would not produce `</think>` in visible content, so this divergence does not affect real inputs. The tests should document this explicitly.

### Why no existing test catches bugs in this function

The function is unexported (`splitQwen35ReasoningContent`, lowercase first letter) but accessible from tests because `qwen35_test.go` is in the same `renderers` package.

No test calls it directly. All existing tests that include assistant messages with thinking content use a non-empty `Thinking` field (`"Need to call add and multiply."`, `"Need weather before giving advice."`, `"Need UV index for sunscreen advice."`, `"Keep it short."`, `"Let me check."`). Every one hits Path 1 and returns immediately. Path 2 — both 2a (with open tag) and 2b (without open tag) — has zero test coverage. A bug in Path 2 would only manifest when a third-party client sends conversation history with inline thinking tags.

### What the tests need to verify

One test function — `TestSplitQwen35ReasoningContent` — with table-driven subtests calling the function directly. All test inputs use the realistic content format the model actually generates (with `\n` after `<think>` and `\n` before `</think>`), not sanitized strings without newlines.

**Test 1 — Path equivalence (the most important test).** Three encodings of the same model turn must produce identical `(reasoning, remaining)`:

- Client A (explicit field): `splitQwen35ReasoningContent("Paris is 18°C.", "Let me check the weather")` — Path 1
- Client B (inline with both tags): `splitQwen35ReasoningContent("<think>\nLet me check the weather\n</think>\nParis is 18°C.", "")` — Path 2a
- Client C (inline without open tag): `splitQwen35ReasoningContent("Let me check the weather\n</think>\nParis is 18°C.", "")` — Path 2b

All three must return `("Let me check the weather", "Paris is 18°C.")`. The `\n` after `<think>` and `\n` before `</think>` must be stripped by the extraction — the official template strips them via `lstrip('\n')` and `rstrip('\n')` before the final `|trim`, and the fork strips them via `strings.TrimSpace` on reasoning and `strings.TrimLeft(..., "\n")` on remaining content. The renderer then re-adds `\n` on both sides when wrapping: `<think>\n` + reasoning + `\n</think>`. The round-trip produces the same bytes.

**Test 2 — Inline tags with internal newlines (realistic multi-line reasoning).** `splitQwen35ReasoningContent("<think>\nStep 1: look up weather\nStep 2: format answer\n</think>\nParis is 18°C.", "")` → reasoning `"Step 1: look up weather\nStep 2: format answer"`, remaining `"Paris is 18°C."`. The `\n` immediately after `<think>` and immediately before `</think>` are stripped. The internal `\n` between reasoning steps is preserved. This matches the official template: `rstrip('\n')` strips only the trailing newline before `</think>`, `lstrip('\n')` strips only the leading newline after `<think>`, and `|trim` strips surrounding whitespace — but internal newlines survive because `trim` only operates on the ends.

**Test 3 — Close tag without open tag (Path 2b, the model's raw output after prefill).** `splitQwen35ReasoningContent("Let me check\n</think>\nParis is 18°C.", "")` → reasoning `"Let me check"`, remaining `"Paris is 18°C."`. The official template's `content.split('</think>')[0]` does not require a `<think>` tag. The subsequent `.split('<think>')[-1]` is a no-op when no `<think>` exists (Python's `"text".split('<think>')` returns `["text"]`, and `[-1]` returns `"text"`). The fork's `LastIndex` returns -1 and the `else` branch at line 74 takes the entire `before` string. Both produce the same result.

**Test 4 — Explicit field wins over inline tags (no double-extraction).** `splitQwen35ReasoningContent("<think>\ninline reasoning\n</think>\nvisible", "explicit reasoning")` → reasoning `"explicit reasoning"`, remaining `"<think>\ninline reasoning\n</think>\nvisible"`. Path 1 fires at line 65 and returns immediately. The content is NOT parsed — it is returned exactly as received, with `<think>` tags still present as literal characters. The renderer wraps only the explicit field's reasoning in `<think>...\n</think>` tags. The inline tags in content become literal text in the visible portion of the prompt. This test confirms no double-extraction occurs.

**Test 5 — No thinking (Path 3 baseline).** `splitQwen35ReasoningContent("just content, no thinking", "")` → reasoning `""`, remaining `"just content, no thinking"`. Neither `messageThinking != ""` nor `strings.Index(content, "</think>") != -1`. The renderer wraps the empty reasoning as `<think>\n\n</think>\n\n` — an empty thinking block — matching the official template.

**Test 6 — Whitespace-only `Thinking` field.** `splitQwen35ReasoningContent("content", "   \n  ")` → reasoning `""`, remaining `"content"`. The string `"   \n  "` is not `""`, so Path 1 fires. `strings.TrimSpace("   \n  ")` returns `""`. Content is returned unchanged — Path 2's tag parsing is never reached. The official template's `message.reasoning_content is string` would also be true for `"   \n  "`, and `|trim` would also return `""`. The behaviors match. This documents that whitespace-only fields do NOT fall through to content parsing.

**Test 7 — Empty content with explicit thinking (tool-call-only turn).** `splitQwen35ReasoningContent("", "Let me call the function")` → reasoning `"Let me call the function"`, remaining `""`. Path 1 fires. The model reasoned but produced no visible text — its response is entirely tool calls. The renderer wraps this as `<think>\nLet me call the function\n</think>\n\n` followed by tool call XML. This is a common pattern in agentic use.

**Test 8 — Both empty.** `splitQwen35ReasoningContent("", "")` → reasoning `""`, remaining `""`. Path 3. The assistant message has no thinking and no visible content — the turn is entirely tool calls with no reasoning. The renderer wraps as `<think>\n\n</think>\n\n` followed by tool call XML.

### Think mode variants

Not needed. The fork's `splitQwen35ReasoningContent` is a pure function of its two string arguments. It has no access to `isThinking`, `ThinkValue`, or any renderer state — the `isThinking` parameter was deliberately removed as part of the fix described in Gap 2. Think mode only affects downstream behavior: whether the renderer wraps historical messages in `<think>` tags (tested by Gap 2) and whether the generation prompt uses `<think>\n` or `<think>\n\n</think>\n\n` (tested by other tests).

---

## Gap 4: The Tool Definition Field Ordering Test Locks In the Wrong Ordering

**What the bug is:** The Go `api.ToolFunctionParameters` struct declares its fields in the order `Type`, `Required`, `Properties`. Go's `json.Marshal` serializes struct fields in declaration order. The result is `{"type": "object", "required": [...], "properties": {...}}`. The official Qwen 3.5 Jinja2 template uses HuggingFace Transformers' `tojson` filter, which calls Python's `json.dumps` with insertion-order preservation. HuggingFace's `get_json_schema()` function (at `chat_template_utils.py` lines 210-213) constructs the parameters dict with `properties` before `required`. The model was trained on `{"type": "object", "properties": {...}, "required": [...]}`.

**Why this matters for the test suite specifically:** The test at `model/renderers/qwen35_test.go` line 259 hardcodes the expected JSON as:
```
"parameters": {"type": "object", "required": ["location"], "properties": {"location": {...}, ...}}
```
This is `required` before `properties` — the Go struct order, not the HuggingFace training data order. If someone fixes the field ordering (by implementing a custom marshaler for `ToolFunctionParameters` that puts `properties` before `required`), this test will **fail** and signal a false regression. The test is actively preventing the fix.

**What needs to change:** The test's expected JSON string at line 259 should use the HuggingFace/OpenAI convention: `"parameters": {"type": "object", "properties": {"location": {...}, ...}, "required": ["location"]}`. This change should be made at the same time as the actual fix to the serialization ordering, so that the test transitions from locking in the wrong behavior to locking in the correct behavior. Until the fix is implemented, the test should be annotated with a comment explaining that its expected output reflects a known deviation from the official template's field ordering.

**Note on blast radius:** `ToolFunctionParameters` is a shared type used by every tool-calling renderer in Ollama. Changing its marshaling behavior requires verifying that all other model families also expect `properties` before `required`, or implementing a renderer-local custom marshaler. This verification work is documented in `consolidated_report.md` Part 4 (Research topics 1-5). The test change should wait until the verification is complete and the fix is implemented.

---

## Gap 5: No Test for Thinking Mode Switching Across a Full Agentic Loop (KV Cache Round-Trip)

**What is missing:** The 22 KV cache round-trip test cases in `server/qwen35_kvcache_roundtrip_test.go` all use a single `think` value for the entire test. No test simulates a multi-turn agentic conversation where thinking mode changes between turns — for example, turn 1 with `think: true` (the assistant produces reasoning + tool call), then turn 2 with `think: false` (the user wants a direct answer without reasoning).

**Why this matters for KV cache reuse:** When thinking mode switches, the renderer must still produce a byte-identical prefix for all historical messages. If the renderer incorrectly strips thinking blocks from history when `think: false` (the upstream Ollama bug that the fork fixed), the historical portion of the prompt changes, and the KV cache for the entire conversation history is invalidated. This is the most expensive possible cache miss.

**What the test needs to verify:** A round-trip test where:
1. Turn 1: `think=true`, assistant produces `<think>reasoning</think>\n\ncontent\n\n<tool_call>...`. Parse it, extract thinking, content, and tool call.
2. Turn 2: `think=false`, send back the parsed tool call + tool result + new user message. Render the new prompt.
3. Verify that the historical assistant message in the turn-2 prompt still contains `<think>reasoning</think>` (not stripped), so the KV cache prefix from turn 1 is reused.

**Relationship to Gap 2:** This gap tests the KV cache consequences of the same renderer behavior that Gap 2 tests at the unit level. Gap 2 verifies that the renderer output is correct (thinking blocks preserved in history when think mode switches). This gap verifies that the correct output actually produces byte-identical prefixes across the mode switch — a stronger assertion, because it is possible for the output to be semantically correct but byte-different in a way that invalidates the cache (for example, if the renderer introduced extra whitespace or reordered attributes in think=false mode). Both gaps are needed: Gap 2 catches the most obvious regression (thinking blocks stripped entirely), while this gap catches subtler regressions that only manifest as cache misses in multi-turn agentic conversations.

**Think mode variants:** Inherent to this gap — the entire point is testing behavior when the think mode changes between turns. The test requires at minimum two renders with different think values for the same conversation history. The minimum viable test covers think=true→think=false (the common direction: user starts with reasoning enabled, then switches to direct answers for speed). The reverse (think=false→think=true) is also worth testing but is lower priority because it is less common in practice and the renderer code path is symmetric.

---

## Gap 6: No Test for Tool Definition Serialization in the KV Cache Round-Trip

**What is missing:** The 22 KV cache round-trip test cases verify tool call **argument** round-trips (the `<parameter=...>` XML values). They do not verify tool **definition** round-trips (the JSON in the system prompt). The tool definitions are set up once at line 78-94 of the test file and are identical between the first and second renders, so they always match. But the test does not verify the actual bytes of the tool definition JSON against the official template's expected output.

**Why this matters:** If `marshalWithSpaces` changes its behavior (for example, if someone introduces key sorting, or if a Go upgrade changes `json.Encoder` behavior), the tool definition JSON in the system prompt would change between Ollama versions. A user who upgrades Ollama mid-conversation would see their entire KV cache invalidated on the next turn. More immediately, the test does not catch tool definition formatting bugs (like the `required`/`properties` ordering issue in Gap 4) because it only checks prefix reuse between two renders that use the same code path — both renders produce the same wrong output, so they match.

**What the test needs to verify:** At minimum, one test case that renders a prompt with tools and verifies the exact JSON bytes of each tool definition against a golden string that matches what HuggingFace Transformers' `apply_chat_template` produces for the same tool. This catches both formatting bugs and regressions.

---

## Gap 7: No Dedicated Tests for Grammar-Constrained Tool Call Generation

The grammar-constrained tool call generation feature (commits `b1038b91`, `59d1b367`, `4044b63f`, documented in `consolidated_report.md` Part 5) adds 212 lines across 8 files. It is the first grammar-constrained tool call implementation in any Ollama model. The following components have no dedicated tests:

**`ToolCallGrammarFromJSON` (Go wrapper at `llama/llama.go`):** No test calls this function with a JSON array of `api.Tool` definitions and verifies the returned GBNF grammar string. The function delegates to the C++ `tool_call_grammar_from_json()` via CGo, so a test would exercise both the Go wrapper and the C++ grammar builder. The test should verify: the grammar string contains the correct function name alternation (`tool-get-weather | tool-search`), the correct parameter name rules, and the `root ::= tool-call+` repetition.

**`NewGrammarLazy` and `NewGrammarSamplerLazy` (lazy grammar activation at `llama/llama.go` and `sample/samplers.go`):** No test verifies that a lazy grammar stays dormant (does not mask any tokens) until the trigger pattern matches, then activates and constrains generation. The existing `TestGrammar` test in `sample/samplers_test.go` only tests `NewGrammarSampler` (eager mode). A lazy grammar test would need to: create a lazy grammar with a trigger pattern, call `Apply()` with tokens that do not match the trigger (verify no tokens are masked), then call `Apply()` with tokens that do match the trigger (verify tokens are masked according to the grammar).

**Dead-end detection (`sample/samplers.go`):** The fork added a check after `s.grammar.Apply(tokens)` that detects when all tokens have been set to `-inf` by the grammar (indicating the grammar reached an impossible state) and returns a clear error message instead of the cryptic "logits sum to NaN" that would otherwise occur. No test verifies this error message is produced when a grammar rejects all tokens.

**Trigger pattern correctness (the thinking-mode-dependent regex patterns in `server/routes.go` lines 2249-2252):**
- Thinking mode pattern: `[\s\S]*</think>[\s\S]*?(<tool_call>[\s\S]*)` — requires `</think>` to appear before `<tool_call>`, preventing the grammar from activating on tool calls hallucinated inside thinking text.
- Non-thinking mode pattern: `[\s\S]*?(<tool_call>[\s\S]*)` — triggers on the first `<tool_call>` since no `</think>` is expected (the renderer already prefilled `<think>\n\n</think>\n\n`).

No test verifies that these regex patterns match the correct strings and reject the incorrect ones. For example: the thinking-mode pattern should NOT match `<think>I'll call <tool_call><function=foo>` (tool call inside thinking), and SHOULD match `reasoning</think>\n\n<tool_call><function=foo>` (tool call after thinking closes).

**Think mode variants for trigger patterns:** The trigger patterns are inherently mode-dependent — the thinking-mode pattern and non-thinking-mode pattern are different regexes selected based on think mode at `server/routes.go` lines 2249-2252. A test for trigger pattern correctness should verify both patterns, which implicitly covers both think modes. The thinking-mode pattern must reject tool calls inside `<think>` blocks (a safety property — the grammar should not activate on hallucinated tool calls in reasoning text). The non-thinking-mode pattern can trigger on the first `<tool_call>` because the renderer has already prefilled `<think>\n\n</think>\n\n`, guaranteeing that no thinking text precedes the tool call. The other grammar components (`ToolCallGrammarFromJSON`, `NewGrammarLazy`, dead-end detection) are independent of think mode — they operate on the grammar string and token logits, neither of which changes with think mode.

**Grammar wiring in `ChatHandler` (`server/routes.go`):** No test verifies that when a chat request arrives with `parser == "qwen3.5"` and tools are present, the `CompletionRequest` has its `Grammar` and `GrammarTriggerPatterns` fields populated. And that when no tools are present, these fields are empty.

---

## Gap 8: No Parser Test for Incomplete (Cancelled) Tool Calls

**What is missing:** The 4 cancellation scenarios in the KV cache round-trip test (`qwen35_kvcache_roundtrip_test.go` cases 19-22) verify the full pipeline: render → parse → verify no `ToolCall` objects are produced. But the `Qwen35Parser` and `Qwen3CoderParser` unit tests in `model/parsers/qwen35_test.go` and `model/parsers/qwen3coder_test.go` have no cancellation test — no test feeds an incomplete `<tool_call>` block (for example, `<tool_call>\n<function=get_weather>\n<parameter=location>\nParis`) to the parser and verifies that zero tool calls are returned and the partial XML is not surfaced as content.

**Why a unit-level test matters in addition to the integration test:** The integration test exercises the cancellation path through the full render-parse-rerender pipeline, which provides confidence that the system behaves correctly end-to-end. But a parser unit test isolates the parser's behavior and makes the assertion more precise: does the parser hold back the partial tool call XML in its internal buffer? Does it emit the content before the `<tool_call>` tag? Does it emit nothing? The answers to these questions affect what the user sees in their client during a cancelled generation.

**What the test needs to verify:**
- Input: `Some content before<tool_call>\n<function=get_weather>\n<parameter=location>\nParis` (no closing `</parameter>`, `</function>`, or `</tool_call>`).
- Expected: `content` should be `"Some content before"` (or empty, depending on whether the parser emits content before an incomplete tool call — this needs to be verified against actual behavior and documented). `thinking` should be empty. `calls` should be empty (no completed tool calls).
- Streaming variant: feed the same input character by character and verify the same final state.

**Think mode variants:** The parser unit test should cover both `think=true` (parser starts in `CollectingThinking` state, must encounter `</think>` before `<tool_call>` is parsed as a tool call) and `think=false` (parser starts in `CollectingContent` state, `<tool_call>` is immediately parseable). In think=true mode, a cancelled generation might contain only thinking text without a closing `</think>`, in which case a partial `<tool_call>` inside the thinking block should be treated as literal thinking text, not as a tool call attempt. In think=false mode, the partial `<tool_call>` is directly in the content stream and is held back by the parser's incomplete-tag buffering. These are different code paths through the parser — the think=true path goes through the thinking state machine in `Qwen35Parser`, while the think=false path goes directly to the `Qwen3CoderParser`'s tool call detection — and both should be tested for correct cancellation behavior.

---

## Gap 9: No Test for Mixed Argument Types Across Parallel Tool Calls

**What is missing:** Tests that include multiple tool calls in one assistant turn (`TestQwen35RendererBackToBackToolCallsAndResponses`, `TestQwen35RendererInterleavedThinkingAndTools`) use only scalar arguments (strings and integers). No test combines scalar arguments in one tool call with structured JSON arguments (maps or arrays) in another tool call within the same assistant message.

**Why this matters:** In real agentic use, the model frequently calls multiple tools in one turn with different argument types. For example: `get_weather(location="Paris")` (string scalar) alongside `filter(criteria={"temp_min": 20, "unit": "celsius"})` (JSON object). The `formatToolCallArgument` function at `model/renderers/qwen3coder.go` line 235 handles scalars and structured types through different code paths (strings are returned verbatim, maps go through `marshalQwenToolCallArgument` which uses `json.Encoder` with `SetEscapeHTML(false)` and space insertion). A test with mixed types in parallel tool calls would exercise both paths in a single render call and verify they don't interfere with each other.

**What the test needs to verify:** An assistant message with two tool calls — one with a string argument and one with a `map[string]any` argument. The rendered output should contain the string argument verbatim (no JSON encoding) and the map argument as spaced, non-HTML-escaped JSON. The `<tool_call>` blocks should be properly separated.

**Think mode variants:** Not needed. Tool call argument formatting is handled by `formatToolCallArgument` at `model/renderers/qwen3coder.go` line 235, which takes a Go value and returns a string. It has no access to think mode state. The XML structure around tool calls (`<tool_call>`, `<function=...>`, `<parameter=...>`) is rendered identically in both think modes. The only think-mode-dependent aspects of an assistant message are the `<think>` block wrapping (which is unconditional for messages after `lastQueryIndex` and does not interact with tool call rendering) and the generation prompt suffix (which appears after `<|im_end|>`, outside the assistant message body). Neither interacts with the tool call argument formatting code paths being tested.

---

## Gap 10: No Test That `formatToolCallArgument` Is Exercised Through the Qwen35 Renderer Path

**What is missing:** The `formatToolCallArgument` function is tested directly in `qwen3coder_test.go` (`TestFormatToolCallArgument`, lines 490-536), but those tests call the function in isolation. The Qwen35 renderer calls `formatToolCallArgument` at `model/renderers/qwen35.go` line 162, but the test that verifies this (`TestQwen35RendererStructuredToolArgumentsUseSpacedJSON`) uses `strings.Contains` to check a substring — it does not verify the full tool call block byte-by-byte. If the Qwen35 renderer's tool call rendering code at lines 149-166 had a bug in how it constructs the XML around the formatted argument (for example, a missing newline between `<parameter=name>` and the value), the substring check might still pass if the substring spans the correct portion.

**What would strengthen this:** The `TestQwen35RendererStructuredToolArgumentsUseSpacedJSON` test should use a byte-exact comparison of the entire assistant message block (like `TestQwen35RendererAssistantPrefillWithThinking` does at line 375 with `got != want`), not just a `strings.Contains` substring check. This would catch any whitespace or newline deviation in the XML structure around the argument.

**Think mode variants:** Not needed — same reasoning as Gap 9 (tool call rendering is independent of think mode). If byte-exact comparison is adopted, the expected output would include the generation prompt suffix, which differs by think mode, but that is incidental to this gap's purpose and can be handled by testing in one mode only.

---

## Gap 11: No Test for `lastQueryIndex` Edge Cases

**What `lastQueryIndex` does:** At `model/renderers/qwen35.go` lines 114-127, the renderer walks messages backwards to find the last user message that is NOT a `<tool_response>`-wrapped message. Messages after `lastQueryIndex` get `<think>` blocks unconditionally (line 143). Messages at or before `lastQueryIndex` get plain rendering (line 146).

**What is missing:** No test verifies the behavior when:
- All user messages are `<tool_response>`-wrapped (meaning every user message is a tool result). In this case, `lastQueryIndex` stays at `len(messages) - 1` (the initial value), and no assistant messages get `<think>` blocks. The official Jinja2 template raises an exception in this case (`raise_exception("No user message found")`), but the Go renderer silently proceeds. A test should document which behavior the fork chooses and why.
- There is only one user message and it is a `<tool_response>` message. Same edge case as above but simpler.
- The conversation has no user messages at all (only system + assistant). `lastQueryIndex` stays at `len(messages) - 1`. The assistant message would be at an index <= `lastQueryIndex`, so it gets plain rendering (no `<think>` block). A test should verify this.

**Think mode variants:** The `lastQueryIndex` computation itself is independent of think mode — it walks messages backwards looking for non-tool-response user messages, with no reference to `isThinking`. However, the edge case where `lastQueryIndex` stays at `len(messages) - 1` (no non-tool-response user message found) means no assistant messages get `<think>` wrapping, yet the generation prompt at the end still depends on think mode. A test for these edge cases should verify both think=true and think=false to confirm the generation prompt is correct even when no assistant messages receive `<think>` blocks.

---

## Relationship Between Gaps

Gaps 1 and 2 are now closed. These were the two most urgent gaps because they protect the fork's two most important correctness fixes — fixes that differentiate this fork from upstream Ollama and that would silently regress without dedicated test coverage.

**Gaps 2 and 3 are two levels of the same fix.** The fork's unconditional thinking block fix involved two code changes: removing the `isThinking` parameter from `splitQwen35ReasoningContent` (line 65, tested by Gap 3) and removing the `isThinking &&` gate from the rendering condition (line 143, tested by Gap 2). Gap 2 is now closed — it tests the renderer-level consequence and catches regressions to either part of the fix. Gap 3 remains open and would add depth by testing the extraction function's internal code paths in isolation — ensuring each path (explicit `Thinking` field, inline `<think>` tags in content, no thinking) produces the correct output independent of how the renderer uses it.

Gap 4 (wrong field ordering locked in by tests) should be addressed simultaneously with the actual ordering fix, not before — changing the test expectation without changing the code would cause the test to fail.

Gap 5 (thinking mode switching in KV cache) tests the cache-level consequences of the same renderer behavior that Gap 2 tests at the unit level. Both are needed — Gap 2 catches thinking blocks being stripped, Gap 5 catches subtler byte-level divergences that only manifest as cache misses.

Gap 7 (no grammar tests) is important for long-term maintenance but lower urgency because the grammar feature is new, the code is unlikely to be accidentally modified, and the grammar's correctness is validated indirectly through the KV cache round-trip tests and real-model integration tests.

Gaps 6, 8, 9, 10, and 11 are medium-priority gaps that improve confidence in edge cases. They are ordered roughly by the likelihood and severity of the bugs they would catch.

**Think mode coverage as a cross-cutting concern:** Gaps 1 and 2 have closed the most critical think=false coverage holes (generation prompt after tool calls, and historical thinking block preservation). The remaining gaps that need both think modes are: Gap 5 (inherent to mode switching), Gap 8 (different parser code paths), and Gap 11 (generation prompt correctness in edge cases). The gaps that do not need think=false variants are: Gap 3 (pure function), Gap 4 (JSON serialization), Gap 6 (tool definition bytes), Gap 7 (grammar components other than trigger patterns), Gap 9 (argument formatting), and Gap 10 (XML structure). Gap 7's trigger pattern sub-component is inherently mode-dependent and requires both patterns to be tested.
