# BigBIueWhale/ollama Fork: Comprehensive Research Report for Qwen 3.5 27B

**Date:** 2026-03-13 (updated 2026-03-21: Section 4.2 corrected with verified HuggingFace Transformers ground truth — exact `ToolProperty` struct layout specified, `items` ordering confirmed non-issue, blast radius table added; new Section 4.4 for `formatToolCallArgument` scalar divergences; action items updated)
**Fork:** `BigBIueWhale/ollama` on branch `main`, merge base `cc90a035` (Ollama `v0.17.4`). Use `git log --oneline cc90a035..HEAD` for the current commit list.
**Model:** Qwen 3.5 27B (Alibaba Qwen), a hybrid recurrent architecture (`qwen3next`) combining GatedDeltaNet recurrent layers with standard full-attention layers. Running via Unsloth Dynamic 2.0 `UD-Q4_K_XL` GGUF quantization (17.6 GB, 851 tensors, text-only, at `/tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf`)
**Inference engine:** Ollama — a Go-based inference engine that vendors the GGML tensor math library from the llama.cpp project (maintained by ggml-org) but reimplements everything else in Go: chat template rendering uses hardcoded Go renderers instead of llama.cpp's Jinja2 template engine, sampling is a Go rewrite instead of llama.cpp's `llama-sampler.cpp`, and the model runner is `ollamarunner` instead of llama.cpp's HTTP server. This means llama.cpp bugs in GGML affect Ollama, but llama.cpp's correct Jinja2 template handling does NOT help Ollama — Ollama must reimplement the same logic in Go renderers and parsers.
**Upstream Ollama (previous):** `ollama/ollama` master @ `82848a78` (commit message: `model: fix renderer and parser for qwen3.5 (#14605)`)
**Upstream Ollama (latest):** `ollama/ollama` master @ `9896e36` (4 commits newer than `82848a78`)
**llama.cpp (previous):** `ggml-org/llama.cpp` master @ `d969e933` (commit hash: `d969e933e172821b4519f66aa4b660bc0846b320`)
**llama.cpp (latest):** `ggml-org/llama.cpp` master @ `a0ed91a` (March 5–6, 2026)
**llama.cpp (newest):** `ggml-org/llama.cpp` master @ `c5a7788` ("ggml: add GATED_DELTA_NET op"). Shallow clone (`--depth=1`). Trie upgraded to `uint32_t` Unicode codepoints in `peg-parser.cpp`. Tool grammar builder refactored from `chat.cpp` into new `chat-peg-parser.cpp` with parameterized markers. Public API (`json-schema-to-grammar.h`) byte-for-byte identical to vendored version.
**llama.cpp (post-refactor):** Two additional clones exist at `/tmp/llama-cpp-fresh/` @ `0cd4f47` and `/tmp/llama_cpp_devstral2512_audit/` @ `aa2d278` (both March 10, 2026). These contain a significant refactoring of the chat template handling: the per-model specialized handlers (`common_chat_params_init_qwen3_coder`, `common_chat_params_init_qwen3_vl`, etc.) have been replaced with a unified autoparser path in `common_chat_template_direct_apply()`. This refactoring fixes the `enable_thinking` generation prompt bug described in Section 7.1 — the unified path correctly passes `enable_thinking` to the Jinja template context at `chat.cpp:773`, so the template's own `add_generation_prompt` block produces the correct 6-token non-thinking prefill. See Section 7.1 for details.
**Official reference:** Qwen 3.5 Jinja2 template from `Qwen/Qwen3.5-35B-A3B` ([publicly accessible on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/tokenizer_config.json)). **Also verified** against `Qwen/Qwen3.5-27B` ([publicly accessible](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/tokenizer_config.json)) — both repositories contain byte-identical `chat_template` fields.
**Qwen3-Coder reference:** `Qwen/Qwen3-Coder-30B-A3B-Instruct` Jinja2 template ([verified via public access](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)) — confirms XML tool definitions and system-first ordering. **Important nuance:** `Qwen3CoderRenderer` is directionally correct but not fully byte-faithful. Two remaining fidelity gaps are confirmed from the official template: `renderAdditionalKeys()` should use the template's `render_extra_keys(... | tojson | safe)` behavior for nested map/list values, and multi-type `type` fields are rendered with Python string formatting in the template rather than JSON serialization.
**GGUF template note:** The Unsloth Dynamic 2.0 `UD-Q4_K_XL` GGUF embeds a **modified** Jinja2 chat template — not byte-identical to the official `Qwen/Qwen3.5-27B` HuggingFace template. Unsloth changed the tool call arguments section: `tool_call.arguments is defined` → `tool_call.arguments is mapping`, and replaced `tool_call.arguments|items` tuple unpacking with key iteration + bracket access (`for args_name in tool_call.arguments` / `tool_call.arguments[args_name]`). This is a defensive workaround (likely for older llama.cpp Jinja engine versions where `|items` didn't return proper tuples). Functionally equivalent for normal use because llama.cpp's `func_args_not_string` workaround (at `common/chat.cpp:2947-2965`) ensures arguments are always parsed objects before template execution. The `enable_thinking` handling and all other template logic is identical to the official template. llama.cpp's GGUF converter (`convert_hf_to_gguf.py`) does NOT modify templates — this is a Unsloth-specific patch. Impact on Ollama users: none — Ollama does not execute the GGUF-embedded Jinja template; it uses its own Go renderers.

---

## Implementation Coverage Note: Manual Go versus Template-Driven Tool Calling in Ollama

Not every tool-calling model supported in Ollama uses the same implementation strategy. This matters for the JSON round-trip bug and for any future shared serializer changes: some models are fully hand-implemented in Go, while others still rely on model-supplied templates plus Ollama's generic tool parser path.

**Full manual Go renderer + manual Go parser** (`model/renderers/renderer.go`, `model/parsers/parsers.go`):
- `qwen3.5`, `qwen3-coder`, `qwen3-vl-instruct`, `qwen3-vl-thinking`, `lfm2`, `lfm2-thinking` / `lfm2.5-thinking`, `cogito`, `deepseek3.1` renderer + `deepseek3` parser, `olmo3`, `olmo3.1`, `nemotron-3-nano`, `functiongemma`, `glm-4.7`, `glm-ocr`

**Manual Go parser, but prompt rendering still comes from the model template**:
- `qwen3`, `qwen3-thinking`, `ministral`, `gpt-oss` / `gptoss` (Harmony parser path)

**Template-driven rendering + generic template-aware tool parsing**:
- `llama3.1`, `llama3.2`, `mistral`, `mistral-nemo`, `mistral-small`, `mixtral:8x22b`, `qwen2`, `qwen2.5`, `qwq`, `granite3.3`, `devstral` / `devstral-small-2`

The important Devstral-specific point: **Devstral Small 2 (`mistralai/Devstral-Small-2-24B-Instruct-2512`) is not implemented as a dedicated Go renderer/parser pair in Ollama.** It follows the template-driven path in `server/prompt.go`, with tool extraction delegated to `tools/template.go` + `tools/tools.go`. Even upstream llama.cpp is still wrong relative to Mistral's own tokenizer/normalizer implementation in `mistral-common` (`https://github.com/mistralai/mistral-common`): three mismatches remain — (1) the embedded default prompt text differs from Mistral's official `2512` prompt, (2) Mistral v13 normalizes tool-result messages back into original tool-call order while llama.cpp preserves caller-supplied order, and (3) Mistral supports assistant-prefix / `continue_final_message` semantics that llama.cpp's chat-template input model does not, so the final assistant turn is closed differently in that edge case.

---

## Scorecard: Fork versus Upstream Ollama versus llama.cpp

| Area | Fork | Upstream Ollama @ `9896e36` | llama.cpp @ `a0ed91a` | Winner |
|------|------|----------|--------|--------|
| Qwen 3.5 renderer (tool definitions, ordering, thinking blocks) | Dedicated `Qwen35Renderer` with prefill bug fix + unconditional `<think>` wrapping in history (matches official Jinja2 template — `isThinking` gate removed in commit `4044b63f`) | Dedicated `Qwen35Renderer` without prefill bug fix; `isThinking` gate bug on history think blocks | Executes official Jinja2 template directly — correct for history rendering | **Fork** (prefill fix + history think block fix) |
| Image placement in vision messages | All images prepended to front of message — `Images []ImageData` is a flat array with no positional info | Same — all 3 vision renderers (`Qwen35`, `Qwen3VL`, `LFM2`) prepend | Handles inline image positioning natively via mixed `content` array | **llama.cpp** |
| Qwen 3.5 parser (thinking extraction, tool call delegation) | Dedicated `Qwen35Parser` (byte-for-byte identical to upstream) | Dedicated `Qwen35Parser` | Jinja2 template execution | Tie (Ollama) |
| Prefill guard (`len(message.ToolCalls)==0`) | Fixed in **all 3** renderers (`qwen35.go`, `qwen3coder.go`, `qwen3vl.go`) | Broken in **all 3** renderers | N/A — llama.cpp uses `add_generation_prompt` as a separate parameter, never conflates message closure with generation prompt | **Fork** |
| `</think>` closure in `Qwen3VLRenderer` | Always closes `</think>` tag | Conditionally gates on `content != ""` — leaves unclosed `<think>` tag for thinking-only responses | Always closes via Jinja2 template | **Fork** |
| `mropeInterleaved` default for third-party GGUFs | Architecture-aware default (`c.Architecture() != "qwen3next"` → `true` for `qwen35`) | Hardcoded `false` — **silently wrong RoPE type** for all third-party Qwen 3.5 GGUFs | Hardcoded per architecture in `llama_model_rope_type()`: `LLAMA_ROPE_TYPE_IMROPE` for Qwen 3.5 | **Fork** (both fork and llama.cpp correct; upstream broken) |
| `repeat_last_n` API parameter | Wired through to Go sampler via `NewSampler` parameter | Silently ignored (hardcoded `DefaultPenaltyLookback = 64`) | Correctly wired as `penalty_last_n` | **Fork** |
| Penalty sampling architecture (repeat, presence, frequency — all three share one token window) | Private `recordToken()` inside `Sample()`, ring buffer, excludes prompt tokens — matches original Keskar et al. 2019 CTRL paper. All 3 penalty types benefit. | Public `Accept()` called externally by runner, feeds prompt tokens into penalty window — all 3 penalty types corrupted | Same prompt-contamination bug as upstream Ollama — `server-context.cpp:208-215` feeds all prompt tokens into penalty ring buffer. Masked by `repeat_penalty=1.0` default | **Fork** |
| `repeatPenalty` default | `1.0` (same as upstream — reverted from `1.1` in commit `57e3f80f` to avoid affecting other models; safe to raise per-model via Modelfile because fork's ring buffer excludes prompt tokens) | `1.0` (penalty system is a no-op; llama.cpp comments: `// 1.0 = disabled`) | `1.0` | Tie (but fork's architecture makes it safe to raise; upstream's and llama.cpp's don't) |
| Third-party GGUF compatibility | Architecture-based `mropeInterleaved` default (the critical fix) + shared `ssm_dt.bias` alt tag + `Validate()` + nil checks | Hardcoded `false` mropeInterleaved default (wrong for third-party qwen35 GGUFs) + shared `ssm_dt.bias` alt tag + `Validate()` + nil checks | N/A (llama.cpp uses its own RoPE type system per architecture, not GGUF metadata keys) | **Fork** (due to mropeInterleaved) |
| GGUF converter KV emission | Unconditional (always writes `rope.mrope_interleaved` and `ssm.v_head_reordered`) | Conditional (only writes when `true` — third-party GGUFs that omit the key fall through to wrong default) | N/A (different convention) | **Fork** |
| `SetInplace` → balanced concat tree | Fixed (matches upstream Ollama commit `3490e959`) | Fixed | Uses `ggml_set_inplace()` (works in llama.cpp's GGML but not on Ollama's Metal/Vulkan backends) | Tie (Ollama) |
| Tokenizer performance | Binary search prompt truncation, `strings.Contains` early-out, stack buffer BPE merge | None of these optimizations | N/A (different tokenizer implementation) | **Fork** |
| JSON tool serialization | Fixed: shared tool-definition HTML-escaping end-to-end + Qwen-local `formatToolCallArgument` spaced JSON + `required`/`properties` ordering. Remaining open: `ToolProperty` `enum`/`description` ordering, `ToolProperty` field loss (`nullable`, `additionalProperties`, `prefixItems`), `formatToolCallArgument` scalar boolean/None capitalization (`true`/`false`/`null` instead of `True`/`False`/`None`). For simple tools (no enum, no Optional, no bool args), the fork produces **byte-identical** output to HuggingFace Transformers — verified by saving both outputs and running `diff`. | Full original bug set (HTML escaping, compact separators, key ordering, field loss) | Correct — Jinja2 `tojson` with no HTML escaping, spaced separators, preserved dict insertion order | **Fork** (fork > upstream; llama.cpp > both) |
| `enable_thinking` generation prompt | Correct via `emitEmptyThinkOnNoThink` — prefills `<think>\n\n</think>\n\n` when `think: false` | Same (correct) | **Fixed in post-refactor versions** (`0cd4f47`+): unified `common_chat_template_direct_apply()` passes `enable_thinking` at `chat.cpp:773`, so the template produces the correct 6-token prefill. **Broken in versions up to `c5a7788`**: the old per-model handler `common_chat_params_init_qwen3_coder` at `chat.cpp:1527` did not pass `enable_thinking` to Jinja; post-processing at `chat.cpp:1534-1536` produced `<think>\n</think>` (3 tokens) instead of the official `<think>\n\n</think>\n\n` (6 tokens) — a token sequence never seen in training | **Fork** and **Upstream Ollama** (for older llama.cpp); **Tie** (for post-refactor llama.cpp) |
| History `<think>` blocks gated on thinking mode | **FIXED** (commit `4044b63f`) — unconditional `<think>` wrapping for messages after `lastQueryIndex` | **BROKEN** (gated on `isThinking`) | **Correct** (template handles it) | **Fork** |
| Tool call parser robustness | Free-form streaming parser + **grammar-constrained generation DONE** (commit `4044b63f`): lazy GBNF grammar activates on `<tool_call>`, enforces valid function/parameter names and XML structure. Mode-dependent triggers, dead-end detection, Format→Grammar guard. | Same free-form parser (trusts model output, no grammar constraints) | Grammar-constrained — PEG parser forces model to produce syntactically valid tool calls during generation | **Fork matches llama.cpp** — both use grammar-constrained generation. Fork is the first Ollama model with this. |
| Model-specific sampler defaults | Not implemented | Not implemented | Not implemented (`common.h:186-198` — same defaults for all models) | Tie (all wrong) |
| Float precision in `tojson` | N/A (Go renderer) | N/A (Go renderer) | Minor flaw — 6 significant digits via C++ `ostringstream` versus Python's ~17 (`value.cpp:1290`); also `1.0` → `1` | N/A |

---

## Part 1: Where the Fork Beats Upstream Ollama

### 1.1 CRITICAL: Prefill Bug Fix — Upstream Ollama's Most Dangerous Active Bug

**Fault: Upstream Ollama (`82848a78` through `9896e36`, still unfixed).** All three upstream Ollama renderers — `qwen35.go`, `qwen3coder.go`, `qwen3vl.go` — have this bug. The fork correctly fixed it in commit `fbae6976` by adding `&& len(message.ToolCalls) == 0` to the prefill condition in all three renderers. llama.cpp does not have this bug — it uses `add_generation_prompt` as a separate parameter passed to the Jinja2 template, which never conflates message closure with generation prompt skipping. Prefill is an Ollama-specific concept that llama.cpp does not need.

**Upstream bug locations**: the prefill condition in `Render()` in `qwen35.go`, `qwen3coder.go`, and `qwen3vl.go`
**Fork fix locations**: same — the fork added `&& len(message.ToolCalls) == 0` to the prefill condition in all three renderers

All three upstream Ollama renderers have:
```go
prefill := lastMessage && message.Role == "assistant"
```

The fork adds the guard in all three:
```go
prefill := lastMessage && message.Role == "assistant" && len(message.ToolCalls) == 0
```

This treats any last assistant message as a prefill, **including messages with tool calls**. The consequences differ by renderer:

**In `qwen35.go` and `qwen3vl.go`** — **doubly broken**: The `<|im_end|>` tag is gated by `if !prefill`, so when `prefill=true` on an assistant+toolcalls message, BOTH the `<|im_end|>` is omitted AND the generation prompt (`<|im_start|>assistant\n<think>\n`) is skipped. The rendered prompt ends with `</tool_call>` — no closing tag, no new start. The model's view of the conversation is corrupted.

**In `qwen3coder.go`** — the tool-call branch unconditionally writes `<|im_end|>`, but the post-loop generation prompt is gated by `if lastMessage && !prefill`, which is skipped since `prefill=true`. The model gets no `<|im_start|>assistant\n` to begin generating.

**When does this trigger?** Any time a client sends a chat request where the last message is an assistant message with tool calls populated: a client replays a conversation from saved history that ended at a tool-call boundary, an agent framework constructs the message list with the assistant's tool-call message last, or the fork's own test suite (the `TestQwenToolParser` cases in `qwen3coder_test.go`) proves this is an expected and tested input shape.

The official Qwen 3.5 Jinja2 template emits `{{- '<|im_end|>\n' }}` for every assistant message — this line appears outside any conditional block in the template and runs unconditionally regardless of tool calls, thinking mode, message position, or any other state. The `add_generation_prompt` flag is a separate concern — it controls whether `<|im_start|>assistant\n<think>\n` is appended after all messages, independently of message closures. Ollama's prefill conflates two things (omitting `<|im_end|>` for streaming continuation, and skipping the generation prompt) that the official template keeps strictly independent. The `len(message.ToolCalls) == 0` guard restores the separation: assistant messages with tool calls are complete turns (always closed), while assistant messages without tool calls may be prefills (left open for continuation).

### 1.2 MEDIUM: `</think>` Not Closed When Content Is Empty in `Qwen3VLRenderer`

**Fault: Upstream Ollama (`82848a78` through `9896e36`, still unfixed).** The fork correctly fixed this in commit `fbae6976`. This no longer affects Qwen 3.5 (which uses its own dedicated `Qwen35Renderer`, which always closes `</think>`). But `Qwen3VLRenderer` is still used by `qwen3-vl-instruct` and `qwen3-vl-thinking`, where this bug remains live in upstream Ollama.

**File**: `model/renderers/qwen3vl.go`, in the `Render()` method's assistant message handling

Upstream Ollama conditionally emits `\n</think>\n\n` only when `content != ""`. For thinking-only assistant messages (thinking content but empty visible content), the `<think>` tag is opened but `</think>` is never written. The fork always writes `\n</think>\n\n` before conditionally writing content. llama.cpp always closes `</think>` — the Jinja2 template itself always emits the closing tag.

### 1.3 CRITICAL: `mropeInterleaved` Defaults to `false` — Silent Data Corruption in Upstream Ollama for Third-Party GGUFs

**Fault: Upstream Ollama (`82848a78` through `9896e36`, still unfixed).** The fork (commit `9ec17fc1`) correctly defaults to `c.Architecture() != "qwen3next"` in `New()` in `model/models/qwen3next/model.go`, which evaluates to `true` for `qwen35` — matching llama.cpp's approach. llama.cpp (`d969e933`) **hardcodes** the RoPE type per architecture in `llama_model_rope_type()` at `llama-model.cpp:9109-9113`: `case LLM_ARCH_QWEN35: return LLAMA_ROPE_TYPE_IMROPE` (value 40). This is a **silent data corruption bug** in upstream Ollama.

Upstream Ollama defaults to `false`. The fork defaults based on architecture. Third-party GGUFs (from llama.cpp's converter, Unsloth, bartowski) do not write the ollama-internal `rope.mrope_interleaved` key (it is not in llama.cpp's GGUF constants at `gguf-py/gguf/constants.py`, confirmed in `d969e933`). The actual Qwen 3.5 27B model has `mrope_interleaved: true` (matching llama.cpp's hardcoded `LLAMA_ROPE_TYPE_IMROPE`).

**Concrete impact traced through code**: `mropeInterleaved` controls the RoPE type in the model's RoPE initialization:
- `true` → `rope.WithInterleaveMRoPE()` → sets `opts.Type |= 1<<3 | 1<<5` = **40** = `GGML_ROPE_TYPE_IMROPE`
- `false` → `rope.WithMRoPE()` → sets `opts.Type |= 1<<3` = **8** = `GGML_ROPE_TYPE_MROPE`

With upstream Ollama's `false` default, all 16 full-attention layers in Qwen 3.5 27B use RoPE type 8 (MROPE) instead of 40 (IMROPE). These are completely different dimensional interleaving patterns for position encoding across the time/height/width sections. The model **silently produces wrong output** — it still runs, but attention patterns are misaligned across every full-attention layer. Users would blame the model quality, not the inference engine.

Note: `ssm.v_head_reordered` defaults are functionally equivalent between fork (`c.Architecture() != "qwen3next"`) and upstream Ollama (`defaultVHeadReordered` which positively matches `"qwen35" || "qwen35moe"`). Both produce `true` for `qwen35`/`qwen35moe` and `false` for `qwen3next`.

### 1.4 Penalty Sampling Architecture — The Fork Matches the Original Academic Paper

**Fault: Both upstream Ollama and llama.cpp. The fork is the only correct implementation per the original academic formulation.**

The repetition penalty was introduced by Keskar et al. (2019) in the CTRL paper (arXiv:1909.05858). The paper's formula operates over *"generated tokens g"* — the set of tokens the model has produced during inference. Prompt tokens are not in `g`.

**Upstream Ollama** feeds prompt tokens into the penalty history via the public `Accept()` method called from the runner at **two call sites**:
1. **Sequence load** (in `processSequence()` in `runner.go`): `seq.sampler.Reset()` then loops through ALL cached inputs calling `seq.sampler.Accept(inp.Token)`. The last 64 prompt tokens become part of the penalty window.
2. **Batch processing** (in `processBatch()` in `runner.go`): As pending input tokens are committed to the cache, each token is `Accept()`-ed. Both prompt tokens and generated tokens flow through this path.

**llama.cpp** (`d969e933` through `a0ed91a`) has the same behavior. The server's `init_sampler()` at `server-context.cpp:208-212` loops through ALL prompt tokens and calls `common_sampler_accept(smpl.get(), id, false)` on each one. The penalty sampler at `llama-sampler.cpp:2622-2656` uses a ring buffer that makes no distinction between prompt-origin and generation-origin tokens. The penalty sampler in `llama-sampler.cpp` is **byte-identical** between `d969e933` and the latest `a0ed91a`. Defaults unchanged: `penalty_repeat=1.0` (disabled), `penalty_last_n=64`.

This is the same mistake HuggingFace `transformers` made (and [acknowledged as a bug](https://github.com/huggingface/transformers/issues/36642) — users reported *"weird strings"* and false penalization of common phrases, fixed in [PR #37625](https://github.com/huggingface/transformers/pull/37625), April 2025). llama.cpp users reported the [same problem](https://github.com/ggml-org/llama.cpp/issues/331).

**The fork's architecture** (in `sample/samplers.go`): The `Sampler` struct contains `recentTokens []int32` (ring buffer), `recentTokenPos int` (write cursor), and `repeatLastN int` (window size from API, user-configurable). The private `recordToken()` method implements a classic circular buffer: O(1) per token, zero allocations after warmup, permanently bounded at `repeatLastN` entries. The runner's only interaction is `seq.sampler.Sample(logits)` — it never touches internal state. No external code can feed prompt tokens into the penalty window.

**llama.cpp's ring buffer architecture** (`d969e933`, `llama-sampler.cpp:23-132`): Uses a proper `ring_buffer<llama_token>` template class with the same design — fixed capacity, O(1) push, modular arithmetic. The key difference: llama.cpp's `llama_sampler_penalties_accept()` is a **public** function called by external code (the server feeds prompt tokens through it), while the fork's `recordToken()` is **private** and called only from `Sample()`.

**Upstream Ollama's history architecture** (in `sample/samplers.go`): The `Sampler` struct contains `history []int32` (an append-then-truncate slice). `Accept()` appends the token, then if `len(history) > DefaultPenaltyLookback`, copies the last 64 entries to the front and truncates — a **slide-and-truncate** pattern (O(n) per trim, allocates on growth). `Reset()` sets history to nil. The runner calls these externally at 3 separate call sites.

**All three penalty types share the same token window — not just `repeatPenalty`.** The `applyPenalties()` function takes a single `recentTokens []int32` slice and applies ALL three penalty types in a single pass:

| Penalty type | Math per token in window | Effect |
|---|---|---|
| `repeatPenalty` | Multiplicative: `logit /= penalty` (positive) or `logit *= penalty` (negative) | Scales down repeated tokens proportionally |
| `frequencyPenalty` | Subtractive: `logit -= frequencyPenalty * count` | Penalizes proportionally to occurrence count |
| `presencePenalty` | Subtractive: `logit -= presencePenalty` | Flat penalty for ANY token that appeared at least once |

**This is especially critical for the actual Modelfile configuration.** The custom Modelfile for Qwen 3.5 27B (`qwen3.5-custom.Modelfile`) uses:
```
PARAMETER repeat_penalty 1.0    # disabled
PARAMETER presence_penalty 1.5  # ACTIVE — the primary anti-repetition mechanism
```

The user disables `repeat_penalty` (1.0 = identity) and relies entirely on `presence_penalty 1.5` to prevent degenerate repetition. With the fork's architecture, this correctly penalizes self-repetition without touching prompt tokens. With upstream Ollama's architecture, `presence_penalty 1.5` would subtract 1.5 from the logit of every token in the last 64 tokens of history — including prompt tokens like tool names, JSON structural tokens, and parameter names. **A 1.5 logit subtraction is severe** (roughly halving the probability at typical logit scales), and applying it to prompt tokens would catastrophically degrade tool calling accuracy.

**Cache shift behavior**: On normal cache shift, neither codebase touches the sampler. On the reprocess error path, upstream Ollama calls `seq.sampler.Reset()`, clearing the entire history — this is wrong because it destroys repetition penalty information at the exact moment it matters most. The fork does nothing to the ring buffer on either path, which is correct: cache shift is a memory management operation, not a semantic boundary.

One remaining nuance: the fork **does** penalize the model's own thinking tokens (they're generated), so if `<think>I need to call get_weather</think>` is shorter than 64 tokens, the `get_weather` tokens are still in the penalty window when the model emits the actual tool call. At `1.1` this is mild (~9% logit reduction), but users who raise the penalty to `1.5+` would break tool calling even with the fork's architecture.

### 1.5 `repeat_last_n` API Parameter — Silently Ignored in Upstream Ollama

**Fault: Upstream Ollama.** The Go runner's `NewSampler` simply doesn't accept the parameter — the API field exists, the default exists (`RepeatLastN: 64` in `api/types.go`), but nothing connects it to the sampler. The fork (commit `ab234955`) fixed this by adding `repeatLastN` as a parameter to `NewSampler` and wiring `req.Options.RepeatLastN` through in `runner.go`. llama.cpp correctly wires `penalty_last_n`.

**Caveat**: The value IS respected in the legacy C++ `llamarunner` (in `runner/llamarunner/runner.go`), which passes it to llama.cpp's C sampler. So this is a Go-runner-only regression, not a universal API lie. But the Go runner is the default for all new model architectures including `qwen3next`.

### 1.6 `repeatPenalty` Default — Reverted From 1.1 to 1.0 (Commit `57e3f80f`)

The fork originally changed the default from `1.0` to `1.1` to make the penalty system functional out of the box. This was **reverted to `1.0`** in commit `57e3f80f` to avoid affecting non-Qwen models. The fork's ring-buffer architecture makes it safe to raise the penalty per-model via Modelfile without the prompt-token contamination that makes this dangerous in upstream Ollama and llama.cpp.

The exact penalty math in all three codebases (fork's `applyPenalties()` in `transforms.go`, upstream Ollama's equivalent, llama.cpp's `llama-sampler.cpp`) is identical: `if logit > 0: logit /= repeatPenalty`, `if logit < 0: logit *= repeatPenalty`.

### 1.7 GGUF Converter Unconditional KV Emission

**Fork converter** (`convert/convert_qwen3next.go`): Always writes both keys unconditionally:
```go
kv["rope.mrope_interleaved"] = q.RopeParameters.MRopeInterleaved
kv["ssm.v_head_reordered"] = q.shouldReorderVHeads()
```

**Upstream Ollama converter**: Only writes when `true`. For `rope.mrope_interleaved`, upstream's conditional emission is fine for Ollama-converted GGUFs (the converter writes `true` for `qwen35`), but third-party GGUFs that omit the key fall through to upstream's `false` default (wrong for `qwen35` — see Section 1.3).

Note: `ssm.v_head_reordered` and `rope.mrope_interleaved` are **ollama-internal conventions**. They do not exist in llama.cpp's GGUF constants. llama.cpp sidesteps the problem entirely: RoPE type is hardcoded per architecture in `llama_model_rope_type()`, and V-head reordering is done **physically during GGUF conversion** by `_LinearAttentionVReorderBase` (`convert_hf_to_gguf.py:4782-4791`) — the converter permutes tensor weights so no runtime flag is needed. Third-party GGUFs from Unsloth, bartowski, or llama.cpp's converter will **never** contain these keys.

### 1.8 Third-Party GGUF Compatibility Fixes

- **`ssm_dt.bias` tensor name** (in `deltanet.go`): `gguf:"ssm_dt,alt:ssm_dt.bias"`. Both fork and upstream have this — the alt tag allows loading GGUFs where the llama.cpp converter wrote `ssm_dt.bias` instead of `ssm_dt.weight`. **Shared fix.**
- **Recurrent layer classification**: The fork uses the interval formula directly (in `New()` in `model.go`), bypassing the `headCountKV` array entirely. Upstream's `inferRecurrentLayers()` handles scalar broadcasts via a fallback path that reaches the same formula. Both produce correct results for all real-world GGUFs, but via different mechanisms.
- **Architecture-based defaults for `mropeInterleaved`**: Fork-only fix — this is the critical one (see Section 1.3).
- **Architecture-based defaults for `vHeadReordered`**: Functionally equivalent — both produce `true` for qwen35.
- Both codebases have: `Validate()` method (in `model.go`) catching missing tensors at load time, and inline nil checks in `GatedDeltaNet.Forward()` for `SSMDT`, `SSMA`, `SSMConv1D`, `SSMNorm`, and `SSMOut`.

### 1.9 Tokenizer Performance Optimizations

- **Binary search prompt truncation** (`server/prompt.go`): O(log N) tokenize calls instead of linear scan.
- **Special token splitting** (`tokenizer/special.go`): `strings.Contains` early-out before regex matching. 13 test cases validate equivalence.
- **Stack buffer BPE merge** (`tokenizer/vocabulary.go`): 128-byte stack buffer eliminates heap allocations in BPE merge hot path. Leverages Go 1.12+ map-lookup compiler optimization.

---

## Part 2: Qwen 3.5 Renderer and Parser — Dedicated Implementation

The fork now has a dedicated `Qwen35Renderer` (`model/renderers/qwen35.go`, 195 lines) and `Qwen35Parser` (`model/parsers/qwen35.go`, 239 lines), ported from upstream Ollama's `82848a78` implementation with four fixes applied. The `Qwen35Parser` is byte-for-byte identical to upstream Ollama. The `Qwen35Renderer` differs from upstream in four places:
1. **Prefill bug fix** (in the prefill condition in `Render()`): adds `&& len(message.ToolCalls) == 0` — see Section 1.1
2. **History think block fix** (in the `<think>` wrapping condition in `Render()`): removes `isThinking &&` — unconditional `<think>` wrapping matching official Jinja2 template — see Section 2.5
3. **Reasoning extraction fix** (in `splitQwen35ReasoningContent`): no longer takes `isThinking` parameter — always uses `messageThinking` when available — see Section 2.5
4. **Multiple `</think>` tag fix** (in `splitQwen35ReasoningContent`): uses `strings.LastIndex` for remaining content extraction instead of `strings.Index`, matching the official template's `content.split('</think>')[-1]` semantics — remaining content comes from after the LAST `</think>`, not the first

The previous approach of routing `"qwen3.5"` through `Qwen3CoderRenderer`/`Qwen3CoderParser` caused seven distinct template fidelity problems (sections 2.1–2.7 below), **all now resolved**. All existing tests pass, and 17 new tests (5 renderer + 11 parser + 1 integration routing test in `qwen3_test.go:211`) cover the new code.

Note: the fork's `Qwen3CoderRenderer` AND `Qwen3CoderParser` already differ from upstream's versions — both have thinking support that upstream lacks:
- **Fork's `Qwen3CoderRenderer`** (`qwen3coder.go`): has `isThinking` and `emitEmptyThinkOnNoThink` fields, thinking block rendering, think prefill, and the prefill bug fix — all in the `Render()` method.
- **Fork's `Qwen3CoderParser`** (`qwen3coder.go`): has `hasThinkingSupport` and `defaultThinking` fields, 4 parser states (including `CollectingThinking` and `ThinkingDoneTransition`), and `HasThinkingSupport()` that returns the field value. Upstream has only 2 parser states and `HasThinkingSupport()` returns `false` unconditionally.

### 2.1 FIXED: Tool Definition Format — JSON Instead of XML

The dedicated `Qwen35Renderer` uses `marshalWithSpaces()` for JSON tool definitions, matching the official Qwen 3.5 Jinja2 template's `{{ tool | tojson }}`. The Qwen 3.5 / Qwen3-Coder hybrid format split (verified against official HuggingFace templates):

| | Tool definitions (system prompt) | Tool calls (assistant output) | System+tools order |
|---|---|---|---|
| **Qwen3-Coder** | **XML**: `<function><name>get_weather</name>...` | **XML**: `<tool_call><function=get_weather>...` | System first, tools second |
| **Qwen 3.5** | **JSON**: `{{ tool | tojson }}` | **XML**: identical to Qwen3-Coder | Tools first, system appended after |

### 2.2 FIXED: System+Tools Ordering Reversed for Qwen 3.5

The `Qwen35Renderer` emits the tools block first, then appends system message content after `</IMPORTANT>`, matching the official template. This matters for attention patterns.

### 2.3 FIXED: Image/Vision Support for Qwen 3.5

The `Qwen35Renderer` has its own `renderContent()` method with `useImgTags` field. Qwen 3.5 27B is a **native multimodal early-fusion model** — it has a 27-layer vision encoder (1152 hidden size, 16 heads, patch size 16) built into its architecture. The official template handles image items with `<|vision_start|><|image_pad|><|vision_end|>` and video items with `<|vision_start|><|video_pad|><|vision_end|>`. Both upstream Ollama and the fork support images but not videos.

**CAVEAT — image placement is wrong in all Ollama vision models.** Every image in `renderContent()` is prepended before `content.Content`. The `api.Message` struct stores images as a flat `Images []ImageData` array with zero positional information. All three vision-capable renderers in the entire Ollama codebase (`Qwen35Renderer`, `Qwen3VLRenderer`, `LFM2Renderer`) have this same front-prepend behavior. The official Jinja2 template handles inline image positioning natively because the HuggingFace `messages` format embeds images as interleaved content items. Fixing this requires changing the `api.Message` struct — Ollama's public HTTP API — to support interleaved content items, then updating all renderers, the runner's `[img-N]` regex splitter, and every client.

For text-only Unsloth GGUFs (851 tensors, no `vision.block_count`), the model layer correctly sets `Vision = nil` and returns `ErrNoVisionModel` if images are sent.

### 2.4 FIXED: Default System Message Injection Removed

The `Qwen3CoderRenderer` injects `"You are Qwen, a helpful AI assistant..."` when tools are present but no system message is given (correct for Qwen3-Coder). The `Qwen35Renderer` has no default system message injection (correct for Qwen 3.5).

### 2.5 FULLY FIXED: History `<think>` Blocks — Unconditional Wrapping Matches Official Template (Commit `4044b63f`)

Both the fork (before `4044b63f`) and upstream Ollama gate the rendering of `<think>` blocks in assistant message history on `isThinking` (controlled by the API's `think: true/false` parameter). This is a two-part bug.

**Part 1 — Rendering condition (the `<think>` wrapping in `Render()` in `qwen35.go`):**
The official template's `enable_thinking` parameter is checked in exactly one place — the `add_generation_prompt` block at the end of the template (see Section 3.1). It does not appear anywhere in the message rendering loop. The rendering condition for `<think>` blocks uses only `{%- if loop.index0 > ns.last_query_index %}` — no `enable_thinking` check. Assistant messages after `last_query_index` ALWAYS include `<think>` blocks regardless of whether thinking is enabled for the current turn. This also applies to messages with empty reasoning: when `reasoning_content` is an empty string, the template produces `<think>\n\n</think>\n\n` (an empty think block), not a bare message without think tags.

**Part 2 — Reasoning extraction (`splitQwen35ReasoningContent` in `qwen35.go`):**
The official template always uses `reasoning_content` when it exists as a string (`message.reasoning_content is string`) — no `enable_thinking` check. In Jinja2, an empty string `""` passes the `is string` test, so path 1 is taken even for messages with no reasoning. When `isThinking=false` and `messageThinking` is non-empty, the old code ignores the stored thinking field and falls through to content parsing, which typically finds nothing. The reasoning is silently lost.

**The fix** (two changes in `model/renderers/qwen35.go`):
1. `splitQwen35ReasoningContent(content, messageThinking string)` — removed `isThinking` parameter
2. `if i > lastQueryIndex` — removed `isThinking &&` prefix from the `<think>` wrapping condition

**When this bug triggers:** Multi-turn conversations where thinking mode switches between turns. Previous assistant responses had thinking content stored in `message.Thinking`. With `think: false`, both the extraction and the rendering gate fail, and the model sees its own previous responses with reasoning stripped out — a prompt format it was never trained on.

**Upstream Ollama still has this bug in both places.**

**llama.cpp comparison:** llama.cpp does NOT have this bug for history rendering — it executes the official Jinja2 template directly. However, llama.cpp versions up to `c5a7788` had their own generation prompt bug where `enable_thinking` was not passed to the Jinja context (see Section 7.1 Bug 1). This was fixed in the post-refactor versions (`0cd4f47`+) which replaced the per-model specialized handlers with a unified autoparser that correctly passes `enable_thinking`.

**Note on Qwen3-Next-Thinking**: The `Qwen3CoderRenderer`'s conditional `message.Thinking != ""` gating is actually **correct for Qwen3-Next-Thinking** models, whose official template only emits think blocks for non-last messages when `reasoning_content` is non-empty. The two model families were trained on contradictory thinking block semantics. Having separate renderers solves this cleanly.

### 2.6 FIXED: `lastQueryIndex` Doesn't Check for `<tool_response>` Wrappers

The `Qwen35Renderer` implements the `multiStepTool` / `<tool_response>` filtering pattern (the `lastQueryIndex` backward scan in `Render()`) matching the official template exactly: walks messages backwards, skips user messages that are entirely `<tool_response>...</tool_response>` wrapped, finds the first non-tool-response user message as `last_query_index`.

### 2.7 FIXED: Prefill Triggers on Assistant Messages With Tool Calls

See Section 1.1 for the full analysis. The `Qwen35Renderer` has the fix in the prefill condition in `Render()`.

### What the Dedicated `Qwen35Parser` Provides

The `Qwen35Parser` has a 3-state thinking machine (`CollectingThinking` → `ThinkingDoneEatingWhitespace` → `CollectingContent`) that extracts thinking content before delegating post-`</think>` content to an embedded `Qwen3CoderParser`. Key properties:

- **Tool calls inside `<think>` blocks are treated as thinking text**, not parsed as tool calls.
- **Leading `<think>` tag stripped at most once** via `maybeConsumeLeadingThinkOpenTag()`.
- **Streaming-safe partial tag overlap detection** with trailing whitespace withholding.
- **`HasToolSupport()` returns `true`, `HasThinkingSupport()` returns `true`**.
- **Parser `done` flag:** The `done` flag NOT checked by `parseEvents()`/`eat()` methods. Trailing whitespace and partial tag overlaps are silently dropped on stream end. This is minor and systemic across all Ollama parsers.
- The parser is **byte-for-byte identical** to upstream Ollama's implementation. 12 test functions cover all edge cases.

---

## Part 3: Thinking and Non-Thinking Mode Deep Analysis

### 3.1 The Official Template: One Template, Two Modes

The official Qwen 3.5 Jinja2 template (identical across `Qwen/Qwen3.5-27B` and `Qwen/Qwen3.5-35B-A3B` on HuggingFace) receives `enable_thinking` as a per-request template parameter. It is checked in **exactly one place** in the entire template — the `add_generation_prompt` block at the very end:

| `enable_thinking` | Generation Prompt Prefill | Model Behavior |
|-------------------|-------------------------|----------------|
| `true` (or undefined — **default**) | `<|im_start|>assistant\n<think>\n` | Model generates reasoning inside `<think>...</think>`, then content |
| `false` | `<|im_start|>assistant\n<think>\n\n</think>\n\n` | Empty think block → model generates content directly |

`enable_thinking` does **not** appear anywhere in the message rendering loop. Historical assistant messages are **never gated** by `enable_thinking`. They receive `<think>` block wrapping based solely on their position relative to `last_query_index` — if a message appears after the last real user query, it gets `<think>\nreasoning\n</think>\n\n` wrapping unconditionally, even if `enable_thinking` is `false`, and even if the reasoning content is empty (producing `<think>\n\n</think>\n\n`). The `<|im_end|>` tag that closes each assistant message is also emitted unconditionally — it appears outside any conditional block and runs for every assistant message regardless of thinking mode, tool calls, or message position.

There are **no `/think` or `/no_think` tokens**, no special token IDs for mode switching. The `<think>` and `</think>` are token IDs 248068-248069 in the vocabulary, marked as `"special": false` — regular tokens. The mode is determined entirely by the prefill.

Reasoning content is extracted from each historical assistant message via two paths in the official template: (1) an explicit `reasoning_content` field on the message (checked via `message.reasoning_content is string`), or (2) a fallback that parses inline `<think>...</think>` tags from the message content. Path 1 takes priority. Note that in Jinja2, an empty string `""` passes the `is string` test, so a message with `reasoning_content: ""` takes path 1 (using the empty string directly) and does NOT fall through to content parsing.

### 3.2 The `emitEmptyThinkOnNoThink` Field — Correct Implementation

The `Qwen35Renderer` is constructed with `emitEmptyThinkOnNoThink: true` in `RendererForName()` in `renderer.go`. This correctly matches the official template:
- `think: true` → `<think>\n` (model generates reasoning)
- `think: false` → `<think>\n\n</think>\n\n` (empty think block, model generates content)
- Without `emitEmptyThinkOnNoThink`: nothing emitted — would be WRONG

The field exists because some model families (e.g., Qwen3-VL-Instruct, constructed with `isThinking: false` and no `emitEmptyThinkOnNoThink`) were never trained with `<think>` blocks at all.

### 3.3 Parser Behavior — Correct for Both Modes

The `Qwen35Parser.Init()` correctly starts in `CollectingThinking` state when `think: true` (or nil) and `CollectingContent` state when `think: false`. This matches the renderer behavior.

### 3.4 Thinking Architectures Across Qwen Model Families

| Model Family | `isThinking` Default | `emitEmptyThinkOnNoThink` | History Think Blocks | Notes |
|---|---|---|---|---|
| **Qwen 3.5** (`Qwen35Renderer`) | `true` | `true` | Unconditional (per official template — fork fixed) | Correct |
| **Qwen3-Coder** (`Qwen3CoderRenderer`) | `false` | `false` | Conditional on `message.Thinking != ""` && `isThinking` | Correct for Qwen3-Coder's template. Fork added thinking support upstream lacks |
| **Qwen3-VL-Thinking** (`Qwen3VLRenderer`) | `true` | `false` (**NOT set!**) | Gates on `isThinking` | **Potentially wrong** — when `think: false`, NO think block is prefilled. Needs verification against official template |
| **Qwen3-VL-Instruct** (`Qwen3VLRenderer`) | `false` | N/A | No think blocks | Correct for non-thinking variant |

### 3.5 Fork's Qwen3CoderRenderer Thinking Asymmetry

The fork added thinking support to `Qwen3CoderRenderer` (in commit `fbae6976`) that upstream lacks. The fork supports thinking for `qwen3-coder` models if the `think: true` API parameter is passed. Upstream silently ignores `think: true` for qwen3-coder. Since Qwen3-Coder models don't have an official `enable_thinking` mechanism in their Jinja2 template, this is an extension beyond the training data.

---

## Part 3B: Client Input Validation

The fork (and upstream Ollama) perform no validation of conversation structure before rendering. Clients can send invalid conversations — unknown roles, misplaced system messages, empty tool function names, images in system messages — and the renderer silently produces training-absent prompts. The official Qwen 3.5 Jinja2 template defends against this with 8 `raise_exception()` calls; the fork has none.

This topic is covered in full in the dedicated report: **[Client Input Validation Report](./input_validation_report.md)**. That report includes:

- The complete request lifecycle (5 stages from HTTP endpoint to grammar pipeline) with what validation exists vs. what's missing at each stage
- Concrete prompt traces showing the exact malformed output for 6 categories of invalid client input
- Side-by-side comparison with llama.cpp's validation (including the corrected HTTP status code mapping — `raise_exception()` maps to HTTP 400 via `std::invalid_argument` wrapping at `chat.cpp:1532`, not HTTP 500)
- The architectural solution: 5 one-liner validation checks placed inside `Render()` at the exact code points where the relevant state is already computed, eliminating all code duplication
- A complete validation checklist with what to add, what NOT to add (and why), and where NOT to add it

---

## Part 4: JSON Serialization Mismatches in Tool Definitions and Tool Call Arguments

The fork (and upstream Ollama) serialize tool definitions differently from the Qwen 3.5 model's training data. The official Qwen 3.5 Jinja2 template uses the `tojson` Jinja2 filter. In HuggingFace Transformers (`transformers` @ `3a3b59c`), this filter is overridden (at `chat_template_utils.py:465-468`) to use `json.dumps` with default separators (`', '` and `': '`), no HTML escaping, no key sorting, and insertion order preservation. This is registered at line 476: `jinja_env.filters["tojson"] = tojson`. When the Qwen 3.5 template calls `{{ tool | tojson }}`, all arguments use defaults: `ensure_ascii=False`, `sort_keys=False`, `separators=None` (which in Python 3 defaults to `(', ', ': ')`).

**Important distinction: This is NOT stock Jinja2's `tojson`.** Stock Jinja2 (`jinja2/utils.py:htmlsafe_json_dumps`) uses `json.dumps` with `sort_keys=True`, then runs `.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026').replace("'", '\\u0027')` — alphabetical key sorting plus HTML escaping. HuggingFace's override does neither. The model was trained with HuggingFace's override, so that is the ground truth.

**How we verified this**: We ran `apply_chat_template` with `transformers` v5.2.0 against the official `Qwen/Qwen3.5-27B` tokenizer (downloaded from HuggingFace) and compared the rendered output byte-for-byte against what the fork's `Qwen35Renderer` produces for the same tool and conversation. We also ran Go programs using Ollama's exact `api.Tool` struct types to verify the actual serialization behavior. Every claim below is backed by empirical output.

llama.cpp avoids these bugs because it executes the Jinja2 template directly via its vendored Jinja engine, producing the same JSON the model was trained on.

> **CRITICAL IMPLEMENTATION NOTE:** These sub-bugs range from "straightforward" (Sub-bug C, Sub-bug D) to "shared infrastructure change requiring per-model verification" (Sub-bug B's key ordering and field loss). The core concern is that `ToolProperty` and `ToolFunctionParameters` are **shared types** used by many renderers (DeepSeek, GLM, Cogito, Olmo3, Nemotron, etc.), not just the Qwen family. However, per-model verification of all accessible templates shows all use HuggingFace Transformers' insertion-order override — no verified model uses stock Jinja2's `sort_keys=True`.

### 4.1 Sub-Bug A — HTML Character Escaping — DONE (2026-03-11)

Go's `json.Marshal` HTML-escapes `<`, `>`, `&` to `\u003c`, `\u003e`, `\u0026`. The official template's `tojson` outputs these characters literally. A tool description like `"Returns temperature in <fahrenheit> & <celsius>"` becomes completely different tokens when HTML-escaped (379 bytes versus 404 bytes for the full tool JSON).

The fork now fixes the **full tool-definition serialization path** by switching the top-level helper (`marshalWithSpaces` uses `jsonutil.Marshal` which calls `SetEscapeHTML(false)`) and the nested tool-type `MarshalJSON` boundaries in `api/types.go` / `internal/orderedmap` onto a no-HTML-escape JSON path. `formatToolCallArgument` in `qwen3coder.go` was also fixed for the Qwen-local tool-call-argument path.

**Completeness note:** The functions `renderAdditionalKeys()` and `formatToolDefinitionType()` in `qwen3coder.go` still use standard `json.Marshal` with HTML escaping, but these functions are **never called by `Qwen35Renderer`** — they only affect qwen3-coder models. For Qwen 3.5, HTML escaping is fully fixed.

*Shared infrastructure concern (verified):* `marshalWithSpaces` is called by the Olmo3, Qwen 3.5, and Qwen3VL renderers. Official HuggingFace templates for `Qwen/Qwen3.5-*`, `Qwen/Qwen3-VL-*`, `allenai/Olmo-3-7B-Instruct`, and `allenai/Olmo-3.1-32B-Instruct` all use `tools | tojson` for tool definitions, so disabling HTML escaping is correct for those paths.

### 4.2 Sub-Bug B — Key Ordering and Field Loss in Tool Definition Serialization — PARTIALLY DONE (`required`/`properties` fixed; `enum`/`description` ordering, field loss for `nullable`/`additionalProperties`/`prefixItems`, and scalar boolean/None tool-call-argument capitalization remain OPEN)

Go's `json.Marshal` outputs struct fields in declaration order and sorts `map[string]any` keys alphabetically. Additionally, Go's `ToolProperty` struct lacks three fields that HuggingFace's `_parse_type_hint()` can produce — `nullable`, `additionalProperties`, and `prefixItems` — which are silently dropped during JSON deserialization. Both issues were empirically verified by running Go's `json.Marshal` on the actual Ollama `api.Tool` struct types and comparing against HuggingFace's `get_json_schema()` output (traced through `chat_template_utils.py` source code — `_parse_type_hint()` constructs dicts with specific insertion order, then `get_json_schema()` appends fields like `description` and `enum` in a specific sequence, and Python's `json.dumps` with `sort_keys=False` preserves that insertion order):

| Nesting level | Go struct field order | HuggingFace `get_json_schema()` insertion order | Match? |
|---|---|---|---|
| `Tool` | `type`, [`items`†], `function` | `type`, `function` | **Yes**† |
| `ToolFunction` | `name`, `description`, `parameters` | `name`, `description`, `parameters` | **Yes** |
| `ToolFunctionParameters` | `type`, [`$defs`†], [`items`†], `properties`, [`required`‡] | `type`, `properties`, [`required`‡] | **Yes** — FIXED (struct reordered, `Properties` now declared before `Required`) |
| `ToolProperty` (with enum) | `type`, `description`, `enum` | `type`, `enum`, `description` | **No — `enum`/`description` swapped** |
| `ToolProperty` (no enum) | `type`, `description` | `type`, `description` | **Yes** |
| `ToolProperty` (`Optional[T]`) | `type`, `description` | `type`, `nullable`, `description` | **No — `nullable` field lost** |
| `ToolProperty` (`Dict[str, T]`) | `type`, `description` | `type`, `additionalProperties`, `description` | **No — `additionalProperties` field lost** |
| `ToolProperty` (`Tuple[T1, T2]`) | `type`, `description` | `type`, `prefixItems`, `description` | **No — `prefixItems` field lost** |
| `ToolProperty.Items` (as `map[string]any`) | alphabetical (e.g., `description`, `type`) | single key only: `{"type": "..."}` | **Yes** — items are single-key, no ordering issue |
| `ToolPropertiesMap` (property names) | insertion order (orderedmap) | insertion order | **Yes** |

†`Tool.Items`, `ToolFunctionParameters.$defs`, and `ToolFunctionParameters.Items` (all in `api/types.go`) are `omitempty` fields not produced by HuggingFace's `get_json_schema()`. When nil (the common case), they are hidden.

‡`ToolFunctionParameters.Required` (in `api/types.go`) also has `omitempty`, correctly matching Python's conditional `if required: schema["required"] = required` guard in `chat_template_utils.py` — when no parameters are required, both Go and Python omit the field entirely. The ordering deviation (before `properties` in Go, after in Python) applies only when `required` is present, which is the common case for tool-calling models. **Crucially, this fix is safe to apply globally (by reordering the Go struct) without per-model verification:** since `p` < `r` alphabetically, even stock Jinja2's `sort_keys=True` produces `properties` before `required`. Every possible Python/Jinja2 configuration — HuggingFace Transformers' insertion-order-preserving override, stock Jinja2's alphabetical sorting, and llama.cpp's insertion-order-preserving engine — produces `properties` before `required`. The Go struct's current declaration order is the only system that produces the reversed ordering. Per-model template verification confirmed this: Qwen 3.5, Qwen3VL, OLMo3, OLMo3.1, and the GLM-4 family all use `tojson` with insertion order preserved (GLM-4 templates are embedded in `tokenizer_config.json` on HuggingFace — see GLM verification note below); DeepSeek V3's template doesn't render tool definitions at all (Ollama constructs them); gated models (Cogito, LFM2, GLM-OCR) are safe regardless because both possible Python orderings agree.

The `ToolProperty` enum/description mismatch arises because `_parse_type_hint()` at `chat_template_utils.py:136-139` constructs `{"type": ..., "enum": [...]}` first, then `get_json_schema()` at line 377 appends `schema["description"] = desc` last. Python dicts preserve insertion order, so the training data has `enum` before `description`. Go's `ToolProperty` struct declares `Description` before `Enum`, producing the opposite order. (The same ordering holds when `enum` comes from a docstring's `(choices: ...)` block instead of a `Literal` type hint — `get_json_schema()` at line 375 inserts `schema["enum"]` before line 377 inserts `schema["description"]`, producing the same `enum`-before-`description` order.)

**Global safety assessment for this struct reorder:** Since `d` < `e` alphabetically, stock Jinja2's `sort_keys=True` would produce `description` before `enum` — the same order as Go's current struct. HuggingFace Transformers' insertion-order-preserving override produces the opposite (`enum` before `description`). These two Python/Jinja2 configurations disagree in theory, so the correct order depends on which Jinja2 implementation each model's training pipeline used. However, every model whose HuggingFace template has been verified — Qwen 3.5, Qwen3VL, OLMo3, OLMo3.1, and the GLM-4 family (see GLM verification note below) — was trained with HuggingFace Transformers' insertion-order override. No verified model uses stock Jinja2's `sort_keys=True`. The safest approach is to swap `Enum` and `Description` in the `ToolProperty` struct globally (changing field order from `{AnyOf, Type, Items, Description, Enum, Properties}` to `{AnyOf, Type, Items, Enum, Description, Properties}`), which matches all verified models. Alternatively, implement renderer-local custom marshaling for `ToolProperty` in specific renderers to avoid any risk to unverified models.

The field loss rows (`Optional[T]`, `Dict[str, T]`, `Tuple[T1, T2]`) arise because Go's `ToolProperty` struct (in `api/types.go`) has six fields (`anyOf`, `type`, `items`, `description`, `enum`, `properties`) but HuggingFace Transformers' `_parse_type_hint()` can produce three additional fields: `nullable` (line 127, for `Optional[T]` / `Union[T, None]`), `additionalProperties` (line 175, for `Dict[str, T]`), and `prefixItems` (line 168, for `Tuple[T1, T2]`). Go's `json.Unmarshal` silently drops unknown fields, so these constraints vanish during the client→Go→renderer round-trip. The model then sees a tool schema missing type constraints it was trained on — `{"type": "string"}` instead of `{"type": "string", "nullable": true}`, or `{"type": "object"}` instead of `{"type": "object", "additionalProperties": {"type": "integer"}}`. Of the three, `nullable` is the most impactful — `Optional[T]` parameters are common in Python tool definitions.

The fix requires rewriting the `ToolProperty` struct to match the exact field ordering that HuggingFace Transformers' `get_json_schema()` produces. This struct was verified against every type combination in HuggingFace Transformers v5.3.0 — `_parse_type_hint()` builds the base dict with `type` first, then adds type-specific fields (`items`/`additionalProperties`/`prefixItems`) and `nullable` in insertion order, then `get_json_schema()` appends `enum` (from choices annotations) and `description` last:

```go
// CORRECT ToolProperty struct (matches HF get_json_schema insertion order)
type ToolProperty struct {
    AnyOf                []ToolProperty     `json:"anyOf,omitempty"`
    Type                 PropertyType       `json:"type,omitempty"`
    Items                any                `json:"items,omitempty"`
    AdditionalProperties any               `json:"additionalProperties,omitempty"`
    PrefixItems          []any              `json:"prefixItems,omitempty"`
    Nullable             *bool              `json:"nullable,omitempty"`
    Enum                 []any              `json:"enum,omitempty"`
    Description          string             `json:"description,omitempty"`
    Properties           *ToolPropertiesMap `json:"properties,omitempty"`
}
```

Verified examples: `Optional[list[str]]` → `type, items, nullable, description` ✓; `Optional[str]` with `(choices: ...)` → `type, nullable, enum, description` ✓; `str` with `(choices: ...)` → `type, enum, description` ✓.

The `Items` alphabetization is a non-issue for training data. `get_json_schema()` never adds `description` to items sub-objects — items are single-key dicts like `{"type": "string"}`, which have no ordering to break. Only client-provided multi-key items schemas would be affected.

**Concrete byte comparison** (empirically verified with Go and Python producing the same tool definition):
- **Python (training data):** `..., "format": {"type": "string", "enum": ["json", "xml"], "description": "Output format"}, ...`
- **Go (fork output):** `..., "format": {"type": "string", "description": "Output format", "enum": ["json", "xml"]}, ...`

**Shared infrastructure concern — blast radius analysis.** Rewriting the `ToolProperty` struct changes the JSON output of **every renderer that calls `json.Marshal` on `api.Tool` or `api.ToolProperty`**. These are:

| Renderer | Marshal call | Affected by struct change? |
|---|---|---|
| `qwen35.go` | `marshalWithSpaces(tool)` | Yes — the fix target |
| `qwen3vl.go` | `marshalWithSpaces(tool)` | Yes — same `tojson` template |
| `olmo3.go` | `marshalWithSpaces(tools)` | Yes — same `tojson` template |
| `cogito.go` | `json.MarshalIndent(tool)` | Yes — **gated model, template unverified** |
| `deepseek3.go` | `json.Marshal(tool.Function.Parameters)` | Yes (parameters only) — but template doesn't render tool defs |
| `glm46.go` | `json.Marshal(tool)` | Yes — verified, uses `tojson` |
| `glm47.go` | `json.Marshal(tool)` + post-process | Yes — verified, uses `tojson` |
| `glmocr.go` | `json.Marshal(tool)` + post-process | Yes — **gated model, template unverified** |
| `lfm2.go` | `json.Marshal(tool.Function)` via helper | Yes — **gated model, template unverified** |
| `nemotron3nano.go` | Manual field access | No — immune |
| `qwen3coder.go` | Manual field access | No — immune |
| `functiongemma.go` | Manual field access | No — immune |

The change only affects output when both `Enum` and `Description` (or any new field) are present in a `ToolProperty`. For properties with only `Type` + `Description` (the common case), the output is unchanged. The risk is limited to Cogito, GLM-OCR, and LFM2 (gated, unverified templates) — and even then, only for tools with enum parameters.

Every verified model (Qwen 3.5, Qwen3VL, OLMo3, OLMo3.1, and the GLM-4 family) uses HuggingFace Transformers' insertion-order `tojson` override, which produces `enum` before `description`. No verified model uses stock Jinja2's `sort_keys=True`. The `Items` ordering issue is a **non-issue**: HuggingFace Transformers' `get_json_schema()` produces single-key `items` sub-objects (`{"type": "string"}`), which have no ordering to break. No fix is needed for `Items`.

**Per-model verification status:** Completed for all accessible model templates. Every model whose HuggingFace template is accessible (Qwen 3.5, Qwen3VL, OLMo3, OLMo3.1, GLM-4 family, GLM-5) uses `tojson` with insertion order preserved. DeepSeek V3's template doesn't render tool definitions. Gated models (Cogito, LFM2, GLM-OCR, GLM-4-32B-0414) are safe because even alphabetical sorting (`sort_keys=True`) produces the same ordering for the `required`/`properties` and `enum`/`description` cases that matter:
- `properties` before `required`: `p` < `r` alphabetically — matches HuggingFace Transformers insertion order. **FIXED.**
- `enum` before `description`: HuggingFace Transformers produces `enum` first (insertion order). Stock Jinja2 produces `description` first (`d` < `e`). All verified models use HuggingFace Transformers. Swapping `Enum` and `Description` in the `ToolProperty` struct is safe for all verified models.

**GLM verification note (2026-03-21):** GLM-4 templates are embedded in `tokenizer_config.json` on HuggingFace (not as separate `chat_template.jinja` files), which is why earlier searches missed them. All GLM chat models have Jinja2 templates that use `tojson` — confirmed for `THUDM/glm-4-9b-chat`, `THUDM/glm-4-9b-chat-1m`, `THUDM/GLM-4-9B-0414`, `THUDM/GLM-Z1-9B-0414`, and `zai-org/GLM-5`. Templates downloaded to `/tmp/glm4-templates/`. The GLM family shows a template evolution relevant to Ollama's renderers:

| Model | Tool serialization | Format | Object |
|---|---|---|---|
| GLM-4-9B-Chat | `tool['function'] \| tojson(indent=4)` | Pretty-printed multiline, Chinese UI | Function only |
| GLM-4-9B-0414 | `function \| tojson(indent=4, ensure_ascii=False)` | Pretty-printed multiline, Chinese UI | Function only |
| GLM-5 | `tool \| tojson(ensure_ascii=False)` | Single-line with spaces, English UI, `<tools>` wrapper | Whole tool |

Ollama's `glm46.go` renderer uses `json.Marshal(tool)` (compact, no spaces, whole tool, English UI, `<tools>` wrapper) — this matches the **GLM-5** template structure, not GLM-4's. The GLM-4.7 renderer (`glm47.go`) is similar but adds spaces via `formatGLM47ToolJSON`. Both renderers have significant deviations from their namesake GLM-4 templates (wrong JSON format, wrong object scope, wrong UI language). The GLM-4.6 tool tests are SKIPPED upstream (`"tool call ordering not guaranteed yet"`, added by jmorganca in `4f138a17`), and the GLM-4.7/Qwen3VL tool tests are commented out — reflecting the incomplete state of GLM tool support in Ollama generally.

**Impact of our planned `ToolProperty` changes on GLM:** All three planned changes (enum/description swap, new fields, bool/None capitalization) either improve or are neutral for GLM users. The enum/description swap moves GLM output closer to what every GLM template produces via `tojson(sort_keys=False)`. The new `omitempty` fields are invisible unless clients send them. The `formatToolCallArgument` fix is Qwen-only and does not touch GLM code paths. No GLM user gets worse output from any planned change.

**Remaining fixes — recommended approach:**
1. **`enum`/`description` swap** — Swap the `Enum` and `Description` field declarations in `ToolProperty` in `api/types.go`. Safe for all verified models.
2. **Field loss** — Add `Nullable *bool`, `AdditionalProperties any`, `PrefixItems []any` fields to `ToolProperty` with correct json tags and field ordering matching HuggingFace Transformers' `get_json_schema()` insertion order.
3. **Boolean/None capitalization** — Add `bool` type switch cases and update `nil` handling in `formatToolCallArgument` in `model/renderers/qwen3coder.go`.

For Qwen 3.5 specifically, success means the full round-trip consistency test in `server/qwen35_kvcache_roundtrip_test.go` passes completely, and the rewritten `TestQwen35RendererToolDefinitionsMatchOfficialTemplate` (see `test_gap_analysis.md` Gap 4) passes for all tool types including enum parameters.

### 4.3 Sub-Bug C — Compact Separators in `formatToolCallArgument` — DONE (2026-03-10)

The fork now fixes this locally in `qwen3coder.go`: `formatToolCallArgument` emits spaced JSON (`{"key": "value"}`) with HTML escaping disabled, matching the Qwen-family `tojson` convention for tool call arguments. This change is scoped to the Qwen renderers and avoids touching shared serializer infrastructure.

### 4.4 Sub-Bug D — Scalar Boolean and None Capitalization in `formatToolCallArgument` — OPEN

The official Qwen 3.5 Jinja2 template (line 122 of `chat_template.jinja`) renders tool call arguments using two code paths based on the argument's type:

- For dicts and lists: `args_value | tojson | safe` — serializes to JSON. Booleans become `true`/`false`, null becomes `null`. This is standard JSON.
- For everything else (strings, numbers, booleans, None): `args_value | string` — calls Python's `str()` function. `str(True)` produces `True` (capital T), `str(False)` produces `False` (capital F), `str(None)` produces `None`.

Go's `formatToolCallArgument` in `model/renderers/qwen3coder.go` line 256 uses `fmt.Sprintf("%v", value)` for non-string, non-collection types. `fmt.Sprintf("%v", true)` produces `true` (lowercase), not `True`. For `nil`, line 237 hardcodes `"null"`, not `"None"`.

**Byte-exact verification:** Both HuggingFace Transformers v5.3.0 and Go `Qwen35Renderer` outputs were saved to files and diffed for a conversation containing past tool calls with boolean and None arguments. The diff shows exactly three changed lines:

| Argument value | HuggingFace Transformers renders | Go renders | Match? |
|---|---|---|---|
| Scalar `True` | `True` | `true` | **No** |
| Scalar `False` | `False` | `false` | **No** |
| Scalar `None` | `None` | `null` | **No** |
| Large integer `10000000000` | `10000000000` | `1e+10` | **No** |
| `True` inside a list | `true` | `true` | **Yes** (both use JSON) |
| `None` inside a dict | `null` | `null` | **Yes** (both use JSON) |

The boolean/None mismatch affects **top-level scalar** booleans and None. Inside collections (lists, dicts), both pipelines use JSON serialization, which produces lowercase `true`/`false`/`null` — these match. The large integer mismatch affects `float64` values that are integer-valued and >= 1e7 (where Go's `fmt.Sprintf("%v")` switches to exponential notation). JSON integers unmarshal as `float64` in Go, and Go loses the int/float distinction. Python's `json.loads("10000000000")` produces `int(10000000000)`, and `str(10000000000)` produces `"10000000000"`.

**Fix:** Rewrite the type handling in `formatToolCallArgument`:
- `bool`: `true` → `"True"`, `false` → `"False"` (matching Python `str(True)`, `str(False)`)
- `nil`: `"null"` → `"None"` (matching Python `str(None)`)
- `float64` (integer-valued): use `strconv.FormatInt(int64(v), 10)` instead of `fmt.Sprintf("%v", v)`. This produces `"10000000000"` instead of `"1e+10"`. Non-integer floats keep `fmt.Sprintf("%v")` which already matches Python (e.g., `"3.14"`, `"1e-05"`).

This fix is scoped to the Qwen renderers (`formatToolCallArgument` is only called by `Qwen35Renderer` and `Qwen3CoderRenderer`). Both the Qwen 3.5 and Qwen3-Coder official Jinja2 templates use the identical `| string` filter line for scalar tool call arguments (verified by reading both templates).

### 4.5 Why These Bugs Matter — Two Distinct Failure Modes

**Failure mode 1 — Training distribution shift:** Every time the model uses tools, it sees a prompt format that differs from its training data.

**Failure mode 2 — KV cache round-trip mismatch (catastrophic for performance):** In a multi-turn conversation with tool calls, Ollama re-renders the entire conversation history on each turn. If the re-serialized bytes don't exactly match what the model originally generated, the KV cache cannot be reused for that portion — the engine must reprocess everything from the point of divergence.

The round-trip path: model generates spaced JSON → parser uses `json.Unmarshal` into Go types (original formatting lost) → renderer re-serializes via `formatToolCallArgument` → bytes differ → KV cache miss at divergence point.

**Empirically verified round-trip mismatches** (full `parseValue()` → Go type → `formatToolCallArgument()` tested for every parameter type):

| Parameter type | Round-trip match? |
|---|---|
| `string`, `integer`, `number`, `boolean`, `null`, untyped | **YES** |
| **`object`** | **NO — 3 simultaneous corruptions**: (1) spacing removed, (2) keys alphabetized, (3) HTML escaping injected |
| **`array`** | **NO — 2 corruptions**: (1) spacing removed, (3) HTML escaping injected |

Sub-bug C (spacing) and Sub-bug A (HTML escaping) are now fixed in the fork for tool call arguments. Key ordering (corruption 2 for objects) still requires a separate solution.

**Combined KV cache scaling effect:** In a 20-turn agentic coding conversation with 2-3 tool calls per turn, each tool call argument with any nested map/array triggers a KV cache miss. By turn 15-20, each response takes dramatically longer — invisible in testing because the model still produces correct output.

llama.cpp avoids all three sub-bugs by executing the Jinja2 template directly. Its `tojson` implementation at `common/jinja/value.cpp:179-180` defaults to spaced separators, preserves dict insertion order (line 1330: `as_ordered_object()` with comment "IMPORTANT: need to keep exact order"), and writes `<>&` literally.

However, llama.cpp's `tojson` has **one minor flaw**: float serialization at `value.cpp:1290` uses C++ `ostringstream` default formatting (6 significant digits), while Python's `json.dumps` preserves ~17 digits. Additionally, `1.0` becomes `1` in C++ but stays `1.0` in Python. Unlikely to affect model behavior in practice.

### 4.6 KV Cache Round-Trip Test (`server/qwen35_kvcache_roundtrip_test.go`)

The fork has 22 parameterized test cases covering round-trip consistency. The `allowCanonicalizationDrift` mechanism is broken — it checks raw form versus raw form instead of re-rendered form. The tests correctly identify 4 classes of round-trip failures: spacing, key ordering, number lexical forms, and HTML escaping.

### 4.7 Prerequisite Research for Remaining Fixes

**Research topic 1 — Shared infrastructure dependency map:** Build a complete call graph of which renderers call which shared functions (`marshalWithSpaces`, `jsonMarshalNoEscape` if created, `formatToolCallArgument`, `renderAdditionalKeys`, `formatToolDefinitionType`). For each call site, document whether the caller is model-family-specific or shared across families. Map which renderers serialize `ToolFunctionParameters` and how (direct `json.Marshal`, `json.MarshalIndent`, `marshalWithSpaces`, or custom). This determines the blast radius of each change.

**Research topic 2 — Per-model template verification:** For every model family whose renderer touches shared serialization code (at minimum: DeepSeek, GLM-4.7, GLM-OCR, Cogito, Olmo3, Nemotron3Nano, LFM2, and all Qwen variants), download and inspect the official HuggingFace Jinja2 template. For each, determine: (a) What JSON serialization function the template uses for tool definitions (`tojson`? `tojson(indent=4)`? custom?). (b) What key ordering the template produces for the `parameters` object. (c) Whether the template expects literal `<>&` or HTML-escaped versions. (d) Whether tool call arguments use spaced or compact JSON. (e) Whether tool schemas in the training data include `nullable`, `additionalProperties`, or `prefixItems` fields (which Go's `ToolProperty` currently drops). Only after this verification is complete can we know which changes are safe globally versus renderer-local.

**Research topic 3 — KV cache round-trip proof per renderer:** For each renderer that handles tool calls (Qwen3Coder, Qwen 3.5, Qwen3VL, DeepSeek, GLM, Cogito, Olmo3, Nemotron), trace the complete round-trip: model output format → parser extraction → Go type storage → renderer re-serialization → prompt bytes. Prove that after the proposed fix, the re-serialized bytes exactly match what the model was trained to generate. Document edge cases: nested objects, arrays, strings containing `<>&"`, unicode, empty objects, null values, `Optional[T]` parameters (nullable), `Dict[str, T]` parameters (additionalProperties), `Tuple` parameters (prefixItems).

**Research topic 4 — Scoping and sequencing strategy:** Decide upfront: fix only Qwen 3.5 (safest, narrowest blast radius), fix all Qwen family models (medium, since they share `tojson` conventions), or fix globally (maximum impact but requires completing Research topic 2 for all models).

**Research topic 5 — Test infrastructure strategy:** The existing renderer tests encode exact byte strings as golden expectations. Changing shared types or functions breaks tests across many renderer test files simultaneously, even if the runtime behavior is correct. Decide: update all affected test expectations (requires understanding whether the new output is correct for each model), or restructure tests to be more resilient to serialization changes (e.g., parse JSON output and compare structurally rather than byte-for-byte).

---

## Part 5: Grammar-Constrained Tool Call Generation — Fully Implemented

### 5.1 The Problem: Ollama Trusts the Model

Ollama's tool call parser is a free-form streaming state machine (the `Add()` method in `model/parsers/qwen3coder.go`). It receives the model's raw text output token-by-token after generation and parses it. When the model misbehaves, the parser silently propagates malformed output:

1. **Hallucinated function name.** `<function=nonexistent_tool>` creates a phantom `api.ToolCall` that propagates to the client.
2. **Malformed XML.** `transformToXML` regex partially matches, `xml.Unmarshal` may fail, and recovery is unclear.
3. **Unclosed `<tool_call>` tag.** The parser's `CollectingToolContent` state accumulates forever; if the model stops without closing, accumulated content is silently lost.
4. **Type-mismatched parameter values.** `parseValue()` falls back to returning raw strings where integers are expected.

In agentic workflows, a single malformed tool call cascades — each subsequent turn compounds the corruption.

### 5.2 llama.cpp Handles This Correctly

llama.cpp builds a PEG grammar from the declared tool schemas (`chat.cpp:1555-1616` at reference commit `a0ed91a`) before generation begins. The grammar is converted to GBNF and applied as logit masks at every token step. The model literally cannot produce an invalid tool call.

Key implementation details:
- **Line 1580:** Each tool becomes: `"<function=" + p.literal(name) + ">\n"` — only declared function names are valid
- **Lines 1596-1600:** String parameters accept free text; non-string parameters are constrained to match their JSON schema
- **Lines 1626-1639:** Lazy activation — grammar only constrains after `<tool_call>` trigger, content before is unconstrained

**No other model in Ollama has grammar-constrained tool call generation.** The fork is the first.

### 5.3 Infrastructure: What Was Already Available

The vendored llama.cpp in the fork already contained all the necessary C++ infrastructure:

**`json-schema-to-grammar.cpp`** (1153 lines): A pure JSON Schema to GBNF converter with zero model-specific code. Provides the `build_grammar()` callback API with `add_rule`, `add_schema`, and `resolve_refs` operations.

**C Bridge** (`sampling_ext.cpp`/`.h`): Already had `schema_to_grammar`, `grammar_init`, `grammar_apply`, `grammar_accept`, `grammar_free`. The gap was that `grammar_init` hardcoded `lazy=false` with no trigger patterns.

**Go Grammar Wrapper** (`llama/llama.go`): `NewGrammar`, `Apply`, `Accept`, `Free` all working. `GrammarSampler` fully integrated into the sampling pipeline.

**Grammar Flow** (`llm/server.go` → runner): `CompletionRequest` has a `Grammar string` field. Runner creates `GrammarSampler` from it when non-empty.

**Lazy Grammar Support** (`llama-grammar.cpp`): The grammar struct has full lazy support with `awaiting_trigger`, `trigger_buffer`, `trigger_patterns` fields. When `awaiting_trigger` is true, `llama_grammar_apply_impl` returns immediately without masking logits.

### 5.4 Constraint Comparison: What the Fork Does and Does Not Validate

| Constraint | llama.cpp | Fork |
|------------|-----------|------|
| XML structure well-formed | Yes | **Yes** — GBNF literals |
| Function name — only declared names | Yes — `p.literal(name)` alternation | **Yes** |
| Parameter name — only declared names | Yes | **Yes** |
| Lazy trigger activation | Yes | **Yes** |
| Unclosed `<tool_call>` / premature EOS | Yes — grammar prevents | **Yes** — `root ::= tool-call+` |
| Required parameters present in order | Yes — `p.repeat(arg_rule, 1, 1)` | **No** — forces fixed ordering which may fight model's learned order |
| Non-string parameter JSON schema constraints | Yes — `p.schema()` | **No** — diminishing returns; see Section 5.6 |
| `</parameter>` closing tag | Optional | **Optional** (matching llama.cpp) |
| Parallel tool calls | Yes — `p.repeat(tool_call, min, max)` | **Yes** — GBNF repetition |

### 5.5 Implementation: All 5 Steps Complete

**Implementation commits:** `b1038b91` (Step 3 GBNF builder), `59d1b367` (header cleanup), `4044b63f` (Steps 1-2, 4-5 + renderer fix)
**Actual total: 8 files changed, +212/-19 lines.**

#### Step 1: C Bridge Extension (~36 lines) — DONE

Added `grammar_init_lazy()` to `llama/sampling_ext.cpp` and `.h`. Calls `llama_grammar_init_impl` with `lazy=true` and regex trigger patterns. The `lazy` parameter was implicit (always `true` — the non-lazy path already exists via `grammar_init`), and trigger tokens were omitted since we use regex patterns exclusively.

#### Step 2: Go Wrapper Extension (~113 lines) — DONE

**`llama/llama.go`:** Added `NewGrammarLazy()` and `ToolCallGrammarFromJSON()`. Both `NewGrammar` and `NewGrammarLazy` delegate to a shared private `newGrammar()` — the branch between `grammar_init` and `grammar_init_lazy` is a single `if len(triggerPatterns) > 0` inside the shared function.

`ToolCallGrammarFromJSON` wraps the C++ `tool_call_grammar_from_json()` via CGo — uses a 64KB grammar buffer and 1KB error buffer (defined in `ToolCallGrammarFromJSON()` in `llama/llama.go`).

**`sample/samplers.go`:** Added `NewGrammarSamplerLazy()`. Both sampler constructors delegate to `newGrammarSampler()`.

**`llm/server.go`:** Added `ToolCallGrammarFromJSON` passthrough, `GrammarTriggerPatterns []string` field on `CompletionRequest`, and guard `if req.Grammar == "" && len(req.Format) > 0` so Format-derived grammar doesn't overwrite tool grammar.

**Concurrency safety (verified):** The `llama.Grammar` Go wrapper has a `sync.Mutex` protecting all C calls (`Apply`, `Accept`, `Free`). Each method checks `if g.c == nil { return }` after acquiring the lock, preventing use-after-free on request cancellation. `close(seq.quit)` triggers `defer grammar.Free()`, but the mutex serializes access with any concurrent `Sample()` call.

#### Step 3: GBNF Grammar Builder (~230 lines C++) — DONE

**Function:** `tool_call_grammar_from_json(tools_json, grammar_out, grammar_max_len, error_out, error_max_len)` in `llama/sampling_ext.cpp`. Takes a JSON array of `api.Tool` definitions, returns a GBNF grammar string.

**Defensive validation (10 error codes):** null pointers, zero-length buffers, invalid/non-array/empty JSON, missing or malformed fields, empty names, null bytes, forbidden characters (`>`, `<`, `\n`, `\r`), duplicate function/parameter names, required parameters not in properties, output truncation, grammar build exceptions.

**Character validation details:** Characters `=`, spaces, quotes, and GBNF metacharacters are NOT rejected — they are handled correctly by `gbnf_format_literal()` (wraps names in double-quoted GBNF strings) and the parser regex `[^>]+`. GBNF rule name collisions from unusual characters are resolved by the builder with numeric suffixes.

**Grammar buffer limit:** The 64KB buffer is sufficient for approximately 30 tools with ~10 parameters each (~30KB). For larger tool sets, the function returns `TOOL_GRAMMAR_ERR_TRUNCATED` gracefully. The limit is a single constant that can be trivially increased.

**Produced GBNF grammar** (verified output for get_weather + search example):
```
root             ::= tool-call+
tool-call        ::= "<tool_call>\n" tool-choice "</tool_call>" ws
tool-choice      ::= tool-get-weather | tool-search
tool-get-weather ::= "<function=get_weather>\n" tool-get-weather-arg-location
                     tool-get-weather-arg-unit? "</function>\n"
...
xml-arg-string   ::= (trie-based exclusion pattern for \n</parameter>, \n<parameter=, \n</function>)
```

The trie-based exclusion pattern for free-text parameter values was ported from `peg-parser.cpp` (~80 lines) because it is `static` (file-local) and not exposed in any header.

**Upstream compatibility note:** The latest llama.cpp (`c5a7788`) upgraded the trie from `unsigned char` to `uint32_t` for Unicode codepoint support, and refactored the tool grammar builder from inline `chat.cpp` to `chat-peg-parser.cpp`. The public API (`build_grammar`, `gbnf_format_literal`, `common_grammar_builder` in `json-schema-to-grammar.h`) is byte-for-byte identical.

#### Step 4: Server Wiring (~65 lines) — DONE

In `ChatHandler` at `server/routes.go`, gated on `m.Config.Parser == "qwen3.5"`:

1. Serialize `req.Tools` to JSON, call `llm.ToolCallGrammarFromJSON()` to build the GBNF string
2. Select mode-dependent trigger patterns based on `req.Think`:
   - **Thinking mode** (`req.Think.Bool() == true`): `[\s\S]*</think>[\s\S]*?(<tool_call>[\s\S]*)` — requires `</think>` before `<tool_call>` so the grammar doesn't activate on hallucinated tool calls inside thinking text
   - **Non-thinking mode** (`req.Think.Bool() == false`): `[\s\S]*?(<tool_call>[\s\S]*)` — no `</think>` required since the renderer prefills `<think>\n\n</think>\n\n` (it's in the prompt, not generated text)
3. Set `Grammar` and `GrammarTriggerPatterns` on the `CompletionRequest`

In `runner/ollamarunner/runner.go`, the grammar sampler creation branches on `len(req.GrammarTriggerPatterns) > 0` to choose lazy versus eager grammar initialization.

#### Step 5: Dead-End Error Detection (~10 lines) — DONE

After `s.grammar.Apply(tokens)` in `sample/samplers.go`, before calling `s.sample(tokens)`: checks if all tokens have `-inf` logits and returns `"sample: grammar rejected all tokens — grammar may be malformed or the generation reached an impossible state"`. This benefits **all models** using grammar-constrained generation.

Previously, a grammar dead-end would produce a cryptic "logits sum to NaN" error.

#### Bonus: Renderer Fix (Not in Original Plan)

During implementation, the renderer `isThinking` gate bug (Section 2.5) was discovered and fixed — the grammar trigger patterns depend on correct `</think>` placement, which led to verifying the renderer against the official template.

### 5.6 Future Enhancement: Full JSON Schema Constraints for Non-String Parameters

The existing `tool_call_grammar_from_json()` could be extended to call `builder.add_schema(name, param_schema)` for each non-string parameter instead of using the free-text `xml-arg-string` rule. ~20-30 lines in the existing function. The `json-schema-to-grammar.cpp` code is already vendored and functional.

Why this was not done now: (1) `schema_to_grammar` produces standalone grammars — merging multiple would cause rule name collisions; (2) the `build_grammar()` callback API is C++ with `std::function`, requiring a C shim for CGo; (3) diminishing returns — the two most damaging failure modes (hallucinated function names and parameter names) are already blocked.

### 5.7 Error Handling Paths

**Happy path:** Grammar in `awaiting_trigger` → model generates freely → `<tool_call>` trigger fires → grammar constrains → well-formed tool call emitted.

**Grammar dead-end:** Detected by Step 5 — clear error message instead of cryptic NaN.

**Model outputs content only:** Grammar stays in `awaiting_trigger`, model generates freely, hits EOS. Correct behavior.

**Grammar conflicts with Format:** Format→Grammar guard ensures tool grammar takes precedence. Content before `<tool_call>` is unconstrained.

**`GGML_ABORT` on incomplete grammar:** `llama_grammar_accept_impl` has a hard assertion if EOS is accepted while grammar hasn't reached completion. Correctly-constructed grammar prevents this: `root ::= tool-call+` sets EOS to `-inf` mid-tool-call.

---

## Part 6: What Changed in Upstream Ollama Since `82848a78`

### 6.1 New Commits (4 Total, `82848a78` → `9896e36`)

| Commit | Description | Qwen 3.5 Impact |
|--------|-------------|-----------------|
| `9896e36` | Fix cloud model limit lookups in IDE integrations | None |
| `15732f0` | Use native Ollama API endpoint for OpenClaw | None |
| `562c76d` | Add qwen3.5 context length for launch | Minor — adds `{Context: 262144, Output: 32768}` for IDE integrations |
| `122c68c` | Loosen thinking level constraint | None — harmony/gptoss feature |

### 6.2 CRITICAL BUGS — Still Unfixed in Upstream at `9896e36`

| Bug | Upstream Status | Fork Status |
|-----|----------------|-------------|
| **Prefill bug** — `Render()` in `qwen35.go` missing `&& len(message.ToolCalls) == 0` | **STILL BROKEN** | Fixed (commit `fbae6976`) |
| **`mropeInterleaved` default** — `New()` in `model.go` defaults to `false` | **STILL BROKEN** | Fixed (commit `9ec17fc1`) |
| **`repeat_last_n` not wired** — `NewSampler` ignores API parameter | **STILL BROKEN** | Fixed (commit `ab234955`) |
| **Penalty sampler feeds prompt tokens** — `Accept()` called on prompt | **STILL BROKEN** | Fixed (ring buffer architecture) |
| **`</think>` not closed in `Qwen3VLRenderer`** — gated on `content != ""` | **STILL BROKEN** | Fixed (commit `fbae6976`) |
| **History `<think>` blocks gated on `isThinking`** | **STILL BROKEN** | Fixed (commit `4044b63f`) |

---

## Part 7: llama.cpp's Own Qwen 3.5 Bugs (4 Confirmed)

While llama.cpp avoids the JSON serialization bugs by executing Jinja templates directly, it has its own set of correctness bugs:

### 7.1 Bug 1 — Non-Thinking Mode Generation Prompt Produces Wrong Token Sequence (Fixed in Post-Refactor llama.cpp)

**Versions affected:** llama.cpp up to and including `c5a7788`. **Fixed in:** post-refactor versions (`0cd4f47`+, March 10, 2026), verified in local clones at `/tmp/llama-cpp-fresh/` and `/tmp/llama_cpp_devstral2512_audit/`.

**The bug (in affected versions):** The per-model specialized handler `common_chat_params_init_qwen3_coder` at `common/chat.cpp:1527` called `apply(tmpl, inputs)` without including `enable_thinking` in the Jinja template context. Since `enable_thinking` was always `undefined`, the template always produced `<think>\n`. llama.cpp then post-processed: when `!inputs.enable_thinking` and the prompt ended with `<think>\n`, it appended `</think>`, producing `<think>\n</think>` (3 tokens). The official non-thinking prefill is `<think>\n\n</think>\n\n` (6 tokens) — with an empty line inside the think block and a double-newline after. The 3-token sequence was never seen during training. This is a real distribution shift.

**The fix (in post-refactor versions):** The per-model specialized handlers (`common_chat_params_init_qwen3_coder`, `common_chat_params_init_qwen3_vl`, etc.) were replaced with a unified `common_chat_template_direct_apply()` function. This unified path correctly passes `enable_thinking` to the Jinja template context at `chat.cpp:773`:
```cpp
{"enable_thinking", inputs.enable_thinking},
```
With `enable_thinking` properly set in the context, the template's own `add_generation_prompt` block produces the correct 6-token non-thinking prefill directly, without needing post-processing heuristics.

The fork and upstream Ollama produce the exact official 6-token prefill via `emitEmptyThinkOnNoThink`, and have done so since before this llama.cpp fix.

### 7.2 Bug 2 — Prompt Tokens Contaminate Penalty Sampler Ring Buffer (Masked by Default)

At `tools/server/server-context.cpp:208-215`, llama.cpp feeds ALL prompt tokens into the penalty sampler. Currently masked by `repeat_penalty=1.0` default.

### 7.3 Bug 3 — No Model-Specific Sampler Defaults

llama.cpp uses the same hardcoded defaults for all models at `common/common.h:186-208`. Qwen recommends `temp=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0`, `repeat_penalty=1.05`. No mechanism exists to read per-model recommended sampling parameters.

### 7.4 Bug 4 — Float Precision Loss in `tojson`

Float serialization at `value.cpp:1290` uses 6 significant digits versus Python's ~17. Also, `1.0` renders as `1` instead of `1.0`. Very unlikely to affect model behavior.

---

## Part 8: What Changed in llama.cpp Since `d969e933`

### 8.1 CRITICAL PERFORMANCE: CUDA Async Copy + Reduced Synchronization (Commit `2cd20b7`)

**`ggml-cuda.cu`**: `ggml_backend_cuda_cpy_tensor_async()` now supports **CPU→CUDA async copies** using `cudaMemcpyAsync`. The fork's vendored version requires both source and destination to be CUDA backends, forcing a synchronous CPU thread block on every CPU-to-GPU tensor copy.

**`ggml-backend.cpp`**: The scheduler's `compute_splits()` was restructured from a per-input synchronization pattern (O(inputs) sync points per compute split) to a "sync-async-async-async-sync-graph" (saaasg) pattern (O(1) sync points per compute split). This is the single most impactful **performance** change llama.cpp has made since the fork's vendored GGML version.

**Not Blackwell/next-gen specific.** Zero references to sm_100, Blackwell, or Ada Lovelace. This benefits all NVIDIA GPUs, especially with partial GPU offload and multi-GPU setups.

**How to adopt:** Update `ml/backend/ggml/ggml/src/ggml-backend.cpp` and `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu`. Focused change to two files in the GGML layer, but the scheduler restructuring is moderately complex — cherry-picking may require manual conflict resolution.

### 8.2 CORRECTNESS: M-RoPE `can_shift()` Guard (Commit `99bd67c`)

`llama-kv-cache.cpp` — `get_can_shift()` now returns `false` for models with `n_pos_per_embd() > 1` (M-RoPE models including Qwen 3.5). K-shift (context window sliding) requires a single scalar position per token; M-RoPE models have 4 position values per token.

**Fork impact — nuanced:** The fork's `can_shift()` returns `true` unconditionally, but this only affects the **C++ `llamarunner`** path. Qwen3next models use the **Go `ollamarunner`**, which gracefully degrades — falls back to full reprocessing instead of corrupt shift. **Should still be adopted** as a defensive fix (3 lines).

### 8.3 KDA Chunk Size (Commit `a0ed91a`) — No Impact

Chunk size changed from hardcoded `64` to `kda ? 16 : 64`. KDA (Key-Dependent Attention) is used by Kimi-linear models. Qwen 3.5 uses GDA (Gate-Dependent Attention) → chunk size remains 64 → no change.

### 8.4 Qwen 3.5 Model Type Detection (Commit `872646b`)

Updated size detection: Dense variants 0.8B, 4B, 9B, 27B added; MoE layer counts fixed. The fork should verify its `qwen3next` model code handles the full range of layer counts (24, 32, 40, 48, 60, 64) and embedding dimensions.

### 8.5 Text-Only Converter Registration (Commit `cf23251`)

Registers `Qwen3_5ForCausalLM` and `Qwen3_5MoeForCausalLM`. The fork's Go converter uses config fields, not HF class names. Not directly applicable.

### 8.6 Multi-Modal Prompt Caching (Commits `f20469d`, `d7d826b`)

llama.cpp's server now caches multi-modal prompts, avoiding redundant vision encoding. Ollama's Go server has no equivalent. For repeated-image scenarios, this is a significant performance gap (potentially 2-10x).

### 8.7 Other GGML Changes

| Change | Commit | Impact |
|--------|--------|--------|
| CDNA3 MFMA flash attention | `ecbcb7e` | +7-39% on AMD MI300X (datacenter only) |
| Vulkan fp16 FA AMD fix | `723c710` | Fixes broken flash attention on AMD RDNA2 Windows |
| MXFP4 CPU repack | `d903f30` | Low priority (Q4_K_XL uses Q4_K, not MXFP4) |
| Vulkan partial offload | `3191462` | AMD Vulkan GPU users |
| AMX batched support fix | `4e76d24` | Intel AMX matmul fix |
| kleidiai SME fp16 q4_0 | `137435f` | ARM aarch64 performance |
| `ggml_is_contiguous_n` fix | `7f5ee54` | Edge case correctness when `ne == 1` |
| Vulkan memory overlap fusion | `3769fe6` | Prevents incorrect fusion |
| Context checkpoint restore fix | `01cd448` | llama.cpp server specific |

---

## Part 9: Speculative Decoding — Confirmed Impossible for Hybrid Recurrent

### 9.1 llama.cpp's Explicit Block

The compatibility test at `common/speculative.cpp:801-835` checks partial tail removal on the model's memory. For hybrid models, the recurrent sub-cache at `llama-memory-recurrent.cpp:142-180` explicitly rejects it. Both `LLM_ARCH_QWEN35` and `LLM_ARCH_QWEN3NEXT` are classified as hybrid → speculative decoding disabled.

### 9.2 The Mathematical Blocker

Recurrent hidden state is **destructively updated** token by token: `S_{t+1} = α ⊙ S_t + β_t^T v_t`. Rolling back requires:
1. **Fine-grained per-token checkpointing** — prohibitively expensive for 48 recurrent layers
2. **Recomputation from last checkpoint** — the fork's checkpoint system saves state every 1,664 tokens, far too coarse for speculative decoding
3. **Run recurrent layers non-speculatively** — only 25% of model is attention; 75% benefit is lost

### 9.3 Bottom Line

Speculative decoding for Qwen 3.5 is not implementation difficulty — it's **fundamental architectural incompatibility**. All alternatives (Medusa, lookahead, attention-only speculation, fine-grained checkpointing) are near-term impractical. **Do not invest time in this.**

---

## Part 10: Parallelism — Restricted but Architecturally Supported

### 10.1 The Restriction

`server/sched.go:448-453` — both the fork and upstream Ollama force `numParallel = 1` for all hybrid/recurrent architectures (`qwen35`, `qwen35moe`, `qwen3next`, `lfm2`, `lfm2moe`, `nemotron_h`, etc.).

### 10.2 The Runner Does Support Parallelism

The Go runner and cache system are architecturally capable: `kvcache/recurrent.go` maintains a `slotForSeq` map with independent states per sequence. Multiple sequences are supported. The runner only restricts parallelism when `Config().Cache == nil`, which is not the case for `qwen3next`.

### 10.3 Why It's Restricted

`deltanet.go:80-87` requires all sequences in a batch to have the **same token count** (`ErrUnsupportedBatchLayout`). This is a structural requirement of chunked linear attention. Different requests have different prompt lengths and can't be batched. During generation (1 token/step), they could theoretically batch.

### 10.4 Opportunity

llama.cpp demonstrates this works using `split_equal(n_ubatch, true)` — per-sequence sub-batching. Lifting the restriction in Ollama would require rewriting the batch construction logic, which is a non-trivial effort.

### 10.5 Vision and Parallelism

- **Ollama**: Vision encoding happens synchronously, one request at a time
- **llama.cpp**: Separate `clip_ctx` + multi-modal prompt caching allows reusing encoded images

---

## Part 11: Full Capability Utilization — What Works and What's Missing

### What Already Works Correctly in the Fork

- **Flash attention** for FullAttention layers (every 4th layer). GatedDeltaNet recurrent layers use chunked linear attention with delta state updates.
- **KV cache quantization** via `OLLAMA_KV_CACHE_TYPE=q8_0` (only for the 16 FullAttention layers — recurrent state stays F32).
- **131K context window** — `num_ctx 131072` works. No artificial cap.
- **YaRN RoPE scaling** with correct attention factor formula: `1.0 / (1.0 + 0.1*ln(ropeScale))`.
- **MRoPE (Multi-Resolution RoPE)** with architecture-based IMROPE default.
- **Grammar/GBNF constrained decoding** for structured output and now tool calls.
- **Mmap with smart defaults** — disabled automatically for partial GPU offload scenarios.
- **Vision (unified GGUF from ollama.com/library only)** — 27-layer vision encoder, multi-turn conversations, images from all messages. Does NOT work with Unsloth text-only GGUFs.
- **Penalty sampling** — fork's private ring buffer excludes prompt tokens.

### What Needs Fixing (Correctness)

| Issue | Impact | Status |
|-------|--------|--------|
| `ToolFunctionParameters` key ordering: `required` before `properties` (should be reversed) | Distribution shift on every tool definition in every system prompt. **Safe to fix globally** — every Python/Jinja2 configuration produces `properties` first (`p` < `r` alphabetically, and HF insertion order also puts `properties` first). Per-model verification complete. | **DONE** — struct reordered in `api/types.go`, all renderer and API tests updated |
| `ToolProperty` key ordering: `description` before `enum` (should be reversed for HF-trained models) | Distribution shift on tool parameters with enumerated values. Every verified model (Qwen 3.5, Qwen3VL, OLMo3, OLMo3.1, GLM-4 family, GLM-5) uses HuggingFace Transformers' `tojson` with `sort_keys=False`, which preserves insertion order and produces `enum` before `description`. Fix: swap `Enum` and `Description` field declarations in `ToolProperty` in `api/types.go`. | **OPEN** |
| `ToolProperty` field loss: `nullable`, `additionalProperties`, `prefixItems` silently dropped during `json.Unmarshal` | Model sees tool schema missing type constraints it was trained on. `Optional[T]` parameters (producing `nullable: true`) are common in Python tool definitions. Fix: add `Nullable *bool`, `AdditionalProperties any`, `PrefixItems []any` fields to `ToolProperty` in `api/types.go`. | **OPEN** |
| `formatToolCallArgument` scalar boolean/None capitalization: Go produces `true`/`false`/`null`, HuggingFace Transformers template produces `True`/`False`/`None` | Different token IDs for past tool call arguments in multi-turn conversations. Fix: add `bool` type switch cases and update `nil` handling in `formatToolCallArgument` in `model/renderers/qwen3coder.go`. | **OPEN** |
| `ToolProperty.Items` map key alphabetization | Non-issue for training data — HuggingFace Transformers' `get_json_schema()` produces single-key `items` sub-objects (`{"type": "string"}`), which have no ordering to break. | **Won't fix** |

### Performance Opportunities

| Opportunity | Impact | Effort | Status |
|-------------|--------|--------|--------|
| **CUDA async copy + reduced sync** | 5-15% CUDA throughput | Medium (2 files) | **Available to adopt** |
| **`num_batch 2048`** in Modelfile | ~10% prompt eval speed | Zero (config only) | **Available to adopt** |
| **Multi-modal prompt caching** | 2-10x repeated-image vision | High | Architectural change |
| **Lift parallelism restriction** | Concurrent requests | High | Batch rewrite needed |
| **Fused GatedDeltaNet kernel** | Reduced graph compilation | Very High | Custom CUDA kernel |
| Speculative decoding | 2-4x generation speed | N/A | **Impossible** (hybrid recurrent) |

### Vision With Third-Party GGUF Files

**Official `ollama.com/library/qwen3.5:27b` (unified GGUF):** Vision loads and runs. All 1307 tensors in one file with `vision.block_count=27`. Only template fidelity gap: images prepended to front of messages instead of inline positioning.

**Unsloth text-only GGUF (`qwen3.5-custom` Modelfile):** No vision, by design. 851 tensors, no `vision.block_count`. Correctly returns `ErrNoVisionModel` if images are sent.

**The trap: `FROM hf.co/unsloth/...`:** Ollama auto-discovers the mmproj file, classifies it as a projector, then the Go engine rejects the entire load because `qwen3next` is in `OllamaEngineRequired()` with no fallback. The custom Modelfile uses `FROM /tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf` (manually downloaded) to avoid this.

**Enabling vision with Unsloth GGUFs requires significant research — six specific challenges:**

1. **The Go engine loads a single GGUF file.** The entire model loading pipeline (`ml/backend/ggml/ggml.go`) assumes one GGUF. The mmproj is a second GGUF with its own tensor namespace. Merging them at load time requires understanding how tensor names map between the two files (the main GGUF uses `v.*` prefixed tensor names for vision, the mmproj may use different naming conventions depending on the converter).

2. **Unsloth's mmproj may not match Ollama's expected vision tensor layout.** Ollama's `qwen3vl.NewVisionModel(c)` reads vision config from GGUF metadata and expects tensors at specific GGUF names (e.g., `v.blk.{n}.attn_qkv.weight`). Unsloth's mmproj was generated by llama.cpp's `convert_hf_to_gguf.py`, which may use different tensor names. A tensor name mapping layer would be needed.

3. **The vision encoder is 27 layers (456 tensors).** This is not a small projector — it's a full Vision Transformer (ViT) with 1152 hidden size, 16 heads, and patch size 16. Loading it separately, mapping its tensors, and integrating it with the Go engine's single-GGUF buffer allocation is non-trivial.

4. **Quantization compatibility.** The mmproj files come in BF16/F16/F32. The text model is Q4_K_XL. The Go engine needs to handle mixed precision across the two GGUFs, placing vision tensors on the correct device with the correct quantization type.

5. **Offline GGUF merge as an alternative.** Instead of fixing the Go engine, a tool that merges the Unsloth text GGUF + mmproj into a single unified GGUF matching Ollama's expected format would avoid engine changes but requires understanding both GGUF layouts and producing correct metadata. This is probably the more practical path.

6. **Testing.** Even after loading, the vision pipeline (`EncodeMultimodal`, `PostTokenize`) was tested with Ollama's own unified GGUFs. Third-party vision tensors from a different converter may have subtle differences (weight ordering, normalization conventions, patch embedding layout) that produce wrong results without error.

### What Doesn't Matter

- **Token healing / BPE anti-corruption**: Not needed — Ollama tokenizes full rendered prompts as single strings.
- **Continuous batching**: Blocked by `numParallel=1` and equal-token constraint.
- **Dynamic NTK RoPE scaling**: Static YaRN sufficient for 131K training length.
- **Separate K/V cache quantization types**: Low impact — recurrent state dominates memory.
- **Lazy expert loading for MoE layers**: 4-bit quantization already fits consumer GPUs.

---

## Part 12: Architecture Notes

### Recurrent Layer Inference

The fork's 4-line interval computation `(i+1)%interval != 0` is identical to llama.cpp's formula. Both produce correct results for all real-world GGUFs, though via different mechanisms than upstream Ollama's 52-line `inferRecurrentLayers()`.

### `SetInplace` → Balanced Concat Tree

Both Ollama codebases now use a balanced binary concat tree for chunk assembly in `deltaNetChunked`. O(log N) graph depth, supported on all backends. llama.cpp still uses `ggml_set_inplace()` (works in llama.cpp's GGML but not on Ollama's Metal/Vulkan backends).

### Vision Packaging: Ollama versus llama.cpp

Ollama bundles vision tensors into the main GGUF (1307 tensors). llama.cpp uses a separate mmproj GGUF loaded via `--mmproj`. Unsloth follows llama.cpp's convention. Ollama's Go engine explicitly rejects split vision at `llm/server.go:149-152`.

### Shared Issues (Both Fork and Upstream)

- **`lastQueryIndex` initialization edge case**: Neither validates the "no user messages" case that the official template explicitly rejects with `raise_exception`.
- **BOS/EOS logging bug at `vocabulary.go:57`**: Checks `v.BOS` where it should check `v.EOS`. Purely cosmetic — the actual EOS append runs unconditionally. Never fires for Qwen 3.5 (`AddEOS` defaults to `false`).

---

## Prioritized Action Items

### Open Items for the Fork

| Priority | Item | Effort | Status |
|----------|------|--------|--------|
| **P0** | Adopt CUDA async copy + reduced sync from `ggml-cuda.cu` and `ggml-backend.cpp` | Medium (2 files, conflict resolution) | **OPEN** |
| ~~**P1**~~ | ~~Fix `ToolFunctionParameters` key ordering: move `Properties` before `Required` in `api/types.go`~~ | ~~Small~~ | **DONE** — struct reordered, all tests updated, test renamed to `TestQwen35RendererToolDefinitionsMatchOfficialTemplate` with HF ground truth |
| **P2** | Fix `ToolProperty` key ordering (`enum`/`description` swapped) — swap `Enum` and `Description` field declarations in `ToolProperty` in `api/types.go`. All verified models (Qwen 3.5, Qwen3VL, OLMo3, OLMo3.1, GLM-4 family, GLM-5) use HF's `tojson` with `sort_keys=False` which produces `enum` before `description`. Improves or is neutral for all models including GLM (see GLM verification note in Section 4.2). | Small | **OPEN** |
| **P2** | Fix `ToolProperty` field loss and ordering — rewrite the `ToolProperty` struct in `api/types.go` to the exact 9-field layout specified in Section 4.2 (verified against every HuggingFace Transformers v5.3.0 type combination). Adds `AdditionalProperties any`, `PrefixItems []any`, `Nullable *bool` at positions 4-6, moves `Enum` to position 7 and `Description` to position 8. This single struct change fixes BOTH the field loss AND the `enum`/`description` ordering. | Small-Medium | **OPEN** |
| **P2** | Fix `formatToolCallArgument` scalar type rendering — rewrite type handling in `model/renderers/qwen3coder.go`: (1) `bool`: `true`→`"True"`, `false`→`"False"`. (2) `nil`: `"null"`→`"None"`. (3) `float64` integer-valued: use `strconv.FormatInt(int64(v), 10)` to produce `"10000000000"` instead of `"1e+10"`. Matches Python `str()` used by both the Qwen 3.5 and Qwen3-Coder templates' `| string` filter (both templates verified identical). | Small | **OPEN** |
| **P1** | Adopt M-RoPE `can_shift()` guard in vendored `llama-kv-cache.cpp` | Small (3 lines) | **OPEN** |
| **P2** | Increase `num_batch` to 2048 in the Modelfile | Zero (config-only) | **OPEN** |
| **P3** | Verify `Qwen3VLRenderer` thinking variant needs `emitEmptyThinkOnNoThink: true` | Research | **OPEN** |
| **P3** | Verify Qwen 3.5 model family size diversity (0.8B through 397B) | Small | **OPEN** |
| **P3** | Adopt Vulkan fp16 FA fix for AMD RDNA2 | Small | **OPEN** |
| **Informational** | Speculative decoding confirmed impossible — do not pursue | N/A | **Dead end** |
| **Informational** | Multi-modal prompt caching — future vision performance improvement | High | **Not started** |
| **Informational** | Third-party GGUF vision — requires significant research | Very High | **Not started** |
| **Informational** | Parallelism — lift `numParallel=1` with per-sequence sub-batching | High | **Not started** |

### Completed Items

| Priority | Item | Commit | Status |
|----------|------|--------|--------|
| ~~P0~~ | Fix prefill bug in all 3 renderers | `fbae6976` | **DONE** |
| ~~P0~~ | Fix `mropeInterleaved` architecture-based default | `9ec17fc1` | **DONE** |
| ~~P0~~ | Wire `repeat_last_n` API parameter to Go sampler | `ab234955` | **DONE** |
| ~~P0~~ | Implement private ring buffer penalty sampler | (original commit) | **DONE** |
| ~~P0~~ | Grammar-constrained tool call generation (5 steps) | `b1038b91`, `59d1b367`, `4044b63f` | **DONE** |
| ~~P0~~ | Fix shared HTML escaping in tool-definition serialization path (Sub-bug A) | (2026-03-11) | **DONE** |
| ~~P1~~ | Fix compact separators in `formatToolCallArgument` (Sub-bug C) | (2026-03-10) | **DONE** |
| ~~P1~~ | Fix `isThinking` gate on history `<think>` blocks (two-part fix) | `4044b63f` | **DONE** |
| ~~P1~~ | Fix `</think>` closure in `Qwen3VLRenderer` | `fbae6976` | **DONE** |
| ~~P1~~ | Revert `repeatPenalty` default from 1.1 to 1.0 | `57e3f80f` | **DONE** |
| ~~P1~~ | Implement unconditional GGUF converter KV emission | (original commit) | **DONE** |

### Known Issues in Other Implementations (Not Fork Action Items)

| Implementation | Issue | Severity |
|----------------|-------|----------|
| **llama.cpp** (up to `c5a7788`) | Non-thinking prefill produced `<think>\n</think>` (3 tokens) instead of official `<think>\n\n</think>\n\n` (6 tokens) — **fixed in post-refactor versions** (`0cd4f47`+) via unified `common_chat_template_direct_apply()` that passes `enable_thinking` to Jinja | Medium-High (**fixed**) |
| **llama.cpp** | Prompt tokens contaminate penalty sampler ring buffer | Medium (masked by defaults) |
| **llama.cpp** | No model-specific sampler defaults | Low |
| **llama.cpp** | Float precision loss in `tojson` (6 digits versus 17) | Very Low |
| **Upstream Ollama** | All bugs in Section 6.2 (prefill, mropeInterleaved, repeat_last_n, penalty sampler, think closure, isThinking gate) | Critical to Medium |
| **Upstream Ollama** | Free-form tool call parser (no grammar constraints) | Structural limitation |
| **Unsloth GGUF** | Modified template (defensive `is mapping` + key iteration) — functionally equivalent | Informational |

---

## Appendix A: File-Level Change Map

### Fork Files Modified (All Complete)

| File | What |
|------|------|
| `model/renderers/qwen35.go` | Dedicated Qwen 3.5 renderer — JSON tool defs, tools-first ordering, image support, `lastQueryIndex`, unconditional `<think>` wrapping in history, prefill bug fix |
| `model/renderers/qwen3coder.go` | Shared qwen3-coder renderer — thinking support, prefill fix, spaced JSON `formatToolCallArgument` |
| `model/renderers/qwen3vl.go` | Qwen3VL renderer — prefill fix, `</think>` closure fix |
| `model/renderers/renderer.go` | Renderer routing — maps `"qwen3.5"` to `Qwen35Renderer` |
| `model/parsers/qwen35.go` | Dedicated Qwen 3.5 parser — thinking extraction, delegates tool calls to embedded `Qwen3CoderParser` |
| `model/parsers/qwen3coder.go` | Shared qwen3-coder parser — thinking support, XML tool call parsing |
| `model/parsers/parsers.go` | Parser routing — maps `"qwen3.5"` to `Qwen35Parser` |
| `model/models/qwen3next/deltanet.go` | GatedDeltaNet — balanced concat tree in `Forward()` |
| `model/models/qwen3next/model.go` | Model definition — architecture-based defaults, `Validate()` |
| `sample/samplers.go` | Ring buffer penalty sampler, lazy grammar sampler, dead-end detection |
| `sample/transforms.go` | `applyPenalties()` — operates on generated-only token window |
| `runner/ollamarunner/runner.go` | Runner — removed `Accept()`/`Reset()`, passes `RepeatLastN`, lazy grammar branch |
| `convert/convert_qwen3next.go` | GGUF converter — unconditional KV emission |
| `server/prompt.go` | Binary search prompt truncation |
| `server/routes.go` | `ChatHandler` grammar wiring for qwen3.5, mode-dependent triggers |
| `tokenizer/special.go` | Special token splitting with `strings.Contains` early-out |
| `tokenizer/vocabulary.go` | Stack buffer in `Merge()` BPE hot path |
| `api/types.go` | `RepeatPenalty: 1.0` default, custom `MarshalJSON` for tool types (no HTML escaping) |
| `llama/sampling_ext.cpp` | C bridge — `grammar_init_lazy()`, `tool_call_grammar_from_json()` |
| `llama/sampling_ext.h` | C bridge header — function declarations, 10 error codes |
| `llama/llama.go` | Go wrappers — `NewGrammarLazy()`, `ToolCallGrammarFromJSON()`, consolidated `newGrammar` |
| `llm/server.go` | `CompletionRequest` with `Grammar`/`GrammarTriggerPatterns`, `ToolCallGrammarFromJSON()` passthrough, Format→Grammar guard |

### Vendored C++ Files (Read-Only Reference)

| File | What |
|------|------|
| `llama/llama.cpp/common/json-schema-to-grammar.cpp` (1153 lines) | JSON Schema → GBNF converter |
| `llama/llama.cpp/common/json-schema-to-grammar.h` (43 lines) | Header — `build_grammar()`, `common_grammar_builder` |
| `llama/llama.cpp/src/llama-grammar.cpp` | Grammar engine — logit masking, state advancement, lazy trigger handling |
| `llama/llama.cpp/src/llama-grammar.h` | Grammar struct with lazy fields, `llama_grammar_init_impl` signature |

### llama.cpp Files Changed Since `d969e933` (Relevant to Qwen 3.5)

| File | Change | Fork Equivalent |
|------|--------|-----------------|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | Async CPU→CUDA copy | `ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` |
| `ggml/src/ggml-backend.cpp` | saaasg sync pattern | `ml/backend/ggml/ggml/src/ggml-backend.cpp` |
| `src/llama-kv-cache.cpp` | M-RoPE `can_shift()` guard | `llama/llama.cpp/src/llama-kv-cache.cpp` |
| `src/models/delta-net-base.cpp` | KDA chunk 16 (no impact) | `model/models/qwen3next/deltanet.go` |
| `src/llama-model.cpp` | Model type detection | N/A (Go handles sizing) |
| `convert_hf_to_gguf.py` | ForCausalLM registration | `convert/convert_qwen3next.go` (different approach) |
| `src/llama-sampler.cpp` | **No changes** — pre-existing penalty contamination bug | `sample/samplers.go` (fork is correct) |
| `common/chat.cpp` | **No changes** — pre-existing non-thinking prefill bug | `model/renderers/qwen35.go` (fork is correct) |
| `common/jinja/value.cpp` | **No changes** — minor `tojson` float precision divergence | N/A (Go renderers) |
| `tools/server/server-context.cpp` | Multi-modal caching; pre-existing prompt→penalty feed bug | `runner/ollamarunner/runner.go` |

### Upstream Ollama Files Changed Since `82848a78`

| File | Change | Fork Status |
|------|--------|-------------|
| `cmd/config/integrations.go` | qwen3.5 context length, glm-5 | Missing |
| `cmd/config/opencode.go` | Cloud model limit lookup fix | Missing |
| `cmd/config/openclaw.go` | Native API endpoint | Missing |
| `server/routes.go` | Loosen thinking level constraint | None — harmony/gptoss feature |

## Appendix B: Build Notes

- **C++ syntax check:** `c++ -std=c++17 -fsyntax-only -I llama/llama.cpp/common -I llama/llama.cpp/include -I llama/llama.cpp/src -I llama/llama.cpp/ggml/include -I ml/backend/ggml/ggml/include -I llama/llama.cpp/vendor -I llama llama/sampling_ext.cpp`
- **Full build** requires CGo; standalone C++ tests need stubs for `string_join`, `string_split`, `string_repeat`, `string_format`, `ggml_backend_cpu_buffer_type`
