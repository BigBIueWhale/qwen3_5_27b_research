# Ollama Fork vs Upstream: Precise Analysis

Fork: `BigBIueWhale/ollama` @ `9ec17fc1` (7 commits atop `v0.17.4` merge) + dedicated Qwen 3.5 renderer/parser.
Upstream: `ollama/ollama` master @ `82848a78` (commit message: `model: fix renderer and parser for qwen3.5 (#14605)`).
llama.cpp: `ggml-org/llama.cpp` master @ `d969e933` (commit hash: `d969e933e172821b4519f66aa4b660bc0846b320`) — the canonical C++ inference engine. Ollama vendors GGML (the tensor math library) from llama.cpp but reimplements everything else in Go: chat template rendering is hardcoded Go renderers instead of llama.cpp's Jinja2 engine, sampling is a Go rewrite instead of llama.cpp's `llama-sampler.cpp`, and the model runner is `ollamarunner` instead of llama.cpp's server. This means llama.cpp bugs in GGML affect Ollama, but llama.cpp's correct Jinja2 template handling does NOT help Ollama — Ollama must reimplement the same logic in Go renderers/parsers. Note: llama.cpp has its own set of bugs separate from Ollama — see `fork_vs_latest_upstream_and_llama_cpp.md` Section 6.4.
GGUF note: The Unsloth Dynamic 2.0 `UD-Q4_K_XL` GGUF at `/tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf` embeds a **modified** Jinja2 template (not byte-identical to the official HuggingFace version). Unsloth changed the tool call arguments section: `tool_call.arguments is defined` → `is mapping`, and replaced `|items` tuple unpacking with key iteration + bracket access. This is a defensive workaround for older Jinja engine compatibility. Functionally equivalent for normal use. See `fork_vs_latest_upstream_and_llama_cpp.md` Section 6.5 for details.
Official reference: Qwen 3.5 Jinja2 template from `Qwen/Qwen3.5-35B-A3B` ([publicly accessible on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/tokenizer_config.json)). **Also verified** against `Qwen/Qwen3.5-27B` ([publicly accessible](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/tokenizer_config.json)) — both repos contain byte-identical `chat_template` fields.
Qwen3-Coder reference: `Qwen/Qwen3-Coder-30B-A3B-Instruct` Jinja2 template ([verified via public access](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)) — confirms XML tool definitions, system-first ordering. Upstream Ollama's `Qwen3CoderRenderer` faithfully matches it.

---

## Scorecard

| Area | Fork | Upstream Ollama @ `82848a78` | Winner |
|------|------|----------|--------|
| Qwen 3.5 renderer (tool defs, ordering, think blocks) | Dedicated `Qwen35Renderer` with prefill bug fix; shared `isThinking` gate bug on history think blocks (section 2.5) | Dedicated `Qwen35Renderer` without prefill bug fix; same `isThinking` gate bug | **Fork** (prefill fix) |
| Image placement in vision messages | All images prepended to front of message — `Images []ImageData` is a flat array with no positional info (see section 2.3 caveat) | Same — all 3 vision renderers (`Qwen35`, `Qwen3VL`, `LFM2`) prepend | Neither (structural Ollama API limitation) |
| Qwen 3.5 parser (thinking extraction, tool call delegation) | Dedicated `Qwen35Parser` (byte-identical to upstream) | Dedicated `Qwen35Parser` | Tie |
| Prefill guard (`len(message.ToolCalls)==0`) | Fixed in **all 3** renderers (`qwen35.go`, `qwen3coder.go`, `qwen3vl.go`) | Broken in **all 3** renderers | **Fork** |
| `</think>` closure in `Qwen3VLRenderer` | Always closes `</think>` tag | Conditionally gates on `content != ""` — leaves unclosed `<think>` tag | **Fork** |
| `mropeInterleaved` default for third-party GGUFs | Architecture-aware default (`c.Architecture() != "qwen3next"`) | Hardcoded `false` — **silently wrong RoPE type** for all third-party Qwen 3.5 GGUFs | **Fork** |
| `repeat_last_n` API parameter | Wired through to Go sampler via `NewSampler` parameter | Silently ignored (hardcoded `DefaultPenaltyLookback = 64`) | **Fork** |
| Penalty sampling architecture (repeat, presence, frequency — all three share one token window) | Private `recordToken()` inside `Sample()`, ring buffer, excludes prompt tokens — matches original Keskar et al. 2019 CTRL paper. All 3 penalty types benefit. | Public `Accept()` called externally by runner, feeds prompt tokens into penalty window — all 3 penalty types corrupted | **Fork** |
| `repeatPenalty` default | `1.0` (same as upstream — reverted from `1.1` to avoid affecting other models; safe to raise per-model via Modelfile because fork's ring buffer excludes prompt tokens) | `1.0` (penalty system is a no-op; llama.cpp comments: `// 1.0 = disabled`) | Tie (but fork's architecture makes it safe to raise; upstream's doesn't) |
| Third-party GGUF compatibility | Architecture-based `mropeInterleaved` default (the critical fix) + shared `ssm_dt.bias` alt tag + `Validate()` + nil checks | Hardcoded `false` mropeInterleaved default (wrong for third-party qwen35 GGUFs) + shared `ssm_dt.bias` alt tag + `Validate()` + nil checks | **Fork** (due to mropeInterleaved; other fixes are shared or equivalent) |
| GGUF converter KV emission | Unconditional (always writes `rope.mrope_interleaved` and `ssm.v_head_reordered`) | Conditional (only writes when `true` — third-party GGUFs that omit the key fall through to wrong default) | **Fork** |
| `SetInplace` → balanced concat tree | Fixed (matches upstream Ollama commit `3490e959`) | Fixed | Tie |
| Tokenizer performance | Binary search prompt truncation, `strings.Contains` early-out, stack buffer BPE merge | None of these optimizations | **Fork** |
| JSON tool serialization (key ordering + HTML escaping + spacing) | Go struct field order, HTML-escaped `<>&`, compact JSON in `formatToolCallArgument` — mismatches HuggingFace's `tojson` override (fix planned, empirically verified — see section 3.1) | Same | Neither yet (both wrong vs HuggingFace's `tojson` override at [`transformers` v5.2.0 `chat_template_utils.py:463-466`](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/utils/chat_template_utils.py#L463-L466): `json.dumps(x, ensure_ascii=False, sort_keys=False)` — NOT stock Jinja2's `tojson` which uses `htmlsafe_json_dumps` with `sort_keys=True`; llama.cpp correct for JSON but has its own `tojson` float precision flaw at `value.cpp:1290` — 6 significant digits vs Python's ~17, and `1.0` → `1` without decimal point) |
| `enable_thinking` generation prompt | Correct via `emitEmptyThinkOnNoThink` — correctly prefills `<think>\n\n</think>\n\n` when `think: false` | Same (correct) | Both correct. **llama.cpp WRONG** — `chat.cpp:1527` does not pass `enable_thinking` to Jinja; post-processing at `chat.cpp:1534-1536` produces `<think>\n</think>` (3 tokens) instead of official `<think>\n\n</think>\n\n` (6 tokens) — a token sequence never seen in training |
| Tool call parser robustness | Free-form streaming parser — currently trusts model output. **Grammar-constrained generation is a committed P0 action item.** See dedicated plan: [`grammar_constrained_tool_calls_plan.md`](grammar_constrained_tool_calls_plan.md) | Same (identical parser code, no plans to fix) | **llama.cpp is structurally correct** — PEG grammar-constrained generation. **Fork will match this.** |

---

## Part 1: Where the Fork Beats Upstream Ollama

### 1.1 CRITICAL: Prefill bug fix — upstream Ollama's most dangerous active bug

**Fault: Upstream Ollama (`82848a78`).** All three upstream Ollama renderers — `qwen35.go`, `qwen3coder.go`, `qwen3vl.go` — have this bug. The fork correctly fixed it in commit `fbae6976` by adding `&& len(message.ToolCalls) == 0` to the prefill condition in all three renderers. This is the fork's most valuable contribution. llama.cpp (`d969e933`) does not have this bug — it uses `add_generation_prompt` as a separate parameter passed to the Jinja2 template, which never conflates message closure with generation prompt skipping. Prefill is an Ollama-specific concept that llama.cpp does not need.

**Upstream bug locations**: `qwen35.go:136`, `qwen3coder.go:140`, `qwen3vl.go:82`
**Fork fix locations**: `qwen35.go:136`, `qwen3coder.go:159` (line differs from upstream because the fork added thinking support to `Qwen3CoderRenderer` in prior commits), `qwen3vl.go:82`

All three upstream Ollama renderers have:
```go
prefill := lastMessage && message.Role == "assistant"
```

The fork adds the guard in all three:
```go
prefill := lastMessage && message.Role == "assistant" && len(message.ToolCalls) == 0
```

This treats any last assistant message as a prefill, **including messages with tool calls**. The consequences differ by renderer:

**In `qwen35.go` and `qwen3vl.go`** — **doubly broken**: The `<|im_end|>` tag is gated by `if !prefill` (`qwen35.go:169`, `qwen3vl.go:122`), so when `prefill=true` on an assistant+toolcalls message, BOTH the `<|im_end|>` is omitted AND the generation prompt (`<|im_start|>assistant\n<think>\n`) is skipped. The rendered prompt ends with `</tool_call>` — no closing tag, no new start. The model's view of the conversation is corrupted.

**In `qwen3coder.go`** — the tool-call branch unconditionally writes `<|im_end|>` (line 156), but the post-loop generation prompt at lines 187-189 is gated by `if lastMessage && !prefill`, which is skipped since `prefill=true`. The model gets no `<|im_start|>assistant\n` to begin generating.

**When does this trigger?** Any time a client sends a chat request where the last message is an assistant message with tool calls populated:
- A client replays a conversation from saved history that ended at a tool-call boundary
- An agent framework constructs the message list with the assistant's tool-call message last
- The fork's own test suite (`qwen3coder_test.go:327-376`) proves this is an expected and tested input shape

The official Qwen 3.5 Jinja2 template **always** emits `<|im_end|>` after every assistant message, regardless of tool calls. The `add_generation_prompt` flag is a separate concern — it controls whether `<|im_start|>assistant\n<think>\n` is appended at the end, independently of message closures. Ollama's prefill conflates two things (omitting `<|im_end|>` for streaming continuation, and skipping the generation prompt) that should be independent.

### 1.2 MEDIUM: `</think>` not closed when content is empty in `Qwen3VLRenderer`

**Fault: Upstream Ollama (`82848a78`).** The fork correctly fixed this in commit `fbae6976` — the `</think>` tag is always emitted after thinking content. Upstream Ollama's `Qwen3VLRenderer` conditionally gates it on `content != ""`, leaving an unclosed `<think>` tag for thinking-only responses. llama.cpp (`d969e933`) always closes `</think>` — the parser at `chat-parser-xml-toolcall.cpp:756` enforces closure, and the Jinja2 template itself always emits the closing tag.

**File**: `model/renderers/qwen3vl.go:96-100`

Upstream Ollama (lines 95-103):
```go
sb.WriteString("<|im_start|>" + message.Role + "\n<think>\n" + strings.Trim(contentReasoning, "\n"))
if content != "" {
    sb.WriteString("\n</think>\n\n" + strings.TrimLeft(content, "\n"))
}
```

Fork (lines 95-103):
```go
sb.WriteString("<|im_start|>" + message.Role + "\n<think>\n" + strings.Trim(contentReasoning, "\n"))
sb.WriteString("\n</think>\n\n")
if content != "" {
    sb.WriteString(strings.TrimLeft(content, "\n"))
}
```

Upstream Ollama conditionally emits `\n</think>\n\n` only when `content != ""`. For thinking-only assistant messages (thinking content but empty visible content), the `<think>` tag is opened but `</think>` is never written.

This no longer affects Qwen 3.5 (which now uses its own dedicated `Qwen35Renderer`, which always closes `</think>` at line 144). But `Qwen3VLRenderer` is still used by `qwen3-vl-instruct` and `qwen3-vl-thinking`, where this bug remains live in upstream Ollama.

### 1.3 CRITICAL: `mropeInterleaved` defaults to `false` — silent data corruption in upstream Ollama for third-party GGUFs

**Fault: Upstream Ollama (`82848a78`).** The fork (commit `9ec17fc1`) correctly defaults to `c.Architecture() != "qwen3next"` at `model.go:589`, which evaluates to `true` for `qwen35` — matching llama.cpp's approach. llama.cpp (`d969e933`) **hardcodes** the RoPE type per architecture in `llama_model_rope_type()` at `llama-model.cpp:9109-9113`: `case LLM_ARCH_QWEN35: return LLAMA_ROPE_TYPE_IMROPE` (value 40). No GGUF metadata lookup needed — the architecture enum determines the RoPE type at model load. This is a **silent data corruption bug** in upstream Ollama: the model loads, runs, and produces text, but the positional encoding is wrong in every full-attention layer. Users would blame the model quality, not the inference engine.

**Upstream Ollama line**: `model/models/qwen3next/model.go:641`
**Fork line**: `model/models/qwen3next/model.go:589`

Upstream Ollama (`model.go:641`):
```go
mropeInterleaved: c.Bool("rope.mrope_interleaved", c.Bool("mrope_interleaved", false)),
```

Fork (`model.go:589`):
```go
mropeInterleaved: c.Bool("rope.mrope_interleaved", c.Bool("mrope_interleaved", c.Architecture() != "qwen3next")),
```

Third-party GGUFs (llama.cpp converter, Unsloth) do not write the ollama-internal `rope.mrope_interleaved` key (it is not in llama.cpp's GGUF constants at `gguf-py/gguf/constants.py`, confirmed in `d969e933`). The actual Qwen 3.5 27B model has `mrope_interleaved: true` (matching llama.cpp's hardcoded `LLAMA_ROPE_TYPE_IMROPE`).

**Concrete impact traced through code**: `mropeInterleaved` controls the RoPE type at `model.go:67-75`:
- `true` → `rope.WithInterleaveMRoPE()` → sets `opts.Type |= 1<<3 | 1<<5` = **40** = `GGML_ROPE_TYPE_IMROPE`
- `false` → `rope.WithMRoPE()` → sets `opts.Type |= 1<<3` = **8** = `GGML_ROPE_TYPE_MROPE`

With upstream Ollama's `false` default, all 16 full-attention layers in Qwen 3.5 27B use RoPE type 8 (MROPE) instead of 40 (IMROPE). These are completely different dimensional interleaving patterns for position encoding across the time/height/width sections. The model **silently produces wrong output** — it still runs, but attention patterns are misaligned across every full-attention layer.

Note: `ssm.v_head_reordered` defaults are functionally equivalent between fork (`c.Architecture() != "qwen3next"`) and upstream Ollama (`defaultVHeadReordered` which positively matches `"qwen35" || "qwen35moe"`). Both produce `true` for `qwen35`/`qwen35moe` and `false` for `qwen3next`.

### 1.4 Penalty sampling architecture (repeat, presence, AND frequency) — the fork matches the original academic paper; upstream Ollama and llama.cpp both match HuggingFace's pre-fix mistake

**Fault: Both upstream Ollama (`82848a78`) and llama.cpp (`d969e933`). The fork is the only one correct per the original academic formulation.**

The repetition penalty was introduced by Keskar et al. (2019) in the CTRL paper (arXiv:1909.05858). The paper's formula operates over *"generated tokens g"* — the set of tokens the model has produced during inference. Prompt tokens are not in `g`. This is not ambiguous; the paper explicitly distinguishes the generation set from the conditioning context.

**Upstream Ollama** feeds prompt tokens into the penalty history via the public `Accept()` method called from the runner at **two call sites**:

1. **Sequence load** (`runner.go:951-957`): `seq.sampler.Reset()` (line 951) then loops through ALL cached inputs calling `seq.sampler.Accept(inp.Token)` (line 956). The last 64 prompt tokens become part of the penalty window.
2. **Batch processing** (`runner.go:694-701`): As pending input tokens are committed to the cache, each token is `Accept()`-ed (line 700). Both prompt tokens and generated tokens flow through this path (the generated token becomes an input for the next forward pass).

**llama.cpp** (`d969e933`) has the same behavior. The server's `init_sampler()` at `server-context.cpp:208-212` loops through ALL prompt tokens and calls `common_sampler_accept(smpl.get(), id, false)` on each one. This feeds prompt tokens into the penalty ring buffer (`llama-sampler.cpp:2644-2656`), same as upstream Ollama.

This is the same mistake HuggingFace `transformers` made (and [acknowledged as a bug](https://github.com/huggingface/transformers/issues/36642) — users reported *"weird strings"* and false penalization of common phrases, fixed in [PR #37625](https://github.com/huggingface/transformers/pull/37625), April 2025). llama.cpp users reported the [same problem](https://github.com/ggml-org/llama.cpp/issues/331): *"the repetition penalty is affecting the anti-prompt and response prefix."* All three projects (upstream Ollama, llama.cpp, HuggingFace `transformers`) feed prompt tokens into the penalty window because it's the simplest implementation — you just call `accept()` on every token. The correct implementation requires distinguishing prompt tokens from generated tokens, which is what the fork does.

**The fork's architecture** (`sample/samplers.go:19-32`): The `Sampler` struct contains `recentTokens []int32` (ring buffer), `recentTokenPos int` (write cursor), and `repeatLastN int` (window size from API, user-configurable). The private `recordToken()` method (lines 196-206) implements a classic circular buffer: O(1) per token, zero allocations after warmup, permanently bounded at `repeatLastN` entries. The runner's only interaction is `seq.sampler.Sample(logits)` at line 760 — it never touches internal state. No external code can feed prompt tokens into the penalty window because there is no public API to do so. This matches the original paper's intent and is the **correct** behavior for agentic tool-calling models where the prompt contains tool names the model must reproduce verbatim.

**llama.cpp's ring buffer architecture** (`d969e933`, `llama-sampler.cpp:23-132`): Uses a proper `ring_buffer<llama_token>` template class with the same design — fixed capacity, O(1) push, modular arithmetic. The ring buffer is initialized with `penalty_last_n` capacity at `llama-sampler.cpp:2763`. The key difference: llama.cpp's `llama_sampler_penalties_accept()` is a **public** function called by external code (the server feeds prompt tokens through it), while the fork's `recordToken()` is **private** and called only from `Sample()`. Both use ring buffers, but the fork's encapsulation prevents the prompt-token contamination problem.

**Upstream Ollama's history architecture** (`sample/samplers.go:21-32`): The `Sampler` struct contains `history []int32` (an append-then-truncate slice). `Accept()` (lines 38-43) appends the token, then if `len(history) > DefaultPenaltyLookback`, copies the last 64 entries to the front of the slice and truncates — a **slide-and-truncate** pattern (O(n) per trim, allocates on growth). `Reset()` (line 36) sets history to nil. The runner calls these externally at 3 separate call sites (cache load at line 951+956, pending inputs at line 700, reprocess error at line 565).

**Cache shift behavior**: On normal cache shift, neither codebase touches the sampler. On the reprocess error path, upstream Ollama calls `seq.sampler.Reset()` (line 565), clearing the entire history — the reprocessed tokens will be `Accept()`-ed back through `pendingInputs`. The fork does nothing to the ring buffer on either path. Not resetting is the correct behavior: cache shift is a memory management operation, not a semantic boundary. The tokens in the ring buffer are tokens the model actually generated recently. Clearing the ring buffer on cache shift would destroy repetition penalty information at the exact moment it matters most — during long generations that have filled the context window.

**All three penalty types share the same token window — not just `repeatPenalty`.** The `applyPenalties()` function (`transforms.go:121-148` in the fork, `transforms.go:37-64` in upstream Ollama) takes a single `recentTokens []int32` slice and applies ALL three penalty types to it in a single pass:

| Penalty type | Math per token in window | Effect |
|---|---|---|
| `repeatPenalty` | Multiplicative: `logit /= penalty` (positive) or `logit *= penalty` (negative) | Scales down repeated tokens proportionally |
| `frequencyPenalty` | Subtractive: `logit -= frequencyPenalty * count` | Penalizes proportionally to occurrence count |
| `presencePenalty` | Subtractive: `logit -= presencePenalty` | Flat penalty for ANY token that appeared at least once |

In the fork, `recentTokens` is the ring buffer containing ONLY generated tokens. In upstream Ollama, `recentTokens` is the `history` slice containing BOTH prompt AND generated tokens. **This means the prompt-token contamination bug affects all three penalty types equally** — not just `repeatPenalty`.

**This is especially critical for the actual Modelfile configuration.** The custom Modelfile for Qwen 3.5 27B (`qwen3.5-custom.Modelfile`) uses:
```
PARAMETER repeat_penalty 1.0    # disabled
PARAMETER presence_penalty 1.5  # ACTIVE — the primary anti-repetition mechanism
```

The user disables `repeat_penalty` (1.0 = identity) and relies entirely on `presence_penalty 1.5` to prevent degenerate repetition. With the fork's architecture, `presence_penalty 1.5` subtracts 1.5 from the logit of every token the model has generated in the last N tokens — correctly penalizing self-repetition without touching prompt tokens. With upstream Ollama's architecture, `presence_penalty 1.5` would subtract 1.5 from the logit of every token in the last 64 tokens of history — including prompt tokens like tool names (`get_weather`), JSON structural tokens (`{`, `}`, `:`), parameter names, and previous tool results. **A 1.5 logit subtraction is severe** (roughly halving the probability of a token at typical logit scales), and applying it to prompt tokens would catastrophically degrade tool calling accuracy.

This is the strongest practical argument for the fork's penalty architecture: the user's own Modelfile configuration — `presence_penalty 1.5` with `repeat_penalty 1.0` — would be actively harmful under upstream Ollama's sampler because it penalizes prompt tokens that the model MUST reproduce for correct tool calls.

### 1.5 `repeat_last_n` API parameter — silently ignored in upstream Ollama's Go runner

**Fault: Upstream Ollama (`82848a78`).** The Go runner's `NewSampler` simply doesn't accept the parameter — the API field exists, the default exists (`api/types.go:1065`: `RepeatLastN: 64`), but nothing connects it to the sampler. The fork (commit `ab234955`) fixed this by adding `repeatLastN` as the 9th parameter to `NewSampler` and wiring `req.Options.RepeatLastN` through at `runner.go:899`. llama.cpp (`d969e933`) correctly wires `penalty_last_n` as a user-configurable parameter (`common.h:195`: `int32_t penalty_last_n = 64`) that flows through to `llama_sampler_init_penalties()`.

**File**: `sample/samplers.go`

Upstream Ollama's `NewSampler` signature (`sample/samplers.go:159`):
```go
func NewSampler(temperature float32, topK int, topP float32, minP float32,
    repeatPenalty float32, presencePenalty float32, frequencyPenalty float32,
    seed int, grammar *GrammarSampler) Sampler
```

No `repeatLastN` parameter. The constant `DefaultPenaltyLookback = 64` (line 19) is hardcoded in both `Accept()` (line 40: truncates history to 64 entries) and `tokenCounts()` in `transforms.go` (line 34: only looks at last 64). The runner call at lines 897-907 passes `req.Options.RepeatPenalty` etc. but NOT `req.Options.RepeatLastN`.

A user setting `repeat_last_n: 128` in the API has **zero effect** — the value flows through `api/types.go` into `req.Options.RepeatLastN` (default 64 at line 1065) but is never read by the Go sampler. It is a dead field.

**Caveat**: The value IS respected in the legacy C++ `llamarunner` at `runner/llamarunner/runner.go:651` (`RepeatLastN: req.Options.RepeatLastN`), which passes it to llama.cpp's C sampler. So this is a Go-runner-only regression, not a universal API lie. But the Go runner is the default for all new model architectures including `qwen3next`.

### 1.6 `repeatPenalty` default 1.1 vs 1.0 — enables the penalty system to actually function

This is not a policy preference. With upstream Ollama's and llama.cpp's `1.0` default:
- The penalty math is identity (no effect on any logit)
- The `Accept()`/`Reset()` bookkeeping is wasted work every forward pass
- The `repeat_last_n` API parameter is doubly useless (hardcoded to 64 in upstream Ollama, AND the penalty is `1.0`)
- llama.cpp explicitly comments: `float penalty_repeat = 1.00f; // 1.0 = disabled` (`common.h:196`)

With the fork's `1.1` default:
- Repeated positive logits are divided by `1.1` (~9% reduction); negative logits are multiplied by `1.1` (10% further suppression)
- The ring buffer actually serves a purpose
- The `repeat_last_n` parameter controls something real
- This is safe ONLY because the fork excludes prompt tokens (see 1.4) — with prompt tokens in the window (as in llama.cpp and upstream Ollama), `1.1` would degrade tool calling

The exact penalty math in all three codebases (fork `transforms.go:134-139`, upstream Ollama `transforms.go:50-57`, llama.cpp `llama-sampler.cpp:2690-2696`) is identical:
```
if logit > 0: logit /= repeatPenalty
if logit < 0: logit *= repeatPenalty
```

One remaining nuance: the fork **does** penalize the model's own thinking tokens (they're generated), so if `<think>I need to call get_weather</think>` is shorter than 64 tokens, the `get_weather` tokens are still in the penalty window when the model emits the actual tool call. At `1.1` this is mild (~9% logit reduction), but users who raise the penalty to `1.5+` would break tool calling even with the fork's architecture.

### 1.7 GGUF converter unconditional KV emission — prevents default-mismatch bugs

**Fork converter** (`convert/convert_qwen3next.go`): Always writes both keys unconditionally:
```go
kv["rope.mrope_interleaved"] = q.RopeParameters.MRopeInterleaved   // line 287
kv["ssm.v_head_reordered"] = q.shouldReorderVHeads()                // line 314
```

**Upstream Ollama converter**: Only writes when `true`:
```go
if q.RopeParameters.MRopeInterleaved { kv["rope.mrope_interleaved"] = true }  // lines 287-289
if q.shouldReorderVHeads() { kv["ssm.v_head_reordered"] = true }              // lines 316-318
```

For `rope.mrope_interleaved`: upstream Ollama's conditional emission is fine for Ollama-converted GGUFs (the converter writes `true` for `qwen35`, which is correct). The risk is for third-party GGUFs that omit the key — upstream Ollama defaults to `false` (wrong for `qwen35`), the fork defaults based on architecture (correct). See section 1.3.

Note: `ssm.v_head_reordered` and `rope.mrope_interleaved` are **ollama-internal conventions**. They do not exist in llama.cpp's GGUF constants (`gguf-py/gguf/constants.py`, confirmed in `d969e933`). llama.cpp sidesteps the problem entirely: RoPE type is hardcoded per architecture in `llama_model_rope_type()` (`llama-model.cpp:9109-9113`), and V-head reordering is done **physically during GGUF conversion** by `_LinearAttentionVReorderBase` (`convert_hf_to_gguf.py:4782-4791`) — the converter permutes tensor weights so no runtime flag is needed. Third-party GGUFs from Unsloth, bartowski, or llama.cpp's converter will **never** contain these keys.

### 1.8 Third-party GGUF compatibility fixes — 3 fork-original commits

The fork's third-party GGUF compatibility approach differs from upstream's. Some fixes originated in the fork, some are shared:

- **`ssm_dt.bias` tensor name** (`deltanet.go:44`): `gguf:"ssm_dt,alt:ssm_dt.bias"`. Both fork and upstream have this — the alt tag allows loading GGUFs where the llama.cpp converter wrote `ssm_dt.bias` instead of `ssm_dt.weight`. **Shared fix.**
- **Recurrent layer classification**: The fork uses the interval formula directly (`model.go:513-517`), bypassing the `headCountKV` array entirely. Third-party GGUFs emit `head_count_kv` as a scalar UINT32, not a per-layer array — the fork is inherently immune because it never reads that array. Upstream's `inferRecurrentLayers()` (upstream `model.go:497-549`) handles scalar broadcasts via a fallback path that reaches the same formula. Both produce correct results for all real-world GGUFs, but via different mechanisms. **Different implementations, same outcome.**
- **Architecture-based defaults for `mropeInterleaved`** (`model.go:589`): The fork defaults to `c.Architecture() != "qwen3next"` (true for qwen35). Upstream defaults to `false`. **Fork-only fix — this is the critical one** (see section 1.3).
- **Architecture-based defaults for `vHeadReordered`** (`model.go:580`): Fork uses `c.Architecture() != "qwen3next"`. Upstream uses `defaultVHeadReordered()` helper (`arch == "qwen35" || arch == "qwen35moe"`). **Functionally equivalent — both produce `true` for qwen35.**

Plus both codebases have: `Validate()` method (`model.go:440-478`) on `Model` implementing the `model.Validator` interface (catches missing tensors at load time instead of nil-pointer panic at inference time), and inline nil checks in `GatedDeltaNet.Forward()` (`deltanet.go:139-148`) for `SSMDT`, `SSMA`, `SSMConv1D` (+`.Weight`), `SSMNorm`, and `SSMOut`. The fork ported these from upstream in commit `8da09b1e`.

### 1.9 Tokenizer performance optimizations

- **Binary search prompt truncation** (`server/prompt.go`): O(log N) tokenize calls instead of linear scan. Monotonicity assumption valid (dropping messages can only reduce token count).
- **Special token splitting** (`tokenizer/special.go`): `strings.Contains` early-out before regex matching. 13 test cases validate equivalence.
- **Stack buffer BPE merge** (`tokenizer/vocabulary.go`): 128-byte stack buffer eliminates heap allocations in BPE merge hot path. Leverages Go 1.12+ map-lookup compiler optimization.

---

## Part 2: Qwen 3.5 Renderer/Parser — All Previously-Open Issues Resolved

The fork now has a dedicated `Qwen35Renderer` (`model/renderers/qwen35.go`, 195 lines) and `Qwen35Parser` (`model/parsers/qwen35.go`, 239 lines), ported from upstream Ollama's `82848a78` implementation with the prefill bug fix (section 1.1) applied. The `Qwen35Parser` is byte-for-byte identical to upstream Ollama. The `Qwen35Renderer` differs by exactly one line: the prefill condition at line 136 adds `&& len(message.ToolCalls) == 0`.

The previous approach of routing `"qwen3.5"` through `Qwen3CoderRenderer`/`Qwen3CoderParser` caused seven distinct template fidelity problems (sections 2.1–2.7 below), **all now resolved**. The `Qwen3CoderRenderer` and `Qwen3CoderParser` are completely untouched by this change — they still serve `"qwen3-coder"` correctly. Only the `"qwen3.5"` routing cases in `renderer.go` and `parsers.go` were changed. All existing tests pass, and 17 new tests (5 renderer + 11 parser + 1 integration routing test in `qwen3_test.go:211`) cover the new code.

Note: the fork's `Qwen3CoderRenderer` AND `Qwen3CoderParser` already differ from upstream's versions — both have thinking support that upstream lacks:
- **Fork's `Qwen3CoderRenderer`** (`qwen3coder.go`): has `isThinking` and `emitEmptyThinkOnNoThink` fields, thinking block rendering at lines 164-170, think prefill at lines 222-227, and the prefill bug fix at line 159. Upstream's version has none of these.
- **Fork's `Qwen3CoderParser`** (`qwen3coder.go`): has `hasThinkingSupport` and `defaultThinking` fields, 4 parser states (including `CollectingThinking` and `ThinkingDoneTransition`), and `HasThinkingSupport()` that returns the field value. Upstream's version has only 2 parser states (no thinking states) and `HasThinkingSupport()` returns `false` unconditionally.
- These differences predate the current change and are part of the fork's earlier commits (`fbae6976`). The current change only affects `"qwen3.5"` routing — `"qwen3-coder"` continues to use these fork-enhanced files.

### 2.1 FIXED: Tool definition format — Qwen 3.5 uses a JSON/XML hybrid that must be matched exactly

**Was**: The fork routed `"qwen3.5"` through `Qwen3CoderRenderer` (at `renderer.go:59-60`), which emits XML tool definitions (`<function><name>...`). The Qwen 3.5 official template uses JSON tool definitions (`{{ tool | tojson }}`).

**Now**: The dedicated `Qwen35Renderer` uses `marshalWithSpaces()` (at `json.go:8-45`) for JSON tool definitions, matching the official Qwen 3.5 Jinja2 template. Test `TestQwen35RendererUsesXMLToolCallingFormat` explicitly asserts the `<tools>` section is present with JSON content.

**Background — the Qwen 3.5 / Qwen3-Coder hybrid format split**: Alibaba Qwen trains two model families on **different tool definition formats but identical tool call formats**. This is verified against the official Jinja2 templates downloaded from HuggingFace ([`Qwen/Qwen3.5-27B/raw/main/tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/tokenizer_config.json), [`Qwen/Qwen3-Coder-30B-A3B-Instruct/raw/main/tokenizer_config.json`](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/raw/main/tokenizer_config.json)):

| | Tool definitions (system prompt — what the model **reads**) | Tool calls (assistant output — what the model **writes**) | Tool call arguments (inside XML parameters) | System+tools order |
|---|---|---|---|---|
| **Qwen3-Coder** | **XML**: `<function><name>get_weather</name><description>...` — the template iterates `tool.function.parameters.properties` and emits XML tags per field | **XML**: `<tool_call><function=get_weather><parameter=location>...` | Scalars: `args_value \| string`. Maps/lists: `args_value \| tojson \| safe` (JSON inside XML) | System first, tools second |
| **Qwen 3.5** | **JSON**: `{{ tool \| tojson }}` — the entire tool dict serialized as one JSON blob | **XML**: identical to Qwen3-Coder, character-for-character | Identical to Qwen3-Coder | Tools first, system appended after |

The tool call output format is the same for both. Only the tool *definition* rendering in the system prompt and the system message ordering differ. This is easy to get wrong — it's why the fork author originally (commit `fbae6976`) routed Qwen 3.5 through the Qwen3-Coder renderer instead of implementing a dedicated one.

The fork's `Qwen35Renderer` and `Qwen35Parser` (at `model/renderers/qwen35.go` and `model/parsers/qwen35.go`, routed via `renderer.go:59-60` and `parsers.go:52-53`) now match the official template's hybrid format:
- **Renderer**: emits JSON tool definitions via `marshalWithSpaces(tool)`, emits XML tool calls via `<tool_call><function=...><parameter=...>`, handles `<tool_response>` wrapping for tool results
- **Parser**: `Qwen35Parser` extracts `<think>` reasoning content, then delegates all post-thinking content (including XML tool call parsing) to an embedded `Qwen3CoderParser` — which finds `<tool_call>...</tool_call>` blocks, transforms the `<function=name>` syntax into valid XML via `transformToXML()`, and XML-unmarshals parameters

A byte-level comparison of the fork's full rendered output against the official template (run via `apply_chat_template` with `transformers` v5.2.0 and the `Qwen/Qwen3.5-27B` tokenizer) shows **the entire structural format matches exactly** — system prompt layout, `<tools>` wrapper, postamble instruction text, `<tool_call>` XML, `<tool_response>` wrapping, `<think>` blocks, `lastQueryIndex` logic, `<|im_start|>`/`<|im_end|>` placement. The **only** difference is inside the JSON tool definition blob itself, due to the JSON serialization bugs described in section 3.1.

**OPEN BUG — JSON serialization mismatches inside tool definitions and tool call arguments (empirically verified, fix planned)**: The JSON blob produced by `marshalWithSpaces(tool)` differs from what HuggingFace's `tojson` override produces. This has been empirically verified by running `apply_chat_template` with `transformers` v5.2.0 against the `Qwen/Qwen3.5-27B` tokenizer and comparing byte-for-byte. Three mismatches exist — see section 3.1 for the full verified analysis.

### 2.2 FIXED: System+tools ordering reversed for Qwen 3.5

**Was**: The `Qwen3CoderRenderer` renders system message first (`sb.WriteString(systemMessage)` at line 88), then tools (lines 90-136) — correct for Qwen3-Coder, wrong for Qwen 3.5.

**Now**: The dedicated `Qwen35Renderer` emits tools block first (lines 90-99), then appends system message content after `</IMPORTANT>` (lines 100-107). Test `TestQwen35RendererUsesXMLToolCallingFormat` explicitly asserts `systemIdx > toolsIdx`.

This matters for attention patterns — tools-first means the model's attention sees the tool schema before the system instruction, which is how it was trained.

### 2.3 FIXED: No image/vision support for Qwen 3.5

**Was**: Commit `fbae6976` rewired `"qwen3.5"` from `Qwen3VLRenderer` (which had `renderContent()` with image support) to `Qwen3CoderRenderer` (which has **no** `useImgTags` field, **no** `renderContent()` method, and **zero** references to `message.Images`). Images sent to a vision-capable Qwen 3.5 GGUF were silently discarded.

**Now**: The dedicated `Qwen35Renderer` has its own `renderContent()` method (lines 47-62) with `useImgTags` field, inserting `[img-N]` tags or `<|vision_start|><|image_pad|><|vision_end|>` tokens. The renderer is constructed with `useImgTags: RenderImgTags` at `renderer.go:60`.

Qwen 3.5 27B is a **native multimodal early-fusion model** — it has a 27-layer vision encoder (1152 hidden size, 16 heads, patch size 16) built directly into its architecture, confirmed by `vision_config` in the [official config.json](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json). llama.cpp (`d969e933`) has full vision support via `tools/mtmd/models/qwen3vl.cpp` (193 lines). The official Qwen 3.5 template handles image items with `<|vision_start|><|image_pad|><|vision_end|>` and video items with `<|vision_start|><|video_pad|><|vision_end|>`. Both upstream Ollama's `Qwen35Renderer` and the fork support images but not videos (line 58: `// TODO: support videos`).

**CAVEAT — image placement is wrong in all Ollama vision models, not just Qwen 3.5.** Every image in `renderContent()` is prepended before `content.Content` (line 48 explicitly comments: *"This assumes all images are at the front of the message"*). The `api.Message` struct (`api/types.go:211`) stores images as a flat `Images []ImageData` array with zero positional information — there is no way to know where in the text each image originally appeared. All three vision-capable renderers in the entire Ollama codebase — `Qwen35Renderer`, `Qwen3VLRenderer`, and `LFM2Renderer` — have this same front-prepend behavior. No Ollama model places images at their correct inline positions. The official Jinja2 template handles inline image positioning natively because the HuggingFace `messages` format embeds images as `{"type": "image"}` items interleaved with `{"type": "text"}` items within the `content` array — the template's `render_content` macro iterates this mixed array and emits vision tokens at their exact positions. Ollama's API collapses this into `Content string` + `Images []ImageData`, destroying the positional relationship. Fixing this requires changing the `api.Message` struct — which is Ollama's public HTTP API — to support interleaved content items, then updating all renderers, the runner's `[img-N]` regex splitter, and every client. This is not a Qwen 3.5-specific bug; it is a structural limitation of Ollama's API design that affects every vision model.

For text-only Unsloth GGUFs (851 tensors, no `vision.block_count`), the model layer correctly sets `Vision = nil` at `model.go:572` and returns `ErrNoVisionModel` at line 298-299 if images are sent. The renderer is irrelevant for text-only usage.

### 2.4 FIXED: Default system message injected when not provided

**Was**: The `Qwen3CoderRenderer` injects `"You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."` when tools are present but no system message is given (line 84-85). The Qwen3-Coder official template genuinely does this; the Qwen 3.5 official template does not.

**Now**: The dedicated `Qwen35Renderer` has no default system message injection. When tools are present but no system message exists, the system block contains only the tools text and instructions.

### 2.5 PARTIALLY FIXED: `</think>` rendering for empty reasoning — `message.Thinking != ""` sub-condition removed, but `isThinking` gate remains (BUG)

**Was**: The `Qwen3CoderRenderer` only emits think blocks when `message.Thinking != ""`:
```go
if isThinking && message.Thinking != "" && i > lastQueryIndex {
```

**Now**: The dedicated `Qwen35Renderer` removed the `message.Thinking != ""` sub-condition — empty reasoning now correctly produces `<think>\n\n</think>\n\n`:
```go
if isThinking && i > lastQueryIndex {
    sb.WriteString(imStartTag + message.Role + "\n<think>\n" + contentReasoning + "\n</think>\n\n" + content)
}
```

**REMAINING BUG — the `isThinking` gate itself is wrong.** The official Qwen 3.5 template (line 100: `{%- if loop.index0 > ns.last_query_index %}`) emits `<think>` blocks for ALL assistant messages after `lastQueryIndex`, **regardless of `enable_thinking`**. The `enable_thinking` flag only controls the prefill at the very end of the prompt (lines 147-153): `<think>\n` when thinking, `<think>\n\n</think>\n\n` when not. Historical messages always get their `<think>` blocks.

This is a **two-part bug** affecting both fork (`qwen35.go`) and upstream (`qwen35.go`), at identical line numbers:

1. **Line 143**: `if isThinking && i > lastQueryIndex` — the `isThinking &&` prefix should be removed. When `think: false` → `isThinking = false` → the condition fails → the `else` branch (line 146) renders the assistant message WITHOUT `<think>` blocks, silently discarding `contentReasoning`.

2. **Line 65**: `splitQwen35ReasoningContent()` — `if isThinking && messageThinking != ""` gates extraction of the `messageThinking` field on `isThinking`. When `isThinking = false`, the function skips the `messageThinking` field entirely, falls through to content-parsing (lines 69-77) which looks for `</think>` in `content` — but when reasoning is stored in the separate `messageThinking` field (not embedded in content), the search finds nothing. Historical reasoning is silently lost.

**When this matters**: A multi-turn conversation where earlier turns used `think: true` (producing messages with `Thinking` field populated), then the user sends a new request with `think: false`. The official template still renders `<think>reasoning</think>` for those historical messages — the model sees the full conversation context. The fork and upstream silently drop the reasoning from history, presenting the model with a conversation that looks like the assistant never thought — a distribution shift from training.

llama.cpp (`d969e933`) handles this correctly because it executes the Jinja2 template directly, which has no `isThinking` gate on history rendering.

**Note on Qwen3-Next-Thinking**: The `Qwen3CoderRenderer`'s conditional `message.Thinking != ""` gating is actually **correct for Qwen3-Next-Thinking** models, whose official template ([`Qwen/Qwen3-Next-80B-A3B-Thinking`](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking/raw/main/tokenizer_config.json), line 44) only emits think blocks for non-last messages when `reasoning_content` is non-empty. The two model families (Qwen 3.5 and Qwen3-Next-Thinking) were trained on contradictory thinking block semantics. Having separate renderers solves this cleanly — `Qwen35Renderer` uses unconditional think blocks, `Qwen3CoderRenderer` uses conditional. If a future `"qwen3-coder-thinking"` routing is needed, `Qwen3CoderRenderer`'s existing thinking code is ready.

### 2.6 FIXED: `lastQueryIndex` doesn't check for `<tool_response>` wrappers

**Was**: The `Qwen3CoderRenderer`'s `lastQueryIndex` loop (`qwen3coder.go:149-155`) simply finds the last `role == "user"` message with no content inspection.

**Now**: The dedicated `Qwen35Renderer` implements the `multiStepTool` / `<tool_response>` filtering pattern (lines 114-127) matching the official Qwen 3.5 Jinja2 template exactly:

```go
multiStepTool := true
lastQueryIndex := len(messages) - 1
for i := len(messages) - 1; i >= 0; i-- {
    message := messages[i]
    if multiStepTool && message.Role == "user" {
        content, _ := r.renderContent(message, 0)
        content = strings.TrimSpace(content)
        if !(strings.HasPrefix(content, "<tool_response>") && strings.HasSuffix(content, "</tool_response>")) {
            multiStepTool = false
            lastQueryIndex = i
        }
    }
}
```

This walks messages backwards, and for each `role == "user"` message: renders its content, trims whitespace, checks whether it is entirely wrapped in `<tool_response>...</tool_response>`. If yes — skip it (it's a tool result, not a real user query). The first non-tool-response user message found is `last_query_index`.

Without this filtering, in a multi-step agentic tool-calling conversation where the client sends tool results as `role: "user"` messages with `<tool_response>` content, the `Qwen3CoderRenderer` would set `lastQueryIndex` to the last tool-response message, causing ALL thinking traces from the assistant's tool-calling responses to be stripped.

### 2.7 FIXED: Prefill triggers on assistant messages with tool calls

See section 1.1 for the full analysis. The fork's `Qwen35Renderer` has the fix at line 136:
```go
prefill := lastMessage && message.Role == "assistant" && len(message.ToolCalls) == 0
```

Upstream Ollama's `Qwen35Renderer` lacks this guard.

### What the dedicated `Qwen35Parser` provides

The `Qwen35Parser` has a 3-state thinking machine (`CollectingThinking` → `ThinkingDoneEatingWhitespace` → `CollectingContent`) that extracts thinking content before delegating post-`</think>` content to an embedded `Qwen3CoderParser` (zero-value struct, `hasThinkingSupport: false`, `defaultThinking: false` — the inner parser only handles XML tool call parsing, not thinking). Key properties:

- **Tool calls inside `<think>` blocks are treated as thinking text**, not parsed as tool calls. Content emitted during `CollectingThinking` state becomes `qwen35EventThinkingContent`, which is accumulated in `thinkingSb` and never passed to `toolParser.Add()`. Only `qwen35EventContent` (emitted in `CollectingContent` state) reaches the inner `Qwen3CoderParser`.
- **Leading `<think>` tag stripped at most once** via `maybeConsumeLeadingThinkOpenTag()`. Some model checkpoints re-emit the `<think>` tag even though the prompt already prefilled it.
- **Streaming-safe partial tag overlap detection** with trailing whitespace withholding — the `eat()` method uses the same `overlap()` and `trailingWhitespaceLen()` helpers from `parsers.go` that `Qwen3CoderParser` uses.
- **Assistant prefill detection**: When `lastMessage` is a prefilled assistant message (non-empty content), the parser starts in `CollectingContent` state, skipping thinking extraction.
- **`HasToolSupport()` returns `true`, `HasThinkingSupport()` returns `true`**.
- The parser is **byte-for-byte identical** to upstream Ollama's implementation. 12 test functions (11 in `qwen35_test.go` + 1 integration routing test `TestQwen35ParserRespectsNoThink` in `qwen3_test.go:211`) cover all edge cases including streaming split `</think>` tags, tool calls inside thinking blocks, whitespace eating between `</think>` and content, truncated thinking without close tags, and `think: false` passthrough.

---

## Part 3: Shared Issues (Neither Fork Nor Upstream Ollama Fixes)

### 3.1 JSON serialization mismatches — HTML escaping, key ordering, and compact spacing (EMPIRICALLY VERIFIED, FIX PLANNED)

**Fault: Both upstream Ollama (`82848a78`) and the fork. Affects every model that uses `tojson` in its training template — not just Qwen 3.5. The fork will fix this.**

llama.cpp (`d969e933`) gets all three right because it executes the Jinja2 template directly via HuggingFace's template engine, producing the same JSON the model was trained on.

**How we verified this**: We ran `apply_chat_template` with `transformers` [v5.2.0](https://github.com/huggingface/transformers/tree/v5.2.0) against the official `Qwen/Qwen3.5-27B` tokenizer (downloaded from [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/tokenizer_config.json)) and compared the rendered output byte-for-byte against what the fork's `Qwen35Renderer` produces for the same tool and conversation. We also ran Go programs using Ollama's exact `api.Tool`, `api.ToolFunction`, `api.ToolFunctionParameters`, `api.ToolProperty`, and `api.ToolPropertiesMap` types to verify the actual serialization behavior. Every claim below is backed by empirical output, not inference from documentation.

**The root cause is Go's `json.Marshal`**. HuggingFace `transformers` overrides Jinja2's built-in `tojson` filter at [`chat_template_utils.py:463-466`](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/utils/chat_template_utils.py#L463-L466) with a custom implementation:

```python
def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
```

This is registered at [line 474](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/utils/chat_template_utils.py#L474): `jinja_env.filters["tojson"] = tojson`. When the Qwen 3.5 template calls `{{ tool | tojson }}`, only the `x` argument is passed — all others use defaults: `ensure_ascii=False`, `sort_keys=False`, `separators=None` (which in Python 3 defaults to `(', ', ': ')`).

This is NOT the same as stock Jinja2's `tojson`. Stock Jinja2 ([`jinja2/utils.py:htmlsafe_json_dumps`](https://github.com/pallets/jinja/blob/3.1.6/src/jinja2/utils.py)) uses `json.dumps` with the environment policy `{'sort_keys': True}`, then runs `.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026').replace("'", '\\u0027')` — alphabetical key sorting plus HTML escaping. HuggingFace's override does neither. The model was trained with HuggingFace's override, so that is the ground truth.

**Bug A — HTML escaping (HIGH impact for coding tools):**

Go's `json.Marshal` HTML-escapes `<`, `>`, `&` to `\u003c`, `\u003e`, `\u0026` by default. This is a [deliberate Go standard library design decision](https://github.com/golang/go/issues/8592) intended for safely embedding JSON in HTML `<script>` tags — a concern that has zero relevance to Ollama, where JSON is serialized into a chat template prompt fed to a neural network with no browser involved. HuggingFace's `tojson` outputs literal `<`, `>`, `&`. Stock Jinja2's `tojson` also HTML-escapes (for the same HTML-embedding reason), but HuggingFace explicitly overrode it because LLM chat templates are not HTML.

Empirically verified: for a tool description `"Returns temperature in <fahrenheit> & <celsius>."`:
```
HuggingFace tojson:      ...Returns temperature in <fahrenheit> & <celsius>....       (379 bytes total)
Go marshalWithSpaces:    ...Returns temperature in \u003cfahrenheit\u003e \u0026 \u003ccelsius\u003e....  (404 bytes total)
```

Each `<` becomes 6 bytes (`\u003c`) instead of 1, producing entirely different tokens. The fix is straightforward: `json.NewEncoder` with `SetEscapeHTML(false)`. Ollama's own `LFM2Renderer` already does this at [`model/renderers/lfm2.go:55-56`](https://github.com/ollama/ollama/blob/82848a78/model/renderers/lfm2.go#L55-L56).

**Bug B — Key ordering at `ToolFunctionParameters` level (MEDIUM impact):**

Empirically verified by running Go's `json.Marshal` on the actual Ollama `api.Tool` struct types and comparing against HuggingFace's `get_json_schema()` output:

| Nesting level | Go struct field order | HuggingFace `get_json_schema` / OpenAI convention | Match? |
|---|---|---|---|
| `Tool` | `type`, `function` | `type`, `function` | **Yes** |
| `ToolFunction` | `name`, `description`, `parameters` | `name`, `description`, `parameters` | **Yes** |
| `ToolFunctionParameters` | `type`, `required`, `properties` | `type`, `properties`, `required` | **No — `required`/`properties` swapped** |
| `ToolProperty` | `type`, `description`, `enum` | `type`, `description`, `enum` | **Yes** |
| `ToolPropertiesMap` (property names) | insertion order (orderedmap) | insertion order | **Yes** |

The only structural key ordering mismatch is at the `ToolFunctionParameters` level: Go always outputs `"required"` before `"properties"` (struct field order in `api/types.go:486-491`) regardless of what order the API client originally sent, because Go's `json.Marshal` uses struct tag order and re-serialization destroys the original JSON key ordering. Both HuggingFace's `get_json_schema()` function (at [`chat_template_utils.py:210-213`](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/utils/chat_template_utils.py#L210-L213)) and the [OpenAI function calling convention](https://platform.openai.com/docs/guides/function-calling) put `properties` before `required`. This is a positional encoding mismatch on every tool-using prompt.

**Additional key ordering concern for `any`-typed fields**: Ollama's `ToolProperty.Items` and `ToolFunctionParameters.Defs` are typed as Go `any`. When JSON is unmarshaled into these fields, nested objects become `map[string]any`, and Go's `json.Marshal` sorts `map[string]any` keys **alphabetically** — not by insertion order. This affects tools with array parameters whose `items` contain nested object schemas, or tools using JSON Schema `$defs`. These nested levels silently get their keys reordered on every roundtrip through Go's JSON serializer.

The fix for the `ToolFunctionParameters` level: swap the `Required` and `Properties` field declarations in `api/types.go` so `Properties` comes before `Required`. For the `any`-typed fields, the fix requires either using an orderedmap for deserialization or a custom JSON marshaler that preserves insertion order.

**Bug C — Compact separators in `formatToolCallArgument` (MEDIUM impact):**

HuggingFace's `tojson` uses Python's default separators `(', ', ': ')` — JSON with spaces after `:` and `,`. The existing `marshalWithSpaces` function at `json.go:8-45` correctly handles this for tool **definitions** in `Qwen35Renderer`. But `formatToolCallArgument` at `qwen3coder.go:195-217` (shared by `Qwen35Renderer`, `Qwen3CoderRenderer`, and all other renderers for rendering nested map/list values inside XML `<parameter>` tags) still uses raw `json.Marshal` — compact separators AND HTML escaping.

Empirically verified for a nested tool call argument `{"unit": "<celsius>", "verbose": true}`:
```
HuggingFace tojson|safe:      {"unit": "<celsius>", "verbose": true}     ← spaced, literal <
Go formatToolCallArgument:    {"unit":"\u003ccelsius\u003e","verbose":true}  ← compact, HTML-escaped
```

Both HTML escaping (Bug A) and compact spacing apply here. The model was trained on the spaced, non-escaped form.

**Which models are affected:** Every model in Ollama whose official HuggingFace training template uses `{{ tool | tojson }}` for tool definitions or `{{ args_value | tojson | safe }}` for tool call arguments — virtually all tool-calling models. For Qwen 3.5 specifically, Bug A affects tool definitions (the JSON blob inside `<tools>`) and Bug C affects tool call argument rendering (nested values inside `<parameter>` tags). Bug B affects tool definitions only.

**Concrete byte-level comparison for the complete tool JSON** (this is the single differing line between the fork's rendered output and the official template):
```
HuggingFace: ..."parameters": {"type": "object", "properties": {"location": ..., "unit": ...}, "required": ["location"]}...
Fork (Go):   ..."parameters": {"type": "object", "required": ["location"], "properties": {"location": ..., "unit": ...}}...
                                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                   required/properties swapped (Go struct field order)
```

**The fix (planned for fork):** Two changes needed:
1. **`marshalWithSpaces` in `json.go`**: Replace the inner `json.Marshal` call with `json.NewEncoder` using `SetEscapeHTML(false)` (fixes Bug A). The spacing logic already handles Bug C for this code path. For Bug B, swap the `Required` and `Properties` field declarations in `api/types.go:486-491`.
2. **`formatToolCallArgument` in `qwen3coder.go:195-217`**: Replace `json.Marshal` with the same `SetEscapeHTML(false)` encoder plus spaced separators (fixes Bugs A and C for tool call arguments).

### 3.2 `lastQueryIndex` initialization edge case

**Fault: Both upstream Ollama (`82848a78`) and the fork.** Neither validates the "no user messages" edge case that the official Qwen 3.5 template explicitly rejects with `raise_exception('No user query found in messages.')`.

Both initialize `lastQueryIndex := len(messages) - 1`. If there are zero user messages (e.g., `[system, assistant]`), it stays at the last index. The check `i > lastQueryIndex` is never true, so no assistant message gets `<think>` blocks. Conversations without user messages are extremely unusual in practice.

**Additional finding**: Upstream Ollama has an inconsistency between its two `lastQueryIndex` implementations. `Qwen35Renderer` (`qwen35.go:121`) calls `strings.TrimSpace(content)` before checking tool-response prefixes, but `Qwen3VLRenderer` (`qwen3vl.go:66`) does NOT trim. The fork's `Qwen35Renderer` trims (matching upstream Ollama), and the fork's `Qwen3VLRenderer` does not trim (also matching upstream Ollama — the file is untouched). Zero practical impact since Ollama's protocol uses `role: "tool"` for tool responses, not `role: "user"` with `<tool_response>` content.

### 3.3 BOS/EOS logging bug at `vocabulary.go:57`

```go
// Line 57 — Should be v.EOS, not v.BOS
if len(ids) > 0 && slices.Contains(v.BOS, ids[len(ids)-1]) {
    slog.Warn("adding eos token to prompt which already has it", "id", v.EOS)
}
```

**Why it is purely cosmetic**: The `if` block containing `slog.Warn` only controls the warning message. The actual `append(ids, v.EOS[0])` at line 61 runs **unconditionally** — the EOS token is always added correctly regardless of the bug.

**Why it never fires for Qwen 3.5**: `AddEOS` defaults to `false` for `qwen3next` (`model.go:686`: `AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false)`), so the entire EOS block at lines 55-62 is never entered.

Both codebases have this bug. Zero priority for either.

### 3.4 *(Merged into section 3.1 — HTML escaping is Bug A in the verified JSON serialization analysis.)*

---

## Remaining Action Items for the Fork

| Priority | Item | Effort | Section | Status |
|----------|------|--------|---------|--------|
| **P0** | Fix HTML escaping in `marshalWithSpaces` (`json.go`) and `formatToolCallArgument` (`qwen3coder.go:195-217`) | Small — replace `json.Marshal` with `json.NewEncoder` + `SetEscapeHTML(false)` in both functions. Reference: Ollama's own `LFM2Renderer` at [`lfm2.go:55-56`](https://github.com/ollama/ollama/blob/82848a78/model/renderers/lfm2.go#L55-L56). | 3.1 Bug A | **OPEN** — empirically verified: Go outputs `\u003c` where model was trained on literal `<`. |
| **P1** | Fix `required`/`properties` key ordering in `ToolFunctionParameters` — **CAUTION: shared struct change** | **Medium-High** — NOT a simple field swap. `ToolFunctionParameters` in `api/types.go:486-491` is a shared type used by ALL tool-calling renderers (DeepSeek, GLM, Cogito, Olmo3, Nemotron, etc.). Python's `tojson` preserves dict insertion order, which varies by template. Swapping fields to match Qwen 3.5 could make ordering WORSE for other models. Requires either per-model HuggingFace template verification or renderer-local marshaling. See `prioritized_research_topics_and_action_items.md` Section A.1 for full analysis. | 3.1 Bug B | **OPEN** — empirically verified for Qwen 3.5; other models unverified. **DO NOT change shared struct without per-model verification.** |
| **P1** | Fix compact separators in `formatToolCallArgument` | Small — apply the same `marshalWithSpaces`-style spacing to `formatToolCallArgument` at `qwen3coder.go:195-217`. Currently uses raw `json.Marshal` (compact `{k:v}`) where model was trained on spaced `{k: v}`. | 3.1 Bug C | **OPEN** — empirically verified. |
| **P1** | Fix `isThinking` gate on history `<think>` block rendering (two-part fix) | Tiny — remove `isThinking &&` from line 143 condition, remove `isThinking &&` from line 65 condition in `splitQwen35ReasoningContent()`. Both in `qwen35.go`. | 2.5 | **OPEN** — both fork and upstream have this bug. Official template (line 100) has no `enable_thinking` gate on history rendering. |
| **P2** | Add a dedicated test in `qwen35_test.go` for the prefill bug fix | Small — test where last message is `{Role: "assistant", ToolCalls: [...]}`, assert `<\|im_end\|>` IS emitted and `<\|im_start\|>assistant\n<think>\n` IS appended | 1.1 | **OPEN** — the fork's single differentiating line (`qwen35.go:136`: `&& len(message.ToolCalls) == 0`) currently has no dedicated test in the qwen35 test suite; it is only tested indirectly via `qwen3coder_test.go:327-376` |

All other previously-open items (P0: JSON tool definitions, P0: tools-first ordering, P2: image support, P3: default system message, P3: unconditional think blocks, P3: lastQueryIndex tool_response filtering) are now **RESOLVED** by the dedicated `Qwen35Renderer`/`Qwen35Parser`. The remaining template fidelity gaps are: the `isThinking` gate on history think blocks (section 2.5), and the JSON serialization bugs (Bugs A/B/C in section 3.1).

---

## What Upstream Ollama Should Fix (for reference — useful for PRs and issue reports)

| Priority | Bug | Impact | Who else gets it right | Section |
|----------|-----|--------|----------------------|---------|
| **P0** | **Prefill detection treats assistant+toolcalls as prefill** — ALL 3 renderers (`qwen35.go:136`, `qwen3coder.go:140`, `qwen3vl.go:82`) use `prefill := lastMessage && message.Role == "assistant"` without `&& len(message.ToolCalls) == 0` | Corrupted prompt structure for ANY agentic workflow where client sends assistant tool-call message last — unclosed `<\|im_end\|>`, missing generation prompt | **Fork**: fixed. **llama.cpp**: N/A (uses `add_generation_prompt` parameter, no conflation) | 1.1 |
| **P0** | **`mropeInterleaved` defaults to `false`** — wrong for ALL third-party Qwen 3.5 GGUFs (Unsloth, bartowski, llama.cpp converter) | Silent data corruption: RoPE type 8 (MROPE) instead of 40 (IMROPE) in every full-attention layer. Model loads and runs but positional encoding is wrong. Users blame model quality. | **Fork**: architecture-based default. **llama.cpp**: hardcoded per arch (`LLAMA_ROPE_TYPE_IMROPE`) | 1.3 |
| **P1** | **`repeat_last_n` API parameter silently ignored** in Go runner — accepted by API, never wired to sampler | Dead API field. Users setting `repeat_last_n: 128` get no effect. | **Fork**: wired as 9th parameter to `NewSampler`. **llama.cpp**: `penalty_last_n` correctly wired | 1.5 |
| **P1** | **`</think>` not always closed** in `Qwen3VLRenderer` — gated on `content != ""` | Unclosed `<think>` tag for thinking-only responses in `qwen3-vl-instruct` and `qwen3-vl-thinking` models | **Fork**: always closes. **llama.cpp**: always closes via Jinja2 template | 1.2 |
| **P1** | **History `<think>` blocks incorrectly gated on `isThinking`** — `Qwen35Renderer` (`qwen35.go:143`) uses `if isThinking && i > lastQueryIndex`, but the official template (line 100) has NO `enable_thinking` gate on history rendering. Two-part bug: line 143 (rendering) + line 65 in `splitQwen35ReasoningContent()` (extraction). When `think: false`, all historical reasoning is silently dropped. | Distribution shift in multi-turn thinking/non-thinking conversations. Model sees history without `<think>` blocks that were present during training. | **Fork**: same bug. **llama.cpp**: correct for history rendering (executes Jinja2 template directly), but has its own generation prompt bug — `enable_thinking` is NOT passed to Jinja context (`chat.cpp:1527`); post-processing produces `<think>\n</think>` (3 tokens) instead of the official `<think>\n\n</think>\n\n` (6 tokens), a sequence never seen in training. See `fork_vs_latest_upstream_and_llama_cpp.md` Section 6.4 Bug 1. | 2.5 |
| **P0** | **JSON serialization wrong in ALL tool-using renderers** — three bugs: (A) Go's `json.Marshal` HTML-escapes `<>&` to `\u003c\u003e\u0026` where model was trained on literal characters, (B) `ToolFunctionParameters` struct outputs `required` before `properties` where model was trained on `properties` before `required`, (C) `formatToolCallArgument` uses compact separators where model was trained on spaced JSON. Empirically verified against `transformers` [v5.2.0](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/utils/chat_template_utils.py#L463-L466) — HuggingFace's `tojson` override uses `json.dumps(sort_keys=False, ensure_ascii=False)`, NOT stock Jinja2's `tojson` which sorts alphabetically and HTML-escapes. | Distribution shift on every tool-using prompt. Bug A affects any tool description containing `<>&` (6 bytes per char instead of 1). Bug B is a positional encoding mismatch at the parameters level (but **CAUTION**: changing the shared `ToolFunctionParameters` struct without per-model verification could make ordering worse for non-Qwen models — see `prioritized_research_topics_and_action_items.md` Section A.1). Bug C affects nested map/list values inside tool call arguments. | **Fork**: same bugs. **llama.cpp**: correct for JSON serialization (executes Jinja2 directly), but has its own `tojson` float precision flaw at `value.cpp:1290` (6 significant digits vs Python's ~17; `1.0` → `1`) | 3.1, 2.1 |
| **P1** | **Penalty sampler feeds prompt tokens into penalty window** via public `Accept()` — `repeatPenalty`, `presencePenalty`, AND `frequencyPenalty` all corrupted | With any penalty > 0/1.0, tool names, JSON tokens, and previous results in the prompt are penalized during generation. Forces `repeat_penalty: 1.0` default (disabled). Makes `presence_penalty` and `frequency_penalty` unsafe for agentic use. | **Fork**: private `recordToken()`, generated-only ring buffer. **llama.cpp**: same bug — `server-context.cpp:208-215` feeds all prompt tokens into `common_sampler_accept()`, penalty sampler at `llama-sampler.cpp:2622-2656` uses ring buffer with no prompt/generated distinction. Masked by llama.cpp's `repeat_penalty=1.0` default at `common.h:199`. | 1.4 |
| **P2** | **GGUF converter conditional KV emission** — `rope.mrope_interleaved` and `ssm.v_head_reordered` only written when `true` | Third-party GGUFs that omit these ollama-internal keys fall through to wrong defaults (see P0 mropeInterleaved above) | **Fork**: unconditional emission. **llama.cpp**: N/A (doesn't use these keys) | 1.7 |
| **P2** | **`repeat_penalty` defaults to `1.0`** — penalty system is a no-op with defaults | `repeat_penalty: 1.0` means the multiplicative penalty math is identity. Combined with prompt-token contamination, upstream cannot safely raise this above 1.0. | **Fork**: also defaults to `1.0` (reverted from `1.1` to avoid affecting other models; the fork's architecture makes it safe to raise per-model via Modelfile). **llama.cpp**: same `1.0` default | 1.6 |

---

## Part 4: Architecture Notes

### Recurrent layer inference — fork's formula matches llama.cpp, upstream Ollama adds extra indirection

The fork's 4-line interval computation (`model.go:513-517`):
```go
interval := int(c.Uint("full_attention_interval", 4))
isRecurrent = make([]bool, numLayers)
for i := range numLayers {
    isRecurrent[i] = (i+1)%interval != 0
}
```

This formula `(i+1)%interval != 0` is identical to llama.cpp's (`d969e933`) `((i + 1) % full_attn_interval != 0)` at `llama-model.cpp:2526` (for `LLM_ARCH_QWEN35`). llama.cpp reads `full_attention_interval` from GGUF metadata via `ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false)` with a default of 4, then populates `hparams.recurrent_layer_arr[i]` with the same formula.

Upstream Ollama's `inferRecurrentLayers()` (52 lines, upstream `model.go:497-549`) is closer to llama.cpp's complete behavior:
1. **Primary**: If `headCountKV` is a per-layer array with a mix of zero (recurrent) and non-zero (full attention) values, use it directly.
2. **Fallback**: If `headCountKV` is uniform (scalar broadcast from third-party GGUFs), fall through to the `full_attention_interval` computation. Also validates edge cases (`interval > numLayers`, `interval <= 0`).

For all real-world GGUFs (Ollama-converted and third-party), both produce identical results:
- **Ollama-converted GGUFs**: Per-layer `headCountKV` array triggers upstream Ollama's primary path, fork's interval computation produces the same pattern.
- **Third-party GGUFs**: Scalar `headCountKV` triggers upstream Ollama's fallback, which uses the same `(i+1)%interval != 0` formula as the fork.

The difference matters only if someone produces a GGUF with a non-uniform `headCountKV` array that deviates from the interval pattern — upstream Ollama would honor the array, the fork would override it. The fork's approach is actually **more robust for third-party GGUFs** because it has zero dependency on `headCountKV` — it goes straight to the interval formula.

### `SetInplace` → balanced concat tree

Both codebases now use a balanced binary concat tree (`GGML_OP_CONCAT`) for chunk assembly in `deltaNetChunked`. O(log N) graph depth, supported on all backends in all GGML versions. The fork originally used `ggml_set_inplace()` — the same approach llama.cpp uses in its canonical GatedDeltaNet implementation (`delta-net-base.cpp:262` in `d969e933`). That approach is algorithmically correct but unsupported on Ollama's vendored GGML's Metal/Vulkan backends (Ollama's GGML has `GGML_OP_SET_ROWS` but not `GGML_OP_SET` on Metal and Vulkan). The fork now matches upstream Ollama commit `3490e959`.

### Vision packaging: Ollama vs llama.cpp

Ollama bundles vision tensors into the main GGUF. The official `qwen3.5:27b` blob has 1307 tensors with `vision.block_count = 27` — the 27-layer vision encoder is embedded alongside the 64-layer text model. llama.cpp uses a separate mmproj GGUF loaded via `--mmproj`. Unsloth follows llama.cpp's convention: text-only main GGUFs (851 tensors, no `vision.block_count`) plus separate `mmproj-{BF16,F16,F32}.gguf` files.

Ollama's Go engine explicitly rejects split vision: `llm/server.go:149-152` checks `if len(projectors) == 0` and returns `errors.New("split vision models aren't supported")` when projectors are present. When pulling from HuggingFace (e.g., `FROM hf.co/unsloth/Qwen3.5-27B-GGUF:Q4_K_XL`), Ollama auto-downloads the mmproj alongside the main GGUF, classifies it as `"application/vnd.ollama.image.projector"` at `create.go:653`, adds it to `ProjectorPaths` at `images.go:319`, and then the Go engine rejects the entire load. The custom Modelfile works around this by using `FROM /tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf` (manually downloaded text-only GGUF) — the mmproj is never pulled, `ProjectorPaths` stays empty, and the model loads successfully as text-only.

For text-only Unsloth GGUFs, the model layer behaves correctly: `model.go:572` checks `vision.block_count` (absent → 0), skips `NewVisionModel`, sets `Vision = nil`. If a user somehow sends an image, `EncodeMultimodal` returns `ErrNoVisionModel` at line 298. The renderer is irrelevant for text-only usage.

Upstream Ollama has secondary architectural issues (not bugs, but design debt):
- The chat handler (`routes.go`) never checks `CapabilityVision` before accepting images — the error surfaces late in `EncodeMultimodal()` rather than at the API boundary. By contrast, llama.cpp's server validates `allow_image` at the API entry point (`server-common.cpp:970`: `"image input is not supported - hint: you may need to provide the mmproj"`) and advertises vision capability in its `/models` metadata (`server-context.cpp:3414`: `"modalities": {"vision": true}`). This missing capability advertisement is why [issue #14508](https://github.com/ollama/ollama/issues/14508) exists — third-party tools (Dify) can't detect that `qwen3.5:27b` supports vision.
- `sched.go:448` forces `numParallel = 1` for all `qwen35` models. This restriction exists because the hybrid recurrent architecture (GatedDeltaNet) is not safe with concurrent requests (ref: [issue #4165](https://github.com/ollama/ollama/issues/4165)), not because of vision — the same restriction applies to non-vision architectures like `lfm2`.

---

## Part 5: Full Capability Utilization — What It Takes to Run Qwen 3.5 27B Q4 at Its Full Potential

Unsloth's UD-Q4_K_XL quantization uses dynamic mixed precision — sensitive layers get higher quantization types (Q6_K, Q8_0) while less sensitive layers get Q4_K. This is the best 4-bit quantization available for consumer hardware. But the quantization quality is wasted if the inference engine doesn't handle everything else correctly. Here is the complete picture.

### What already works correctly in the fork

These require no action — they are verified working:

- **Flash attention** for FullAttention layers (every 4th layer). Enabled by default for `qwen3next` architecture (`fs/ggml/ggml.go:873,901`). Uses `ggml_flash_attn_ext()` with F32 precision. The 48 GatedDeltaNet recurrent layers (75% of the model) do not use flash attention — they use their own chunked linear attention with delta state updates, which is architecturally correct.
- **KV cache quantization** via `OLLAMA_KV_CACHE_TYPE=q8_0` (or `q4_0`). Only applies to the 16 FullAttention layers' causal KV cache — the recurrent layers' conv+delta state always stays F32, which is correct (quantizing recurrent state causes severe quality loss). Requires flash attention to be enabled (it is by default). Supported types checked at `fs/ggml/ggml.go:848-854`.
- **131K context window**. `num_ctx 131072` works — the code at `llm/server.go:167-171` clamps to the GGUF metadata's `context_length`, which the Unsloth GGUF correctly sets. No artificial cap in the engine.
- **YaRN RoPE scaling** at `model/models/qwen3next/model.go:77-84`. Correct attention factor formula: `1.0 / (1.0 + 0.1*ln(ropeScale))`. Parameters flow through to GGML's `ggml_rope_multi`/`ggml_rope_ext` at `ml/backend/ggml/ggml.go:1550-1582`.
- **MRoPE (Multi-Resolution RoPE)** with `mrope_sections` from GGUF metadata (`model.go:536-545`). The fork's `mropeInterleaved` architecture-based default (section 1.3) ensures the correct IMROPE type 40 is used.
- **Grammar/GBNF constrained decoding** at `sample/samplers.go:212-251`. Wraps llama.cpp's GBNF engine 1:1. Guarantees valid JSON for tool calls. Works with qwen3next.
- **Mmap with smart defaults** at `llm/server.go:672-694`. Model weights are memory-mapped. Disabled automatically for partial GPU offload on Metal, Windows CUDA perf, and Linux when model exceeds free memory.
- **Vision (unified GGUF from ollama.com/library only)**. The model code at `model/models/qwen3next/model.go:237-238,297-367,611-617` has full vision support — 27-layer vision encoder, `EncodeMultimodal`, grid-based image processing with spatial merge. Works with Ollama's own `qwen3.5:27b` blob (1307 tensors, `vision.block_count=27`). Images from ALL messages in the conversation are correctly handled — `prompt.go:122-127` collects `ImageData` from every message that survived truncation, the renderer emits `[img-N]` tags with incrementing IDs across all messages (`qwen35.go:129-132`), and the runner splits the entire prompt on `\[img-(\d+)\]` regex (`ollamarunner/runner.go:236-238`) and processes every match via `EncodeMultimodal`. Multi-turn vision conversations work. Does NOT work with Unsloth text-only GGUFs (851 tensors, no vision weights) — see "Vision with Unsloth third-party GGUFs" below. **One minor template fidelity gap:** the renderer puts all image tokens before text within each message (`qwen35.go:50-56`, comment: "assumes all images are at the front"), while the official Qwen 3.5 Jinja2 template renders image tokens inline where they appear in the content list. This only matters for multi-image messages where the user interleaves text and images (e.g., "What's in [img] compared to [img]?"). This is an upstream Ollama design choice, not a fork issue — the fork's renderer is byte-identical to upstream here.
- **Penalty sampling** — fork's private `recordToken()` ring buffer excludes prompt tokens. All 3 penalty types (repeat, presence, frequency) correctly operate on generated tokens only.

### What needs fixing (correctness)

| Issue | Impact | Fix | Section |
|-------|--------|-----|---------|
| **JSON serialization mismatches** in all tool serialization — Go's `json.Marshal` uses struct field order, HTML-escapes `<>&` to `\u003c\u003e\u0026`, and uses compact separators in `formatToolCallArgument`. HuggingFace's `tojson` override uses Python dict insertion order, literal `<>&`, and spaced separators. | Distribution shift on every tool-using prompt. Token positions for tool schemas are shuffled. Tool descriptions containing `<>&` produce entirely different tokens (`\u003c` = 6 bytes vs `<` = 1 byte). | Replace `marshalWithSpaces` with function using `SetEscapeHTML(false)`, spaced separators, and correct key ordering. Planned for fork. | 3.1, 3.4, 2.1 |
| **Prefill test gap** — the fork's single differentiating line (`qwen35.go:136`) has no dedicated test | Risk of regression. | Add test to `qwen35_test.go`. | 1.1 |

### What's missing (performance)

#### Speculative decoding — the single biggest performance opportunity (REQUIRES RESEARCH)

**What it is:** A small "draft" model (e.g., Qwen 3.5 0.6B or 1.5B at Q8_0) generates N candidate tokens cheaply, then the full 27B model verifies them all in a single forward pass. Correct tokens are accepted, wrong ones are rejected and regenerated. For 4-bit models on consumer hardware, generation is overwhelmingly memory-bandwidth-bound (you're moving 17.6 GB of weights through memory for every single token). Speculative decoding amortizes this cost across multiple tokens.

**Expected speedup:** 2-4x on token generation, depending on draft model quality and acceptance rate. This would transform the user experience from "usable but slow" to "genuinely fast" on a single GPU.

**Current status in Ollama:** Not supported at all. Zero infrastructure — no draft model loading, no accept/reject logic, no token tree verification. The only trace is dead code in llama.cpp's vendored `common.h:236-245` (`common_params_speculative` structs), never called by Ollama.

**llama.cpp status:** llama.cpp has speculative decoding via `llama-speculative` with configurable parameters (`n_max=16` draft tokens, `p_min=0.75` acceptance threshold). **However, it is unclear whether llama.cpp's speculative decoding works with hybrid recurrent architectures like qwen3next.** The GatedDeltaNet recurrent layers maintain internal state that depends on the exact token sequence — if draft tokens are rejected, the recurrent state must be rolled back, which is fundamentally different from rolling back a KV cache. This is a deep architectural question that requires studying:
1. Whether llama.cpp's speculative decoding handles recurrent state rollback for Mamba/Jamba/qwen3next models
2. Whether the draft model needs to be the same architecture (hybrid) or can be a pure transformer
3. Whether the recurrent state can be checkpointed/restored efficiently without replaying the entire sequence
4. What the interaction is between speculative decoding and the `numParallel=1` constraint

**Bottom line:** This requires significant research before any implementation work. The payoff is enormous but the complexity for hybrid architectures is unknown.

#### Parallel requests — hard-locked to 1

`sched.go:448` forces `numParallel = 1` for all qwen3next models. The blocker is the GatedDeltaNet recurrent layers: `deltanet.go:80-87` requires all sequences in a batch to have the same token count (`ErrUnsupportedBatchLayout`). The 16 FullAttention layers could theoretically batch across sequences (standard causal KV cache supports it), but the 48 recurrent layers cannot.

**Gotcha:** This is not a bug — it's an architectural constraint of hybrid recurrent models. Ollama, llama.cpp, and every other inference engine have this limitation for Mamba/Jamba/GatedDeltaNet architectures. Two possible approaches exist (run recurrent layers per-sequence while batching attention, or pad sequences to equal length) but both have significant complexity and questionable net benefit.

#### Fused GatedDeltaNet kernel

The current Go-level implementation at `deltanet.go:463-495` loops over chunks in Go, creating ~10 GGML graph nodes per chunk per layer. For a 512-token prompt evaluation: 64 layers × 75% recurrent = 48 recurrent layers × 8 chunks = 384 chunk iterations, each creating multiple graph nodes. This is a graph compilation bottleneck.

GGML already has relevant CUDA kernels (`gla.cu` for Gated Linear Attention, `ssm-conv.cu` for SSM convolution, `solve_tri.cu` for triangular solve, `cumsum.cu` for cumulative sum) but the Go-level deltanet implementation constructs the computation from generic GGML ops (Mulmat, Slice, Concat) rather than calling these specialized ops.

**Gotcha:** Fusing this into a single GGML op requires CUDA/Metal/Vulkan kernel work. The `gla.cu` kernel in the vendored GGML is the closest match, but GatedDeltaNet's delta rule update (`S = α ⊙ S + β^T v` with short convolution preprocessing) differs from standard GLA (`S = S + k^T v`). This would need a custom CUDA kernel or significant adaptation of `gla.cu`.

#### Batch size

Default is 512 (`api/types.go:1074`). llama.cpp defaults to 2048. For 131K context prompt evaluation, a larger batch size reduces the number of forward passes. The GatedDeltaNet's chunk size is 64 (`deltanet.go:12`), so batch sizes should be multiples of 64 for optimal padding.

**Quick fix:** Add `PARAMETER num_batch 2048` to the Modelfile. No code change needed. Verify memory fits first — larger batch sizes require more intermediate tensor memory during prompt evaluation.

### What's missing (capabilities)

#### Vision with Unsloth third-party GGUFs (REQUIRES SIGNIFICANT RESEARCH)

**Two completely different vision situations exist — do not confuse them:**

**Situation A: Official `ollama.com/library/qwen3.5:27b` (unified GGUF) — VISION LOADS AND RUNS, BUT IMAGE PLACEMENT IS WRONG.**
Ollama's own model blob contains all 1307 tensors (851 text + 456 vision) in a single GGUF with `vision.block_count=27`. The model loads with `Vision = qwen3vl.NewVisionModel(c)` at `model.go:611-617`. `EncodeMultimodal` at `model.go:297-319` processes images through the 27-layer vision encoder with grid-based spatial merge. The `Qwen35Renderer` emits `[img-N]` tags (when `useImgTags=true`, which is the case in the server at `routes.go:108`), and the runner at `ollamarunner/runner.go:236-284` splits the prompt on those tags and splices in image embeddings. Images from all messages in the conversation work — not just the most recent message. Verified: `prompt.go:122-127` collects `ImageData` from every surviving message, the renderer's `imageOffset` counter at `qwen35.go:129-132` increments across all messages, and the runner's regex at line 236 matches all `[img-N]` tags anywhere in the prompt. **The fork does not break any of this** — the `Qwen35Renderer` is byte-identical to upstream except the prefill fix at line 136, which does not touch vision code. However, every image is prepended to the front of its message (see section 2.3 caveat) — a structural Ollama-wide limitation affecting all vision models, not a Qwen 3.5-specific bug.

**Situation B: Unsloth text-only GGUF (`qwen3.5-custom` Modelfile) — NO VISION, BY DESIGN.**
The Unsloth UD-Q4_K_XL GGUF has 851 tensors and no `vision.block_count` metadata. At `model.go:611`, `c.Uint("vision.block_count", 0)` returns 0, so `NewVisionModel` is never called and `Vision = nil`. If a user sends an image via the API, `EncodeMultimodal` returns `ErrNoVisionModel` at `model.go:298-299`. The renderer still emits `[img-N]` tags into the prompt text, but the runner's `multimodalProcessor` type assertion at `ollamarunner/runner.go:233` evaluates to `visionModel = false`, so it takes the text-only path at line 241 and the `[img-N]` tags remain as literal text (never processed). This is correct behavior for text-only usage.

**The trap: using `FROM hf.co/unsloth/Qwen3.5-27B-GGUF:Q4_K_XL` in a Modelfile.** When Ollama pulls from HuggingFace instead of using a local file, it auto-discovers the mmproj file alongside the main GGUF, classifies it as `"application/vnd.ollama.image.projector"` at `create.go:653`, and adds it to `ProjectorPaths` at `images.go:319`. The Go engine then hits the check at `llm/server.go:148-152`: `if len(projectors) == 0 { ... } else { err = errors.New("split vision models aren't supported") }`. Since `qwen3next` is in the `OllamaEngineRequired()` list (`fs/ggml/ggml.go:294`), there is no fallback to the llama.cpp runner — **the entire model load fails**. The custom Modelfile at `/home/user/Desktop/vibe_web_terminal/ollama-models/qwen3.5-custom.Modelfile` works around this by using `FROM /tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf` (manually downloaded text-only GGUF) — the mmproj is never pulled, `ProjectorPaths` stays empty, and the model loads successfully as text-only.

**What it would take to enable vision with Unsloth GGUFs — and why this requires significant research:**

1. **The Go engine loads a single GGUF file.** The entire model loading pipeline (`ml/backend/ggml/ggml.go`) assumes one GGUF. The mmproj is a second GGUF with its own tensor namespace. Merging them at load time requires understanding how tensor names map between the two files (the main GGUF uses `v.*` prefixed tensor names for vision, the mmproj may use different naming conventions depending on the converter).

2. **Unsloth's mmproj may not match Ollama's expected vision tensor layout.** Ollama's `qwen3vl.NewVisionModel(c)` at `model.go:611` reads vision config from GGUF metadata (`vision.block_count`, `vision.embedding_length`, etc.) and expects tensors at specific GGUF names (e.g., `v.blk.{n}.attn_qkv.weight`). Unsloth's mmproj was generated by llama.cpp's `convert_hf_to_gguf.py`, which may use different tensor names (e.g., `v.enc.blk.{n}.attn_qkv.weight` or `mm.{n}.weight`). A tensor name mapping layer would be needed.

3. **The vision encoder is 27 layers (456 tensors).** This is not a small projector — it's a full ViT with 1152 hidden size, 16 heads, and patch size 16. Loading it separately, mapping its tensors, and integrating it with the Go engine's single-GGUF buffer allocation is non-trivial.

4. **Quantization compatibility.** The mmproj files come in BF16/F16/F32. The text model is Q4_K_XL. The Go engine needs to handle mixed precision across the two GGUFs, placing vision tensors on the correct device (GPU) with the correct quantization type.

5. **An alternative: merge the GGUFs offline.** Instead of fixing the Go engine, one could write a tool that merges the Unsloth text GGUF + mmproj into a single unified GGUF matching Ollama's expected format. This avoids engine changes but requires understanding both GGUF layouts and producing correct metadata. This is probably the more practical path but still requires research into tensor name mappings and metadata compatibility.

6. **Testing.** Even after loading, the vision pipeline (`EncodeMultimodal` at `model.go:297-319`, `PostTokenize` at `model.go:321-367`) was tested with Ollama's own unified GGUFs. Third-party vision tensors from a different converter may have subtle differences (weight ordering, normalization conventions, patch embedding layout) that produce wrong results without error.

**Bottom line:** This is a research project, not a quick fix. The minimum viable approach (offline GGUF merge tool) still requires deep understanding of both Ollama's and llama.cpp's GGUF vision tensor conventions. The full approach (Go engine multi-GGUF loading) is a significant architectural change to Ollama's model loading pipeline. Vision works perfectly with the official `ollama.com/library` model — the gap is exclusively about third-party GGUFs that split text and vision into separate files.

### What doesn't matter

- **Token healing / BPE anti-corruption**: Not implemented in Ollama or llama.cpp. Handled at the application layer. Ollama tokenizes the full rendered prompt as a single string, so there are no internal boundary artifacts — the boundaries are between special tokens (`<|im_start|>`, `<|im_end|>`) which are in the vocabulary and tokenized correctly.
- **Continuous batching**: Blocked by `numParallel=1` (same root cause as parallel requests). Even if unblocked, the GatedDeltaNet layers would need all sequences padded to equal length, negating the efficiency benefit.
- **Dynamic NTK RoPE scaling**: Would allow exceeding the 131K training context length. Neither Ollama nor llama.cpp implement this at runtime. The static YaRN implementation is sufficient since the Modelfile's `num_ctx 131072` matches the training length.
- **Separate K/V cache quantization types**: Ollama uses one type for both K and V (`llama/llama.go:139-140`). llama.cpp supports separate `cache_type_k`/`cache_type_v`. Low impact — the difference between Q8_0 for both vs Q8_0-K/Q4_0-V is minor compared to the recurrent state dominating memory anyway.
- **Lazy expert loading for MoE layers**: Qwen 3.5 27B has MoE in its architecture (128 experts, 8 active per token). Lazy loading could reduce memory by only loading active experts on demand. Neither Ollama nor llama.cpp implement this. The 4-bit quantization already makes the full model fit in memory on consumer GPUs, so the motivation is limited.
