# Ollama Fork vs Upstream: Precise Analysis

Fork: `BigBIueWhale/ollama` @ `9ec17fc1` (7 commits atop `v0.17.4` merge) + dedicated Qwen 3.5 renderer/parser.
Upstream: `ollama/ollama` master @ `82848a78` (commit message: `model: fix renderer and parser for qwen3.5 (#14605)`).
llama.cpp: `ggml-org/llama.cpp` master @ `d969e933` (commit hash: `d969e933e172821b4519f66aa4b660bc0846b320`) — the canonical C++ inference engine. Ollama vendors GGML (the tensor math library) from llama.cpp but reimplements everything else in Go: chat template rendering is hardcoded Go renderers instead of llama.cpp's Jinja2 engine, sampling is a Go rewrite instead of llama.cpp's `llama-sampler.cpp`, and the model runner is `ollamarunner` instead of llama.cpp's server. This means llama.cpp bugs in GGML affect Ollama, but llama.cpp's correct Jinja2 template handling does NOT help Ollama — Ollama must reimplement the same logic in Go renderers/parsers.
Official reference: Qwen 3.5 Jinja2 template from `Qwen/Qwen3.5-35B-A3B` ([publicly accessible on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/tokenizer_config.json)). **Also verified** against `Qwen/Qwen3.5-27B` ([publicly accessible](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/tokenizer_config.json)) — both repos contain byte-identical `chat_template` fields.
Qwen3-Coder reference: `Qwen/Qwen3-Coder-30B-A3B-Instruct` Jinja2 template ([verified via public access](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)) — confirms XML tool definitions, system-first ordering. Upstream Ollama's `Qwen3CoderRenderer` faithfully matches it.

---

## Scorecard

| Area | Fork | Upstream Ollama @ `82848a78` | Winner |
|------|------|----------|--------|
| Qwen 3.5 renderer (tool defs, ordering, images, think blocks) | Dedicated `Qwen35Renderer` with prefill bug fix | Dedicated `Qwen35Renderer` without prefill bug fix | **Fork** |
| Qwen 3.5 parser (thinking extraction, tool call delegation) | Dedicated `Qwen35Parser` (byte-identical to upstream) | Dedicated `Qwen35Parser` | Tie |
| Prefill guard (`len(message.ToolCalls)==0`) | Fixed in **all 3** renderers (`qwen35.go`, `qwen3coder.go`, `qwen3vl.go`) | Broken in **all 3** renderers | **Fork** |
| `</think>` closure in `Qwen3VLRenderer` | Always closes `</think>` tag | Conditionally gates on `content != ""` — leaves unclosed `<think>` tag | **Fork** |
| `mropeInterleaved` default for third-party GGUFs | Architecture-aware default (`c.Architecture() != "qwen3next"`) | Hardcoded `false` — **silently wrong RoPE type** for all third-party Qwen 3.5 GGUFs | **Fork** |
| `repeat_last_n` API parameter | Wired through to Go sampler via `NewSampler` parameter | Silently ignored (hardcoded `DefaultPenaltyLookback = 64`) | **Fork** |
| Penalty sampling architecture (repeat, presence, frequency — all three share one token window) | Private `recordToken()` inside `Sample()`, ring buffer, excludes prompt tokens — matches original Keskar et al. 2019 CTRL paper. All 3 penalty types benefit. | Public `Accept()` called externally by runner, feeds prompt tokens into penalty window — all 3 penalty types corrupted | **Fork** |
| `repeatPenalty` default | `1.1` (penalty system actually functions) | `1.0` (penalty system is a no-op; llama.cpp comments: `// 1.0 = disabled`) | **Fork** |
| Third-party GGUF compatibility | Architecture-based `mropeInterleaved` default (the critical fix) + shared `ssm_dt.bias` alt tag + `Validate()` + nil checks | Hardcoded `false` mropeInterleaved default (wrong for third-party qwen35 GGUFs) + shared `ssm_dt.bias` alt tag + `Validate()` + nil checks | **Fork** (due to mropeInterleaved; other fixes are shared or equivalent) |
| GGUF converter KV emission | Unconditional (always writes `rope.mrope_interleaved` and `ssm.v_head_reordered`) | Conditional (only writes when `true` — third-party GGUFs that omit the key fall through to wrong default) | **Fork** |
| `SetInplace` → balanced concat tree | Fixed (matches upstream Ollama commit `3490e959`) | Fixed | Tie |
| Tokenizer performance | Binary search prompt truncation, `strings.Contains` early-out, stack buffer BPE merge | None of these optimizations | **Fork** |
| `formatToolCallArgument` JSON spacing | Compact JSON via `json.Marshal` | Compact JSON via `json.Marshal` | Neither (both wrong vs official template's spaced JSON) |

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

### 2.1 FIXED: Tool definition format — Qwen 3.5 was trained on JSON, not XML

**Was**: The fork routed `"qwen3.5"` through `Qwen3CoderRenderer` (at `renderer.go:59-60`), which emits XML tool definitions (`<function><name>...`). The Qwen 3.5 official template uses JSON tool definitions (`{{ tool | tojson }}`).

**Now**: The dedicated `Qwen35Renderer` uses `marshalWithSpaces()` (at `json.go:8-45`) for JSON tool definitions, matching the official Qwen 3.5 Jinja2 template. Test `TestQwen35RendererUsesXMLToolCallingFormat` explicitly asserts the `<tools>` section is present with JSON content.

**Background**: Alibaba Qwen publishes **two different official templates** with a hybrid format split that is easy to miss:

| Template | Tool definitions | Tool calls (model output) | System+tools order |
|----------|-----------------|--------------------------|-------------------|
| **Qwen3-Coder** (`Qwen/Qwen3-Coder-30B-A3B-Instruct`) | XML `<function><name>...` | XML `<tool_call><function=...>` | System first, tools second |
| **Qwen 3.5** (`Qwen/Qwen3.5-27B`) | JSON `{{ tool | tojson }}` | XML `<tool_call><function=...>` **(same!)** | Tools first, system appended after |

The tool call format is character-for-character identical. Only the tool *definition* rendering and system message ordering differ. This is **Alibaba Qwen's confusing design decision** — it's the reason the fork author made the wrong call in commit `fbae6976`. llama.cpp (`d969e933`) sidesteps this entirely by executing the Jinja2 template directly (`chat.cpp:631-639`).

**Template fidelity note on JSON key ordering**: Jinja2's `tojson` calls `json.htmlsafe_dumps()` which uses Python's default separators `(', ', ': ')` — **spaced JSON** — and **sorts keys alphabetically** (confirmed: `"function"` before `"type"` at top level, `"description"` before `"name"` inside `function`). The fork's `marshalWithSpaces` post-processes Go's `json.Marshal` output to add spaces after `:` and `,` — matching the separator style. However, Go's `json.Marshal` uses **struct declaration order** for keys, not alphabetical: it emits `{"type": "function", "function": {...}}` while `tojson` emits `{"function": {...}, "type": "function"}`. This key ordering difference is low-impact (LLMs handle JSON key order variability well) but worth noting for exact template fidelity. The HTML-escaping behavior is identical: both Go's `json.Marshal` and `tojson` produce `\u003c`/`\u003e`/`\u0026` for `<`/`>`/`&`.

### 2.2 FIXED: System+tools ordering reversed for Qwen 3.5

**Was**: The `Qwen3CoderRenderer` renders system message first (`sb.WriteString(systemMessage)` at line 88), then tools (lines 90-136) — correct for Qwen3-Coder, wrong for Qwen 3.5.

**Now**: The dedicated `Qwen35Renderer` emits tools block first (lines 90-99), then appends system message content after `</IMPORTANT>` (lines 100-107). Test `TestQwen35RendererUsesXMLToolCallingFormat` explicitly asserts `systemIdx > toolsIdx`.

This matters for attention patterns — tools-first means the model's attention sees the tool schema before the system instruction, which is how it was trained.

### 2.3 FIXED: No image/vision support for Qwen 3.5

**Was**: Commit `fbae6976` rewired `"qwen3.5"` from `Qwen3VLRenderer` (which had `renderContent()` with image support) to `Qwen3CoderRenderer` (which has **no** `useImgTags` field, **no** `renderContent()` method, and **zero** references to `message.Images`). Images sent to a vision-capable Qwen 3.5 GGUF were silently discarded.

**Now**: The dedicated `Qwen35Renderer` has its own `renderContent()` method (lines 47-62) with `useImgTags` field, inserting `[img-N]` tags or `<|vision_start|><|image_pad|><|vision_end|>` tokens. The renderer is constructed with `useImgTags: RenderImgTags` at `renderer.go:60`.

Qwen 3.5 27B is a **native multimodal early-fusion model** — it has a 27-layer vision encoder (1152 hidden size, 16 heads, patch size 16) built directly into its architecture, confirmed by `vision_config` in the [official config.json](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json). llama.cpp (`d969e933`) has full vision support via `tools/mtmd/models/qwen3vl.cpp` (193 lines). The official Qwen 3.5 template handles image items with `<|vision_start|><|image_pad|><|vision_end|>` and video items with `<|vision_start|><|video_pad|><|vision_end|>`. Both upstream Ollama's `Qwen35Renderer` and the fork support images but not videos (line 58: `// TODO: support videos`).

For text-only Unsloth GGUFs (851 tensors, no `vision.block_count`), the model layer correctly sets `Vision = nil` at `model.go:572` and returns `ErrNoVisionModel` at line 298-299 if images are sent. The renderer is irrelevant for text-only usage.

### 2.4 FIXED: Default system message injected when not provided

**Was**: The `Qwen3CoderRenderer` injects `"You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."` when tools are present but no system message is given (line 84-85). The Qwen3-Coder official template genuinely does this; the Qwen 3.5 official template does not.

**Now**: The dedicated `Qwen35Renderer` has no default system message injection. When tools are present but no system message exists, the system block contains only the tools text and instructions.

### 2.5 FIXED: `</think>` rendering for empty reasoning differs from training template

**Was**: The `Qwen3CoderRenderer` only emits think blocks when `message.Thinking != ""`:
```go
if isThinking && message.Thinking != "" && i > lastQueryIndex {
```

**Now**: The dedicated `Qwen35Renderer` **always** emits the think block for eligible assistant messages, even with empty reasoning:
```go
if isThinking && i > lastQueryIndex {
    sb.WriteString(imStartTag + message.Role + "\n<think>\n" + contentReasoning + "\n</think>\n\n" + content)
}
```

The official Qwen 3.5 template unconditionally emits `<think>\n{reasoning}\n</think>\n\n` for all assistant messages after `last_query_index`. An empty reasoning produces `<think>\n\n</think>\n\n`. llama.cpp (`d969e933`) handles this correctly via the Jinja2 template.

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

### 3.1 `formatToolCallArgument` uses compact JSON for object/array parameters

**Fault: Both upstream Ollama (`82848a78`) and the fork.** The `marshalWithSpaces` function already exists (at `json.go:8-45`) and is used for tool **definitions**, but neither codebase uses it for tool call **arguments**. llama.cpp (`d969e933`) gets this right because the Jinja2 template uses `tojson` for tool call arguments too (e.g., `args_value | tojson | safe` for mappings/sequences), producing spaced JSON automatically.

**File**: `model/renderers/qwen3coder.go:235-257` (shared by all renderers via tool call rendering)

When a tool call parameter value is a map, slice, or array, `formatToolCallArgument` uses `json.Marshal()`, which produces **compact JSON**:
```
Go:       {"key":"value","nested":[1,2,3]}
```

The official template uses Jinja2's `tojson` (via `args_value | tojson | safe` for mappings/sequences), which produces **spaced JSON**:
```
Python:   {"key": "value", "nested": [1, 2, 3]}
```

This was verified empirically: Python's `json.dumps()` default separators are `(', ', ': ')`, and `tojson` uses these defaults. Simple string/number arguments render identically. Low practical impact — complex object/array tool call arguments are rare.

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

### 3.4 HTML-escaping in JSON — confirmed non-issue

Both Go's `json.Marshal` and Jinja2's `tojson` (which calls `json.htmlsafe_dumps`) produce the same `\u003c`/`\u003e`/`\u0026` unicode escapes for `<`, `>`, `&`. Verified empirically. The only real format differences between Go and Python JSON serialization are spacing (section 3.1) and key ordering (section 2.1 note).

---

## Remaining Action Items for the Fork

| Priority | Item | Effort | Section | Status |
|----------|------|--------|---------|--------|
| **P1** | Use `marshalWithSpaces` for object/array values in `formatToolCallArgument` | Small — change `json.Marshal` → `marshalWithSpaces` in `qwen3coder.go:235-257` | 3.1 | **OPEN** |
| **P2** | Add a dedicated test in `qwen35_test.go` for the prefill bug fix | Small — test where last message is `{Role: "assistant", ToolCalls: [...]}`, assert `<\|im_end\|>` IS emitted and `<\|im_start\|>assistant\n<think>\n` IS appended | 1.1 | **OPEN** — the fork's single differentiating line (`qwen35.go:136`: `&& len(message.ToolCalls) == 0`) currently has no dedicated test in the qwen35 test suite; it is only tested indirectly via `qwen3coder_test.go:327-376` |

All other previously-open items (P0: JSON tool definitions, P0: tools-first ordering, P2: image support, P3: default system message, P3: unconditional think blocks, P3: lastQueryIndex tool_response filtering) are now **RESOLVED** by the dedicated `Qwen35Renderer`/`Qwen35Parser`.

---

## What Upstream Ollama Should Fix (for reference — useful for PRs and issue reports)

| Priority | Bug | Impact | Who else gets it right | Section |
|----------|-----|--------|----------------------|---------|
| **P0** | **Prefill detection treats assistant+toolcalls as prefill** — ALL 3 renderers (`qwen35.go:136`, `qwen3coder.go:140`, `qwen3vl.go:82`) use `prefill := lastMessage && message.Role == "assistant"` without `&& len(message.ToolCalls) == 0` | Corrupted prompt structure for ANY agentic workflow where client sends assistant tool-call message last — unclosed `<\|im_end\|>`, missing generation prompt | **Fork**: fixed. **llama.cpp**: N/A (uses `add_generation_prompt` parameter, no conflation) | 1.1 |
| **P0** | **`mropeInterleaved` defaults to `false`** — wrong for ALL third-party Qwen 3.5 GGUFs (Unsloth, bartowski, llama.cpp converter) | Silent data corruption: RoPE type 8 (MROPE) instead of 40 (IMROPE) in every full-attention layer. Model loads and runs but positional encoding is wrong. Users blame model quality. | **Fork**: architecture-based default. **llama.cpp**: hardcoded per arch (`LLAMA_ROPE_TYPE_IMROPE`) | 1.3 |
| **P1** | **`repeat_last_n` API parameter silently ignored** in Go runner — accepted by API, never wired to sampler | Dead API field. Users setting `repeat_last_n: 128` get no effect. | **Fork**: wired as 9th parameter to `NewSampler`. **llama.cpp**: `penalty_last_n` correctly wired | 1.5 |
| **P1** | **`</think>` not always closed** in `Qwen3VLRenderer` — gated on `content != ""` | Unclosed `<think>` tag for thinking-only responses in `qwen3-vl-instruct` and `qwen3-vl-thinking` models | **Fork**: always closes. **llama.cpp**: always closes via Jinja2 template | 1.2 |
| **P1** | **`formatToolCallArgument` uses compact JSON** for object/array parameters | Tool call arguments rendered as `{"key":"value"}` instead of `{"key": "value"}` — wrong vs official template's spaced JSON | **Fork**: same bug. **llama.cpp**: correct via `tojson` filter | 3.1 |
| **P1** | **Penalty sampler feeds prompt tokens into penalty window** via public `Accept()` — `repeatPenalty`, `presencePenalty`, AND `frequencyPenalty` all corrupted | With any penalty > 0/1.0, tool names, JSON tokens, and previous results in the prompt are penalized during generation. Forces `repeat_penalty: 1.0` default (disabled). Makes `presence_penalty` and `frequency_penalty` unsafe for agentic use. | **Fork**: private `recordToken()`, generated-only ring buffer. **llama.cpp**: same bug as upstream Ollama | 1.4 |
| **P2** | **GGUF converter conditional KV emission** — `rope.mrope_interleaved` and `ssm.v_head_reordered` only written when `true` | Third-party GGUFs that omit these ollama-internal keys fall through to wrong defaults (see P0 mropeInterleaved above) | **Fork**: unconditional emission. **llama.cpp**: N/A (doesn't use these keys) | 1.7 |
| **P2** | **`repeat_penalty` defaults to `1.0`** — penalty system is a no-op with defaults | `repeat_penalty: 1.0` means the multiplicative penalty math is identity. Combined with prompt-token contamination, upstream cannot safely raise this above 1.0. | **Fork**: defaults to `1.1` (safe because prompt tokens excluded). **llama.cpp**: same `1.0` default | 1.6 |

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
