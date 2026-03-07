# Grammar-Constrained Tool Call Generation: Implementation Plan

**Date:** 2026-03-07 (plan), 2026-03-08 (implementation complete)
**Fork:** `BigBIueWhale/ollama` @ commit `4044b63f` (11 commits atop the Ollama `v0.17.4` merge base `cc90a035`)
**Implementation commits:** `b1038b91` (Step 3 GBNF builder), `59d1b367` (header cleanup), `4044b63f` (Steps 1-2, 4-5 + renderer fix)
**Reference llama.cpp:** `ggml-org/llama.cpp` @ `a0ed91a` (local clone at `/tmp/llama-cpp-latest/`)
**Vendored llama.cpp in fork:** Effectively at GGML `a5bb8ba4` level (Ollama upstream commit `ef00199f`, with subsequent revert `b1fccabb` that rolled back a b7847 bump)

---

## 1. Why This Needs to Be Fixed

### 1.1 The Problem: Ollama Trusts the Model

Ollama's tool call parser is a free-form streaming state machine (`model/parsers/qwen3coder.go:150-317`). It receives the model's raw text output token-by-token after generation and parses it. The model's output is **untrusted** — it can generate anything — but the parser treats it as trusted input.

When the model misbehaves, the parser silently propagates malformed output as invalid `api.ToolCall` structs to the client:

1. **Hallucinated function name.** The model generates `<function=nonexistent_tool>`. The parser creates an `api.ToolCall` with `Function.Name = "nonexistent_tool"`. The tool matching loop at `qwen3coder.go:358` sets `matchedTool = nil`, so all parameters are parsed without type information (defaulting to raw strings at line 416). This phantom tool call propagates through the API to the client.

2. **Malformed XML.** The model generates garbage between `<tool_call>` and `</tool_call>`. The `transformToXML` regex partially matches, `xml.Unmarshal` may fail (line 349), and `parseToolCall` returns an error. The recovery path is unclear — the error propagates up through `Qwen3CoderParser.Add()` at line 84-85.

3. **Unclosed `<tool_call>` tag.** The parser's `CollectingToolContent` state (line 293) accumulates forever, waiting for `</tool_call>`. If the model stops (EOS) without closing the tag, the accumulated content is silently lost in the parser's buffer — never emitted as either a tool call or as content.

4. **Type-mismatched parameter values.** The schema says `"type": "integer"` but the model outputs `"not a number"`. `parseValue()` at line 404 tries `strconv.ParseInt` which fails, then falls back to returning the raw string at line 454. The caller receives a `string` where it expected an `int` — a silent type mismatch.

### 1.2 Why It Matters for Agentic Workflows

In agentic workflows where tool call results feed back into the conversation, a single malformed tool call cascades: the client receives a phantom function call, fails to execute it (or worse, misroutes it), sends back an error or garbage result, and the model's next turn is conditioned on this corrupted history. Each subsequent turn compounds the corruption.

Qwen 3.5 is a well-trained model that usually generates valid tool calls. But "usually" is not "always" — especially under:
- Adversarial prompts
- Long context degradation (131K context window)
- Low temperature with unlucky sampling
- Quantization artifacts (the Unsloth `UD-Q4_K_XL` is 4-bit, which has higher error rates than FP16)

### 1.3 llama.cpp Handles This Correctly

llama.cpp builds a PEG grammar from the declared tool schemas before generation begins (`chat.cpp:1555-1616` at reference commit `a0ed91a`). The grammar is converted to GBNF and applied as logit masks at every token step. The model literally cannot produce an invalid tool call — invalid tokens get `-inf` logits and are never sampled.

Key implementation details in llama.cpp:

- **Line 1580:** Each tool becomes a grammar rule: `"<function=" + p.literal(name) + ">\n"` — only declared function names are valid
- **Line 1592:** Each parameter becomes: `"<parameter=" + p.literal(param_name) + ">\n"` — only declared parameter names are valid
- **Line 1596-1600:** String parameters accept free text (`p.until_one_of(delimiters)`); non-string parameters are constrained to match their JSON schema via `p.schema(p.json(), rule_name + "-schema", param_schema)`
- **Line 1604:** Required parameters: `p.repeat(arg_rule, 1, 1)` (must appear exactly once); optional parameters: `p.repeat(arg_rule, 0, 1)` (zero or one times) — forces declaration-order sequential output
- **Line 1610-1613:** Parallel tool calls: `p.repeat(tool_call, min_calls, max_calls)` with `max_calls = -1` when parallel is enabled
- **Line 1612:** Complete tool call rule: `"<tool_call>\n" + tool_choice + "</tool_call>" + p.space()`
- **Line 1626:** Lazy activation: `grammar_lazy = true` when tool_choice is auto — grammar only constrains after trigger
- **Line 1637-1639:** Trigger on `<tool_call>` word — content before the trigger is completely unconstrained
- **Line 1628-1635:** Grammar construction uses `build_grammar()` callback API which internally calls `json_schema_to_grammar.cpp` for per-parameter JSON schema constraints

**No other model in Ollama has grammar-constrained tool call generation.** Not DeepSeek, not GLM, not Cogito — they all use the same trust-the-model free-form parsing approach. Implementing this for Qwen 3.5 makes it the first model in the fork with structurally robust tool calling.

---

## 2. What We Already Have (Infrastructure Audit)

The vendored llama.cpp in the fork already contains all the necessary C++ infrastructure. The Go wrapper and sampling pipeline already support grammar-constrained generation for structured outputs (JSON schema via `format`). The gap is specifically: no one has wired grammar constraints to tool calls.

### 2.1 `json-schema-to-grammar.cpp` — Already Vendored, Generic, Elegant

**File:** `llama/llama.cpp/common/json-schema-to-grammar.cpp` (1153 lines)
**Header:** `llama/llama.cpp/common/json-schema-to-grammar.h` (43 lines)

This is a pure JSON Schema to GBNF converter. It has **zero model-specific code** — no `if (model == ...)` anywhere. It's a single recursive visitor class (`common_schema_converter`) that walks a JSON Schema and emits GBNF grammar rules.

Architecture breakdown:

| Lines | Component | What It Does |
|-------|-----------|-------------|
| 1-42 | `build_repetition` | GBNF `{min,max}` repetition helper |
| 44-226 | `_build_min_max_int` | Integer range constraints → character class grammar |
| 228-257 | Built-in rules | Constant tables for `boolean`, `number`, `string`, `integer`, `null`, `uuid`, `date`, `time` |
| 308-535 | `_visit_pattern` | Regex pattern → GBNF conversion (~190 lines, the most complex part) |
| 545-604 | `_not_strings` | Trie-based "not these strings" rule for `additionalProperties` |
| 620-711 | `_build_object_rule` | JSON object with required/optional properties, recursive optional handling |
| 741-813 | `resolve_refs` | `$ref` resolution including remote URL fetch |
| 819-974 | `visit` | Main dispatcher — handles every JSON Schema keyword (`type`, `properties`, `oneOf`/`anyOf`/`allOf`, `enum`, `const`, `pattern`, `minimum`/`maximum`, `items`, `format`, etc.) |
| 1010-1120 | `resolves_to_string` | Type probing helper (used by `chat.cpp` to decide string vs JSON grammar) |
| 1122-1153 | Public API | `json_schema_to_grammar()` and `build_grammar()` with callback API |

The `build_grammar()` function (line 1137) provides a callback API with three operations:
- `add_rule(name, rule)` — add a custom GBNF rule
- `add_schema(name, schema)` — convert a JSON schema to GBNF rules and return the root rule name
- `resolve_refs(schema)` — resolve `$ref` references in a schema

This callback API is what llama.cpp's `chat.cpp` uses to compose custom XML wrapping rules with schema-derived parameter rules into a single unified grammar. It is the key to building our tool call grammar.

### 2.2 C Bridge — Already Partially Exposed

**File:** `llama/sampling_ext.cpp`
**Header:** `llama/sampling_ext.h`

Currently exposed:

| Function | What It Does | Status |
|----------|-------------|--------|
| `schema_to_grammar(json_schema, grammar_buf, max_len)` | Converts JSON schema → complete GBNF string | Works, used by structured outputs |
| `grammar_init(grammar, tokens, n_tokens, pieces, eog_tokens, n_eog_tokens)` | Creates `llama_grammar` from GBNF string | Works, but **hardcodes `lazy=false`** at line 99 |
| `grammar_apply(grammar, token_data_array)` | Masks logits for grammar-invalid tokens | Works |
| `grammar_accept(grammar, token_id)` | Advances grammar state after accepting a token | Works |
| `grammar_free(grammar)` | Frees grammar | Works |

The critical gap at `sampling_ext.cpp:99`:
```cpp
struct llama_grammar *g = llama_grammar_init_impl(
    nullptr, vocab, grammar, "root",
    false,       // lazy — HARDCODED to false
    nullptr, 0,  // trigger_patterns — NONE
    nullptr, 0   // trigger_tokens — NONE
);
```

The underlying `llama_grammar_init_impl` at `llama-grammar.h:175-184` fully supports all three parameters:
```cpp
struct llama_grammar * llama_grammar_init_impl(
    const struct llama_vocab * vocab,
    const struct ollama_vocab * ollama_vocab,
    const char * grammar_str,
    const char * grammar_root,
    bool lazy,                          // <-- need to expose
    const char ** trigger_patterns,     // <-- need to expose
    size_t num_trigger_patterns,
    const llama_token * trigger_tokens, // <-- need to expose
    size_t num_trigger_tokens);
```

### 2.3 Go Grammar Wrapper — Already Works

**File:** `llama/llama.go` — `NewGrammar` (line 715), `Apply` (line 752), `Accept` (line 784), `Free` (line 743)
**File:** `sample/samplers.go` — `GrammarSampler` (line 212), `NewGrammarSampler` (line 216)

The `GrammarSampler` is fully integrated into the sampling pipeline. In `Sample()` at lines 55-82:
1. Apply grammar to top candidate (fast path optimization)
2. If rejected, reset logits, apply grammar to all tokens, resample
3. Accept the chosen token into grammar state

### 2.4 Grammar Flow: Server → Runner — Already Wired

**Server side** (`llm/server.go`): `CompletionRequest` has a `Grammar string` field (line 1461). Currently populated only from `Format` (structured outputs) at lines 1546-1557.

**Runner side** (`runner/ollamarunner/runner.go`): Lines 879-888 create a `GrammarSampler` from `req.Grammar` when non-empty. Grammar is freed with `defer` at line 887.

The `Grammar` field is JSON-serialized and sent over HTTP from server to runner subprocess. No additional transport work needed.

### 2.5 Lazy Grammar in Vendored C++ — Fully Functional

**File:** `llama/llama.cpp/src/llama-grammar.cpp`

The grammar struct at `llama-grammar.h:135-161` has full lazy support:
```cpp
struct llama_grammar {
    // ...
    bool lazy = false;
    bool awaiting_trigger = false;
    std::string trigger_buffer;
    std::vector<token_pos> trigger_buffer_positions;
    std::vector<llama_token> trigger_tokens;
    std::vector<llama_grammar_trigger_pattern> trigger_patterns;
};
```

- `llama_grammar_apply_impl` (line 1260): When `awaiting_trigger` is true, returns immediately without masking any logits — generation is unconstrained
- `llama_grammar_accept_impl` (line 1306): When `awaiting_trigger` is true, checks each accepted token against trigger tokens and patterns. On trigger match, sets `awaiting_trigger = false` and replays buffered tokens into the grammar state (lines 1312-1363)

---

## 3. What llama.cpp Validates vs. What We Will and Will Not Validate

### 3.1 Constraint Comparison Table

| Constraint | llama.cpp (grammar) | Fork: grammar | Notes |
|------------|---------------------|---------------|-------|
| **XML structure** — `<tool_call>`, `</tool_call>`, `<function=...>`, `</function>`, `<parameter=...>`, `</parameter>` well-formed | **Yes** | **Yes** — GBNF literals | Grammar prevents malformed tags from ever being generated |
| **Function name** — only declared tool names | **Yes** — `p.literal(name)` alternation (`chat.cpp:1580`) | **Yes** — literal alternation in GBNF | Highest-value constraint. Prevents phantom function calls. |
| **Parameter name** — only declared param names | **Yes** — `p.literal(param_name)` per parameter (`chat.cpp:1592`) | **Yes** — literal alternation per tool | Prevents invented parameter names |
| **Lazy trigger activation** | **Yes** — `grammar_lazy = true`, trigger on `<tool_call>` word (`chat.cpp:1626-1639`) | **Yes** — vendored C++ fully supports this; needs C bridge exposure (~20 lines) | Without lazy, grammar would constrain ALL output including thinking/content |
| **Unclosed `<tool_call>` / premature EOS** | **Yes** — grammar only allows EOS after complete `</tool_call>` | **Yes** — `root ::= tool-call+` requires at least one complete `<tool_call>...</tool_call>` before EOS is allowed | The model cannot stop mid-tool-call. Grammar forces the full XML structure to completion before end-of-sequence is a valid token. |
| **Required parameters present** | **Yes** — `p.repeat(arg_rule, 1, 1)` forces exactly-once in declaration order (`chat.cpp:1604`) | **No** — not implementing | Forces fixed parameter ordering which may fight model's learned order |
| **Optional parameters** | **Yes** — `p.repeat(arg_rule, 0, 1)` | **No** — not implementing | Same ordering concern |
| **String parameter values** | **Unconstrained** — `p.until_one_of(delimiters)` accepts any text (`chat.cpp:1596-1597`) | **Unconstrained** — free text until `</parameter>` | Neither validates string content against patterns/minLength/maxLength/enum |
| **Non-string parameter values** (int, number, bool, object, array) | **Yes** — `p.schema()` converts JSON schema to GBNF (`chat.cpp:1599`), using `json-schema-to-grammar.cpp` | **No** — not implementing at grammar level | See Section 3.2 for why we skip this in grammar |
| **`</parameter>` closing tag** | **Optional** — `p.optional(p.tool_arg_close(...))` at `chat.cpp:1603` | **Match llama.cpp** — optional in grammar | Some model checkpoints omit closing parameter tags |
| **Parallel tool calls** | **Yes** — `p.repeat(tool_call, min, max)` (`chat.cpp:1610-1613`) | **Yes** — GBNF repetition rule | Both support parallel calls |

### 3.2 Why We Skip Non-String JSON Schema Constraints in the Grammar

llama.cpp constrains non-string parameter values (integers, booleans, objects, arrays) at the grammar level using its `json-schema-to-grammar.cpp` engine. This means an integer parameter can only produce digit characters, a boolean can only produce `true`/`false`, etc.

We will NOT replicate this because:

1. **The `schema_to_grammar` C bridge produces standalone complete grammars.** It returns a full GBNF grammar with its own `root ::= ...` rule. To use it for individual parameter values inside our larger XML-framing grammar, we would need to either merge multiple standalone GBNF rule sets (fragile — rule name collisions, root rule conflicts) or use the `build_grammar()` callback API.

2. **The `build_grammar()` callback API is not exposed to Go.** It's a C++ function that takes a `std::function` callback. Exposing it through CGo requires a C function pointer shim. This is doable (~50-80 lines of new C bridge code that takes tool definitions JSON, builds the grammar internally using `build_grammar()` + `add_schema()`, and returns the complete GBNF string) but adds complexity.

3. **Diminishing returns.** The two most damaging failure modes — hallucinated function names and hallucinated parameter names — are already blocked by the grammar. A parameter type mismatch is a tolerable imperfection; a phantom function call is not. For a well-trained model like Qwen 3.5, non-string type mismatches are rare in practice.

**Future option:** If we later want JSON schema constraints for non-string parameter values, the existing `tool_call_grammar_from_json()` in `sampling_ext.cpp` can be extended to call `builder.add_schema(name, param_schema)` for each non-string parameter instead of using the free-text `xml-arg-string` rule. ~20-30 lines. The `json-schema-to-grammar.cpp` code is already vendored and functional.

### 3.3 Summary

**What the fork WILL enforce at the grammar level (during generation — the model cannot produce these violations):**
- Function names are one of the declared tool names (literal alternation)
- Parameter names are one of the declared parameter names for that function (literal alternation)
- XML structure is well-formed (`<tool_call>`, `<function=...>`, `<parameter=...>` tags properly nested)
- Unclosed `<tool_call>` tags are impossible — `root ::= tool-call+` requires at least one complete `<tool_call>...</tool_call>` before end-of-sequence is a valid token. The model cannot stop mid-tool-call.
- Lazy activation: grammar only constrains output after `<tool_call>` trigger, leaving content/thinking unconstrained
- Parallel tool calls supported via GBNF repetition

**What the fork will NOT enforce (llama.cpp does, but too much work for the return):**
- JSON schema validation for non-string parameter values at the grammar level (integers producing only digits, booleans only `true`/`false`, etc.). Type mismatches are rare for well-trained models and are a tolerated tradeoff (see Section 3.2).
- Required parameter completeness at the grammar level (forcing exactly-once in declaration order). Forces fixed parameter ordering which may fight model's learned order.
- Required/optional parameter count constraints at the grammar level (same ordering concern).

---

## 4. What Happens When the Model Misbehaves — Error Handling Paths

### 4.1 Happy Path (Grammar Working Correctly)

1. Server builds GBNF grammar from tool definitions, sets `req.Grammar` with lazy trigger on `<tool_call>`
2. Runner creates `GrammarSampler` from the GBNF string
3. Model generates thinking and content freely — grammar is in `awaiting_trigger` state, `llama_grammar_apply_impl` returns immediately without masking logits
4. Model generates `<tool_call>` — grammar trigger fires, `awaiting_trigger` becomes `false`, grammar state replays the trigger tokens
5. From here, every token is constrained: only declared function names, only declared parameter names, well-formed XML tags
6. Model generates `</tool_call>` — grammar reaches a completion state, allows EOS
7. Parser receives well-formed tool call, parses it, returns valid `api.ToolCall` to client

### 4.2 Grammar Dead-End (Bug in Grammar Construction)

If the grammar has a bug (reaches a state with no valid continuations):

1. `grammar.Apply(tokens)` at `samplers.go:76` sets ALL token logits to `-inf`
2. `s.sample(tokens)` runs — `temperature()` divides `-inf` by temperature (still `-inf`), `softmax()` produces `NaN`, sum is `NaN`
3. Line 139: `math.IsNaN(float64(sum))` returns `true` → `errors.New("sample: logits sum to NaN, check model output")`
4. `Sample()` returns the error, which propagates up through the runner

**Previous state:** The `NaN` check existed but its error message was misleading ("logits sum to NaN, check model output") — it didn't mention grammar. This was accidental error detection, not intentional.

**Fixed (Step 5, 2026-03-08):** After `s.grammar.Apply(tokens)` at line 76, we now check if all tokens are `-inf` and return a clear error: `"grammar rejected all tokens — grammar may be malformed or the generation reached an impossible state"`. This catches the issue before it becomes a cryptic NaN and benefits all models using grammar-constrained generation.

### 4.3 Model Outputs Content Only (No Tool Call)

1. Grammar stays in `awaiting_trigger` state for the entire generation
2. `llama_grammar_apply_impl` returns immediately on every token — no logit masking
3. Model generates freely, produces content, hits EOS
4. Parser receives content, emits it as normal text
5. No tool calls produced — client gets a text response

This is the correct behavior when the model decides not to call any tools.

### 4.4 Grammar Conflicts with Format (Structured Outputs)

If the user sends both `tools` AND `format` (structured output), both would try to set `req.Grammar`. Currently `format` is handled at `llm/server.go:1539-1558` and tools are handled by the parser at `server/routes.go:2199-2212`.

**Fixed (Step 2, 2026-03-08):** When tools are present, the tool call grammar takes precedence. In `llm/server.go:Completion()`, the Format→Grammar conversion is now guarded by `if req.Grammar == "" && len(req.Format) > 0` — so it only builds a format grammar when no tool grammar was already set. The grammar includes a lazy trigger, so content before `<tool_call>` is unconstrained — the model can produce JSON-formatted content if it wants. If the user specifically needs grammar-constrained JSON AND grammar-constrained tool calls, that's a fundamentally different grammar (not supported).

### 4.5 Grammar Accepts Token, Then Grammar State Becomes Invalid

The `llama_grammar_accept_impl` at `llama-grammar.cpp:1372` has a hard assertion:
```cpp
GGML_ABORT("grammar error: end of grammar token received but grammar stack is not empty");
```

This fires if an EOS token is accepted while the grammar hasn't reached a completion state. In practice this means the generation tried to end in the middle of a tool call (e.g., after `<tool_call>` but before `</tool_call>`). The `GGML_ABORT` kills the process — this is a crash, not a graceful error.

**Mitigation:** A correctly-constructed grammar prevents this for the active (non-lazy) phase. Once the lazy trigger fires on `<tool_call>` and the grammar becomes active, the `root ::= tool-call+` rule requires at least one complete `<tool_call>...</tool_call>` before end-of-sequence is a valid token — `grammar_apply` sets EOS to `-inf` mid-tool-call, so the sampler cannot select it. (Before the trigger fires, the grammar is in `awaiting_trigger` state and follows a different code path in `llama_grammar_accept_impl` that does not check the grammar stack — so this assertion is not reached.) This assertion should only fire if the grammar itself has a construction bug, in which case the process crash is the correct signal.

---

## 5. Implementation Plan

All changes are Qwen-scoped except the thin C bridge extension (step 1-2), which is a generic improvement any model could use.

### Step 1: Extend the C Bridge (~36 lines) --- COMPLETED 2026-03-08

**Files:** `llama/sampling_ext.cpp`, `llama/sampling_ext.h`

Added `grammar_init_lazy` with a simplified signature — `lazy` is implicit (always `true` when calling this function), and `trigger_tokens` was omitted since we only use regex trigger patterns:
```c
struct llama_grammar *grammar_init_lazy(
    char* grammar,
    uint32_t* tokens, size_t n_tokens,
    const char** pieces,
    uint32_t* eog_tokens, size_t n_eog_tokens,
    const char** trigger_patterns, size_t n_trigger_patterns
);
```

Calls `llama_grammar_init_impl` with `lazy=true`, the trigger patterns array, and `nullptr`/`0` for trigger tokens. Includes null-grammar check, `try/catch` for exception safety, and proper `ollama_vocab` cleanup on failure.

> **Implementation note:** The original plan included `bool lazy` and `trigger_tokens` parameters. During implementation, these were dropped for simplicity: `lazy` is always `true` (the non-lazy path already exists via `grammar_init`), and trigger tokens are unnecessary since our trigger mechanism uses regex patterns exclusively.

### Step 2: Extend the Go Wrapper (~58 lines in llama.go, ~30 lines in samplers.go) --- COMPLETED 2026-03-08

**File:** `llama/llama.go`

Added `NewGrammarLazy(grammar, vocabIds, vocabValues, eogTokens, triggerPatterns)` and `ToolCallGrammarFromJSON(toolsJSON)`. Both `NewGrammar` and `NewGrammarLazy` delegate to a shared private `newGrammar(grammar, vocabIds, vocabValues, eogTokens, triggerPatterns)` — the branch between `grammar_init` and `grammar_init_lazy` is a single `if len(triggerPatterns) > 0` inside the shared function, eliminating all duplicated vocab marshaling code.

`ToolCallGrammarFromJSON` wraps the C++ `tool_call_grammar_from_json()` via CGo — uses a 64KB grammar buffer and 1KB error buffer, returns the GBNF string or an error with the C++ error code and message.

**File:** `sample/samplers.go`

Added `NewGrammarSamplerLazy(tok, grammarStr, triggerPatterns)`. Same consolidation pattern: both `NewGrammarSampler` and `NewGrammarSamplerLazy` delegate to `newGrammarSampler(tok, grammarStr, triggerPatterns)`.

**File:** `llm/server.go`

Added `ToolCallGrammarFromJSON` passthrough (routes.go can't import `llama` directly due to CGo). Added `GrammarTriggerPatterns []string` field to `CompletionRequest`. Added guard `if req.Grammar == "" && len(req.Format) > 0` so Format-derived grammar doesn't overwrite a tool call grammar.

> **Implementation note:** The original plan didn't mention `llm/server.go` changes or the Format→Grammar guard. These were identified during implementation as necessary for correct wiring: `routes.go` needs the passthrough function to avoid importing `llama`, and the Format guard prevents structured output from clobbering the tool grammar when both `tools` and `format` are present in a request.

### Step 3: Build GBNF Grammar from Tool Schemas (~150-250 lines) --- COMPLETED 2026-03-07

> **Implementation note:** This step was implemented as a C++ bridge function in `llama/sampling_ext.cpp` rather than as Go code. This approach calls the vendored `build_grammar()` / `gbnf_format_literal()` API from `json-schema-to-grammar.h` directly, avoiding the need to reimplement GBNF construction in Go. The trie-based exclusion pattern for free-text parameter values was ported from `peg-parser.cpp` (~80 lines) because it is `static` (file-local) and not exposed in any header.

**Files changed:** `llama/sampling_ext.cpp` (~230 lines added), `llama/sampling_ext.h` (~70 lines added)

**Function:** `tool_call_grammar_from_json(tools_json, grammar_out, grammar_max_len, error_out, error_max_len)` — takes a JSON array of `api.Tool` definitions, returns a GBNF grammar string.

**Defensive validation (11 error codes):** null pointers, zero-length buffers, invalid/non-array/empty JSON, missing or malformed `function`/`name`/`parameters`/`properties`/`required`, empty names, null bytes, forbidden characters (`>`, `<`, `\n`, `\r`), duplicate function names, duplicate parameter names, required parameters not in properties, output truncation, grammar build exceptions.

**Produced GBNF grammar (verified output for get_weather + search example):**
```
root             ::= tool-call+
tool-call        ::= "<tool_call>\n" tool-choice "</tool_call>" ws
tool-choice      ::= tool-get-weather | tool-search
tool-get-weather ::= "<function=get_weather>\n" tool-get-weather-arg-location
                     tool-get-weather-arg-unit? "</function>\n"
tool-get-weather-arg-location ::= "<parameter=location>\n" xml-arg-string "\n"
                                  ("</parameter>\n")?
tool-get-weather-arg-unit     ::= "<parameter=unit>\n" xml-arg-string "\n"
                                  ("</parameter>\n")?
tool-search      ::= "<function=search>\n" tool-search-arg-query "</function>\n"
tool-search-arg-query ::= "<parameter=query>\n" xml-arg-string "\n"
                          ("</parameter>\n")?
xml-arg-string   ::= (trie-based exclusion pattern for \n</parameter>, \n<parameter=, \n</function>)
ws               ::= [ \t\n]*
```

Key decisions (all match the original plan):
- Function names: literal alternation of declared tool names
- Parameter names: literal alternation of declared parameter names per tool
- Parameter values: free text (no JSON schema grammar constraints — see Section 3.2)
- `</parameter>` closing tag: optional (matching llama.cpp)
- Parallel tool calls: `tool-call+` repetition
- Required parameters: no `?` suffix; optional parameters: `?` suffix

**Upstream compatibility note (verified 2026-03-07):** The latest llama.cpp (`c5a7788`) upgraded the trie from `unsigned char` to `uint32_t` for Unicode codepoint support, and refactored the tool grammar builder from inline `chat.cpp` to `chat-peg-parser.cpp` with parameterized XML markers. The public API we depend on (`build_grammar`, `gbnf_format_literal`, `common_grammar_builder` in `json-schema-to-grammar.h`) is byte-for-byte identical. Our trie handles ASCII-only delimiters correctly; the Unicode upgrade would only matter for non-ASCII tool/parameter names.

### Step 4: Wire into the Server (~59 lines in routes.go, ~6 lines in runner.go) --- COMPLETED 2026-03-08

**Files:** `server/routes.go`, `runner/ollamarunner/runner.go`

In `ChatHandler` at `routes.go`, after parser initialization, gated on `m.Config.Parser == "qwen3.5"` (not `"qwen3coder"` — only Qwen 3.5):

1. Serialize `req.Tools` to JSON, call `llm.ToolCallGrammarFromJSON()` to build the GBNF string
2. Select mode-dependent trigger patterns based on `req.Think`:
   - **Thinking mode** (`req.Think.Bool() == true`): `[\s\S]*</think>[\s\S]*?(<tool_call>[\s\S]*)` — requires `</think>` before `<tool_call>` so the grammar doesn't activate on hallucinated tool calls inside thinking text
   - **Non-thinking mode** (`req.Think.Bool() == false`): `[\s\S]*?(<tool_call>[\s\S]*)` — no `</think>` required since the renderer prefills `<think>\n\n</think>\n\n` (it's in the prompt, not generated text)
3. Set `Grammar` and `GrammarTriggerPatterns` on the `CompletionRequest`

In `runner.go`, the grammar sampler creation branches on `len(req.GrammarTriggerPatterns) > 0` to choose lazy vs eager grammar initialization.

> **Implementation note:** The original plan did not mention mode-dependent trigger patterns. This was identified during implementation by tracing the Qwen 3.5 renderer's different prefill behavior: thinking mode prefills `<think>\n` (model generates `</think>` in output), non-thinking mode prefills `<think>\n\n</think>\n\n` (`</think>` is in the prompt, not output). A single trigger pattern would either break non-thinking mode (if it requires `</think>`) or activate on hallucinated tool calls inside thinking (if it doesn't). The mode-dependent approach handles both correctly.

> **Scope note:** The plan originally mentioned `"qwen3coder"` as a potential target. During implementation, this was intentionally scoped to `"qwen3.5"` only — Qwen3-Coder uses a different tool call format and would need its own grammar builder.

### Step 5: Better Dead-End Error Detection (~10 lines) --- COMPLETED 2026-03-08

**File:** `sample/samplers.go`

After `s.grammar.Apply(tokens)` at line 76, before calling `s.sample(tokens)`:
```go
allRejected := true
for _, tok := range tokens {
    if !math.IsInf(float64(tok.value), -1) {
        allRejected = false
        break
    }
}
if allRejected {
    return -1, errors.New("sample: grammar rejected all tokens — grammar may be malformed or the generation reached an impossible state")
}
```

This benefits **all models** that use grammar-constrained generation (structured output via JSON format, and now tool call grammars). Previously, a grammar dead-end would produce a cryptic "logits sum to NaN" error via the NaN check at line 149.

### Total Effort

| Step | Lines | Scope | Status |
|------|-------|-------|--------|
| 1. C bridge extension | ~36 (C++) | Generic (any model) | **DONE** (2026-03-08) |
| 2. Go wrapper extension | ~58 (llama.go) + ~30 (samplers.go) + ~25 (server.go) | Generic + Qwen wiring | **DONE** (2026-03-08) |
| 3. GBNF grammar builder | ~230 (C++) | Qwen 3.5 XML format | **DONE** (2026-03-07) |
| 4. Server wiring | ~59 (routes.go) + ~6 (runner.go) | Qwen 3.5 only | **DONE** (2026-03-08) |
| 5. Dead-end error detection | ~10 (samplers.go) | Generic (all models) | **DONE** (2026-03-08) |

**Actual total: 8 files changed, +212/-19 lines.** Additional changes beyond the original plan: renderer fix (unconditional `<think>` wrapping in history), Format→Grammar guard, mode-dependent trigger patterns, consolidated Go helpers to eliminate duplication.

### Bonus: Renderer Fix (not in original plan)

**File:** `model/renderers/qwen35.go`

During implementation, we discovered that the renderer incorrectly gated `<think>...</think>` wrapping of assistant message history on the `isThinking` boolean. The official Qwen 3.5 Jinja2 template unconditionally wraps recent assistant messages in `<think>...</think>` regardless of `enable_thinking`. Both upstream Ollama versions have this bug. Fixed by removing `isThinking &&` from two places (the rendering condition and `splitQwen35ReasoningContent`). This was bug A.2 from `prioritized_research_topics_and_action_items.md`.

---

## 6. Future Enhancement: Full JSON Schema Constraints for Non-String Parameter Values

> **Note:** The C++ bridge function `tool_call_grammar_from_json()` described in the original plan as a "future enhancement" was **implemented in Step 3** (2026-03-07). The function now exists in `llama/sampling_ext.cpp` and handles XML framing + literal name constraints. What remains as a future enhancement is adding **per-parameter JSON schema type constraints** (integers producing only digits, booleans only `true`/`false`, etc.).

To add JSON schema constraints for non-string parameter values, the existing `tool_call_grammar_from_json()` would be extended to call `builder.add_schema(name, param_schema)` for each non-string parameter instead of using the free-text `xml-arg-string` rule. The `json-schema-to-grammar.cpp` code is already vendored and functional — the `add_schema` callback on `common_grammar_builder` is already available. The change would be ~20-30 lines in the existing function.

---

## Appendix: Key Source Files Reference

### Fork files modified (all complete)

| File | Lines Changed | What |
|------|--------------|------|
| `llama/sampling_ext.cpp` | +38/-1 | Lazy grammar init C bridge + null-properties fix |
| `llama/sampling_ext.h` | +7 | Header for lazy grammar init |
| `llama/llama.go` | +58/-1 | Go wrappers: `NewGrammarLazy`, `ToolCallGrammarFromJSON`, consolidated `newGrammar` |
| `llm/server.go` | +25/-3 | `ToolCallGrammarFromJSON` passthrough, `GrammarTriggerPatterns` field, Format→Grammar guard |
| `sample/samplers.go` | +30/-1 | Lazy grammar sampler, dead-end detection, consolidated `newGrammarSampler` |
| `runner/ollamarunner/runner.go` | +6/-1 | Lazy vs eager grammar sampler branch |
| `server/routes.go` | +59/-6 | Grammar building, mode-dependent triggers, CompletionRequest wiring |
| `model/renderers/qwen35.go` | +8/-8 | Unconditional `<think>` wrapping in history (renderer fix) |

### Vendored C++ files (read-only reference — already functional)

| File | What |
|------|------|
| `llama/llama.cpp/common/json-schema-to-grammar.cpp` (1153 lines) | JSON Schema → GBNF converter (generic, elegant, zero model-specific code) |
| `llama/llama.cpp/common/json-schema-to-grammar.h` (43 lines) | Header — `json_schema_to_grammar()`, `build_grammar()`, `common_grammar_builder` |
| `llama/llama.cpp/src/llama-grammar.cpp` | Grammar engine — `apply_impl` (logit masking), `accept_impl` (state advancement), lazy trigger handling |
| `llama/llama.cpp/src/llama-grammar.h` | Grammar struct with lazy fields, `llama_grammar_init_impl` signature |

### Reference llama.cpp files (at `/tmp/llama-cpp-latest/` @ `a0ed91a`)

| File | What |
|------|------|
| `common/chat.cpp:1555-1643` | Tool call grammar builder — the reference implementation we're matching |
| `common/peg-parser.h` | PEG parser builder class used by `chat.cpp` |
| `common/peg-parser.cpp:1249-1410` | PEG → GBNF conversion via `build_grammar()` callback |
