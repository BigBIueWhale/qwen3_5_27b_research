# Grammar-Constrained Tool Call Generation: Implementation Plan

**Date:** 2026-03-07
**Fork:** `BigBIueWhale/ollama` @ commit `57e3f80f` (8 commits atop the Ollama `v0.17.4` merge base `cc90a035`)
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

**File:** `llama/sampling_ext.cpp` (137 lines)
**Header:** `llama/sampling_ext.h` (47 lines)

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

| Constraint | llama.cpp (grammar) | Fork: grammar | Fork: strict parser | Notes |
|------------|---------------------|---------------|---------------------|-------|
| **XML structure** — `<tool_call>`, `</tool_call>`, `<function=...>`, `</function>`, `<parameter=...>`, `</parameter>` well-formed | **Yes** | **Yes** — GBNF literals | **Yes** — `xml.Unmarshal` rejects malformed XML | Grammar prevents malformed tags from ever being generated |
| **Function name** — only declared tool names | **Yes** — `p.literal(name)` alternation (`chat.cpp:1580`) | **Yes** — literal alternation in GBNF | **Yes** — reject when `matchedTool == nil` (currently silently accepts; fix needed) | Highest-value constraint. Prevents phantom function calls. |
| **Parameter name** — only declared param names | **Yes** — `p.literal(param_name)` per parameter (`chat.cpp:1592`) | **Yes** — literal alternation per tool | **Yes** — validate against tool schema (currently accepts unknown params; fix needed) | Prevents invented parameter names |
| **Lazy trigger activation** | **Yes** — `grammar_lazy = true`, trigger on `<tool_call>` word (`chat.cpp:1626-1639`) | **Yes** — vendored C++ fully supports this; needs C bridge exposure (~20 lines) | N/A (parser is post-generation) | Without lazy, grammar would constrain ALL output including thinking/content |
| **Required parameters present** | **Yes** — `p.repeat(arg_rule, 1, 1)` forces exactly-once in declaration order (`chat.cpp:1604`) | **No** — not implementing | **Partially** — can check completeness after parsing | Forces fixed parameter ordering which may fight model's learned order |
| **Optional parameters** | **Yes** — `p.repeat(arg_rule, 0, 1)` | **No** — not implementing | **No** — harmless: duplicate optional param overwrites previous | Same ordering concern |
| **String parameter values** | **Unconstrained** — `p.until_one_of(delimiters)` accepts any text (`chat.cpp:1596-1597`) | **Unconstrained** — free text until `</parameter>` | **No change needed** | Neither validates string content against patterns/minLength/maxLength/enum |
| **Non-string parameter values** (int, number, bool, object, array) | **Yes** — `p.schema()` converts JSON schema to GBNF (`chat.cpp:1599`), using `json-schema-to-grammar.cpp` | **No** — not implementing at grammar level | **Yes** — `parseValue()` returns errors on type mismatch instead of silent fallback | See Section 3.2 for why we skip this in grammar |
| **`</parameter>` closing tag** | **Optional** — `p.optional(p.tool_arg_close(...))` at `chat.cpp:1603` | **Match llama.cpp** — optional in grammar | **Yes** — `xml.Unmarshal` handles both | Some model checkpoints omit closing parameter tags |
| **Parallel tool calls** | **Yes** — `p.repeat(tool_call, min, max)` (`chat.cpp:1610-1613`) | **Yes** — GBNF repetition rule | **Yes** — parser already loops back to `LookingForToolStart` after each `</tool_call>` | Both support parallel calls |

### 3.2 Why We Skip Non-String JSON Schema Constraints in the Grammar

llama.cpp constrains non-string parameter values (integers, booleans, objects, arrays) at the grammar level using its `json-schema-to-grammar.cpp` engine. This means an integer parameter can only produce digit characters, a boolean can only produce `true`/`false`, etc.

We will NOT replicate this because:

1. **The `schema_to_grammar` C bridge produces standalone complete grammars.** It returns a full GBNF grammar with its own `root ::= ...` rule. To use it for individual parameter values inside our larger XML-framing grammar, we would need to either merge multiple standalone GBNF rule sets (fragile — rule name collisions, root rule conflicts) or use the `build_grammar()` callback API.

2. **The `build_grammar()` callback API is not exposed to Go.** It's a C++ function that takes a `std::function` callback. Exposing it through CGo requires a C function pointer shim. This is doable (~50-80 lines of new C bridge code that takes tool definitions JSON, builds the grammar internally using `build_grammar()` + `add_schema()`, and returns the complete GBNF string) but adds complexity.

3. **The strict parser catches type mismatches anyway.** The model may waste a few tokens producing an invalid value, but the parser rejects it before propagating. For a well-trained model like Qwen 3.5, non-string type mismatches are rare.

4. **Diminishing returns.** The two most damaging failure modes — hallucinated function names and hallucinated parameter names — are already blocked by the grammar. A parameter type mismatch is a recoverable error; a phantom function call is not.

**Future option:** If we later want JSON schema constraints, the cleanest path is a new C bridge function `tool_call_grammar_from_json(const char* tools_json)` that takes the Ollama `[]api.Tool` JSON, calls `build_grammar()` internally with `add_rule()` for XML framing and `add_schema()` for parameter schemas, and returns the complete GBNF string. ~50-80 lines of C++. The `json-schema-to-grammar.cpp` code is already vendored and functional — no porting needed.

### 3.3 Summary

**What the fork WILL enforce at the grammar level (during generation — the model cannot produce these violations):**
- Function names are one of the declared tool names (literal alternation)
- Parameter names are one of the declared parameter names for that function (literal alternation)
- XML structure is well-formed (`<tool_call>`, `<function=...>`, `<parameter=...>` tags properly nested)
- Lazy activation: grammar only constrains output after `<tool_call>` trigger, leaving content/thinking unconstrained
- Parallel tool calls supported via GBNF repetition

**What the fork WILL enforce at the parser level (after generation — model may waste tokens, but errors are caught):**
- Function name matches a declared tool (error if not)
- Parameter names match declared parameters (error if not)
- Parameter value types match declared schema types (error instead of silent fallback to string)
- Unclosed `<tool_call>` tags surfaced as errors when generation ends
- Malformed XML between `<tool_call>` and `</tool_call>` surfaced as errors

**What the fork will NOT enforce (llama.cpp does, but too much work for the return):**
- JSON schema validation for non-string parameter values at the grammar level (integers producing only digits, booleans only `true`/`false`, etc.). Strict parser catches these after generation.
- Required parameter completeness at the grammar level (forcing exactly-once in declaration order). Strict parser can validate completeness after parsing.
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

The parser is belt-and-suspenders — it validates what the grammar already enforced. In the happy path, the parser should never see an invalid tool call.

### 4.2 Grammar Dead-End (Bug in Grammar Construction)

If the grammar has a bug (reaches a state with no valid continuations):

1. `grammar.Apply(tokens)` at `samplers.go:76` sets ALL token logits to `-inf`
2. `s.sample(tokens)` runs — `temperature()` divides `-inf` by temperature (still `-inf`), `softmax()` produces `NaN`, sum is `NaN`
3. Line 139: `math.IsNaN(float64(sum))` returns `true` → `errors.New("sample: logits sum to NaN, check model output")`
4. `Sample()` returns the error, which propagates up through the runner

**Current state:** The `NaN` check exists but its error message is misleading ("logits sum to NaN, check model output") — it doesn't mention grammar. This is accidental error detection, not intentional.

**Recommended improvement (~5 lines):** After `s.grammar.Apply(tokens)` at line 76, check if all tokens are `-inf`. If so, return a clear error: `"grammar rejected all tokens — possible grammar construction bug"`. This catches the issue before it becomes a cryptic NaN.

### 4.3 Model Outputs Content Only (No Tool Call)

1. Grammar stays in `awaiting_trigger` state for the entire generation
2. `llama_grammar_apply_impl` returns immediately on every token — no logit masking
3. Model generates freely, produces content, hits EOS
4. Parser receives content, emits it as normal text
5. No tool calls produced — client gets a text response

This is the correct behavior when the model decides not to call any tools.

### 4.4 Grammar Conflicts with Format (Structured Outputs)

If the user sends both `tools` AND `format` (structured output), both would try to set `req.Grammar`. Currently `format` is handled at `llm/server.go:1539-1558` and tools are handled by the parser at `server/routes.go:2199-2212`.

**Resolution:** When tools are present, the tool call grammar takes precedence. The grammar includes a lazy trigger, so content before `<tool_call>` is unconstrained — the model can produce JSON-formatted content if it wants. If the user specifically needs grammar-constrained JSON AND grammar-constrained tool calls, that's a fundamentally different grammar (not supported in the initial implementation). ~3 lines of conditional logic.

### 4.5 Grammar Accepts Token, Then Grammar State Becomes Invalid

The `llama_grammar_accept_impl` at `llama-grammar.cpp:1372` has a hard assertion:
```cpp
GGML_ABORT("grammar error: end of grammar token received but grammar stack is not empty");
```

This fires if an EOS token is accepted while the grammar hasn't reached a completion state. In practice this means the generation tried to end in the middle of a tool call (e.g., after `<tool_call>` but before `</tool_call>`). The `GGML_ABORT` kills the process — this is a crash, not a graceful error.

**Mitigation:** A correctly-constructed grammar should never hit this — it allows EOS only at valid completion points (after `</tool_call>` or after content with no tool call trigger). But if it does happen, the process crash is the correct signal that the grammar has a bug.

---

## 5. Implementation Plan

All changes are Qwen-scoped except the thin C bridge extension (step 1-2), which is a generic improvement any model could use.

### Step 1: Extend the C Bridge (~20 lines)

**Files:** `llama/sampling_ext.cpp`, `llama/sampling_ext.h`

Add `grammar_init_lazy` (or extend `grammar_init` with additional parameters):
```c
struct llama_grammar *grammar_init_lazy(
    char* grammar,
    uint32_t* tokens, size_t n_tokens,
    const char** pieces,
    uint32_t* eog_tokens, size_t n_eog_tokens,
    bool lazy,
    const char** trigger_patterns, size_t n_trigger_patterns,
    llama_token* trigger_tokens, size_t n_trigger_tokens
);
```

This calls `llama_grammar_init_impl` with `lazy=true` and the trigger arrays instead of hardcoding `false`/`nullptr`.

### Step 2: Extend the Go Wrapper (~20 lines)

**File:** `llama/llama.go`

Add `NewGrammarLazy(grammarStr, vocabIds, pieces, eogTokens, triggerPatterns, triggerTokens)` that calls the new C binding. The lazy grammar behavior is internal to the C++ engine — the existing `Apply`/`Accept` methods on `GrammarSampler` work transparently with lazy grammars because the engine handles trigger detection, buffering, and constraint activation internally.

**File:** `sample/samplers.go`

Add `NewGrammarSamplerLazy(...)` that mirrors `NewGrammarSampler` but uses the lazy init path.

### Step 3: Build GBNF Grammar from Tool Schemas (~150-250 lines)

**New file:** `model/parsers/qwen3coder_grammar.go` (or similar)

A function that takes `[]api.Tool` and produces a GBNF grammar string encoding the Qwen `<tool_call>/<function=name>/<parameter=name>` XML format.

The grammar structure (pseudocode GBNF):
```
root ::= tool-call+
tool-call ::= "<tool_call>\n" tool-choice "</tool_call>" space
tool-choice ::= tool-get-weather | tool-search | ...
tool-get-weather ::= "<function=" "get_weather" ">\n" param-location param-unit? "</function>\n"
param-location ::= "<parameter=" "location" ">\n" free-text "\n</parameter>\n"
param-unit ::= "<parameter=" "unit" ">\n" free-text "\n</parameter>\n"
free-text ::= [^\n<]* (... characters that aren't tag openers ...)
space ::= " " | "\n"
```

Key decisions:
- Function names: literal alternation of declared tool names
- Parameter names: literal alternation of declared parameter names per tool
- Parameter values: free text (no JSON schema grammar constraints — see Section 3.2)
- `</parameter>` closing tag: optional (matching llama.cpp)
- Parallel tool calls: `tool-call+` repetition

### Step 4: Wire into the Server (~20 lines)

**File:** `server/routes.go`

In the chat completion handler, after parser initialization (~line 2199-2212), when the parser is `"qwen3.5"` (or `"qwen3coder"`) and `len(req.Tools) > 0`:

1. Call the grammar builder from step 3 to produce the GBNF string
2. Set the GBNF string on the `CompletionRequest` so it flows to the runner
3. Configure lazy trigger on `<tool_call>`

The existing `GrammarSampler` creation path at `runner/ollamarunner/runner.go:879-888` handles the rest.

### Step 5: Strict Parser Validation as Defense-in-Depth (~50-100 lines)

**File:** `model/parsers/qwen3coder.go`

Changes to existing parser functions:
- `parseToolCall()` (line 341): Return an error when `matchedTool == nil` instead of silently accepting
- `parseToolCall()`: Reject parameters whose names don't match any declared parameter
- `parseValue()` (line 404): Return errors on type mismatch instead of falling back to raw strings
- `eat()` (line 150): Surface unclosed `<tool_call>` as an error when `done=true`

This step can be implemented immediately as a standalone improvement, independent of the grammar steps.

### Step 6 (Recommended): Better Dead-End Error Detection (~5 lines)

**File:** `sample/samplers.go`

After `s.grammar.Apply(tokens)` at line 76, before calling `s.sample(tokens)`:
```go
allRejected := true
for _, t := range tokens {
    if !math.IsInf(float64(t.value), -1) {
        allRejected = false
        break
    }
}
if allRejected {
    return -1, errors.New("sample: grammar rejected all tokens — grammar may be malformed")
}
```

### Total Effort

| Step | Lines | Scope |
|------|-------|-------|
| 1. C bridge extension | ~20 | Generic (any model) |
| 2. Go wrapper extension | ~20 | Generic (any model) |
| 3. GBNF grammar builder | ~150-250 | Qwen-specific |
| 4. Server wiring | ~20 | Qwen-specific |
| 5. Strict parser validation | ~50-100 | Qwen-specific |
| 6. Dead-end error detection | ~5 | Generic (any model) |
| **Total** | **~265-415** | |

### Sequencing

- Step 5 (strict parser) can be implemented immediately as a standalone improvement
- Steps 1-4 (grammar-constrained generation) depend on each other and should be implemented together
- Step 6 (error detection) can be done at any time

---

## 6. Future Enhancement: Full JSON Schema Constraints via C++ Bridge

If we later want to match llama.cpp's full JSON schema validation at the grammar level (non-string parameter values constrained to match their declared types), the cleanest path is:

**New C bridge function** (~50-80 lines of C++ in `sampling_ext.cpp`):
```c
int tool_call_grammar_from_json(
    const char* tools_json,    // JSON array of tool definitions
    const char* format,        // "qwen35" or "qwen3coder"
    char* grammar_out,         // output GBNF string
    size_t max_len
);
```

This function would:
1. Parse the tools JSON
2. Call `build_grammar()` with a callback that:
   - Adds custom GBNF rules for XML framing (`<tool_call>`, `<function=...>`, etc.)
   - Calls `builder.add_schema(name, param_schema)` for each non-string parameter to get type-constrained GBNF rules from `json-schema-to-grammar.cpp`
   - Composes everything into a unified grammar
3. Return the complete GBNF string

The `json-schema-to-grammar.cpp` code is already vendored and functional. No porting needed — just a thin C++ function that orchestrates the existing pieces.

---

## Appendix: Key Source Files Reference

### Fork files to modify

| File | Lines to Change | What |
|------|----------------|------|
| `llama/sampling_ext.cpp` | ~20 new | Lazy grammar init C bridge |
| `llama/sampling_ext.h` | ~5 new | Header for lazy grammar init |
| `llama/llama.go` | ~20 new | Go wrapper for lazy grammar |
| `sample/samplers.go` | ~25 new | Lazy grammar sampler + dead-end detection |
| `model/parsers/qwen3coder_grammar.go` | ~150-250 new | GBNF grammar builder from tool schemas |
| `server/routes.go` | ~20 changed | Wire grammar into tool call path |
| `model/parsers/qwen3coder.go` | ~50-100 changed | Strict parser validation |

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
