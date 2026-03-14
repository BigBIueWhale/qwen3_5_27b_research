# Client Input Validation: What Exists, What's Missing, and How to Fix It Without Code Duplication

**Date:** 2026-03-14
**Fork:** `BigBIueWhale/ollama` at `/home/user/shared_vm/ollama/`, branch `main`
**Scope:** Validation of chat API requests from client applications — preventing clients from constructing conversations that produce prompt formats the Qwen 3.5 model was never trained on.

---

## What "Client Input Validation" Means

When a client application (Claude Code, Open WebUI, LangChain, a custom agentic coding assistant, or any other program) wants to talk to the Qwen 3.5 model running on the Ollama fork, it sends an HTTP POST request to `http://localhost:11434/api/chat`. The body of that request is a JSON object containing:

- A `messages` array — the full conversation history (system instructions, user queries, assistant responses, tool call results)
- An optional `tools` array — function definitions the model is allowed to call (like `get_weather`, `search_code`, `run_tests`)
- An optional `think` value — whether the model should show its chain-of-thought reasoning in `<think>` blocks

**Client input validation** means: before the server spends GPU time running the model, does it check whether the conversation the client sent follows the conventions the model was trained on? If the conversation violates those conventions — a system message in the wrong position, an unknown role like `"function"`, two user messages in a row without an assistant response between them — does the server reject it with a clear error? Or does it silently render the malformed conversation into a prompt string, feed it to the model, and let the model produce degraded output?

The Qwen 3.5 model was trained on conversations that follow a very specific byte-level format, defined by the official Jinja2 template from `Qwen/Qwen3.5-27B` on Hugging Face. If the server feeds the model a prompt that deviates from this format, two things go wrong:

1. **The model gets quietly worse.** It doesn't crash or error — it just produces lower-quality tool calls, weaker reasoning, and more hallucinations. The user blames the model ("Qwen 3.5 is unreliable") rather than the server that silently corrupted the prompt.

2. **The KV cache (the expensive intermediate computation that lets the server skip reprocessing prior turns) gets invalidated.** In a 20-turn agentic coding session, each response takes progressively longer because the server has to reprocess the entire history instead of reusing cached results. This is invisible in functional testing because the output is still correct — just slow.

---

## The Journey of a Chat Request Through the Ollama Fork

Here is every stage a chat request passes through, from the moment the client sends it to the moment the model sees a prompt string. At each stage: what validation exists today, what the official Qwen 3.5 template would reject, and what llama.cpp (the reference C++ inference engine maintained by `ggml-org/llama.cpp`) does.

### Stage 1: HTTP JSON Parsing (`server/routes.go`, the `ChatHandler` function)

**What this stage does:** The Gin web framework (Ollama's HTTP server library) receives the raw HTTP POST body and deserializes it from JSON bytes into a Go `api.ChatRequest` struct. This is where raw bytes become typed Go objects — a `messages` array becomes a slice of `api.Message` structs, each with a `Role` string field, a `Content` string field, an `Images` slice, a `ToolCalls` slice, and so on.

**What validation happens now:**
- **JSON syntax errors** are caught — if the client sends `{"messages": [}` (missing closing bracket), Gin returns HTTP 400. HTTP 400 means "Bad Request" — the standard HTTP status code meaning "the server understood the protocol but the content of your request is structurally invalid; this is the client's fault, fix it and try again."
- **`TopLogprobs` range** is checked at `routes.go:1995-1998` — must be 0-20 or the server returns HTTP 400.
- **Model name validity** is checked at `routes.go:2000-2017` — if the model name can't be parsed, HTTP 400.
- **Empty messages array** is checked at `routes.go:2178` — if `len(req.Messages) == 0`, the server returns HTTP 200 with a "load" status (this just loads the model into GPU memory without generating anything).
- **Think capability** is checked at `routes.go:2149-2165` — if the client requests `think: true` but the loaded model doesn't support thinking, HTTP 400 with the message `"<model> does not support thinking"`.

**What validation is missing — the conversation structure is never checked:**
- **No message role validation.** The `api.Message.UnmarshalJSON()` method at `api/types.go:232` normalizes the role to lowercase (`m.Role = strings.ToLower(m.Role)`) but never checks whether the result is one of the four valid roles: `"system"`, `"user"`, `"assistant"`, `"tool"`. A client can send `"role": "function"` (the old OpenAI role name, used by GPT-3.5 and early GPT-4 APIs), `"role": "developer"` (the new OpenAI role name, used by GPT-4.1 and later), `"role": ""` (empty string), or `"role": "banana"` (nonsense), and the server accepts all of them without returning an error.
- **No message ordering validation.** The server does not check whether system messages appear only at position 0, whether user and assistant messages alternate, whether tool messages follow assistant messages that actually made tool calls, or whether the conversation contains at least one real user query (as opposed to only tool call results).
- **No content validation.** Empty content strings, images in system messages, assistant messages with both a `Thinking` field AND inline `<think>` tags in `Content` — all silently accepted.

**What the official Qwen 3.5 Jinja2 template would reject here:** The official template has 8 `raise_exception()` calls — hard errors that halt template rendering and return the error message to the caller. The 8 rejections are (exact error messages from the template source):
1. `"No messages provided."` — empty or missing messages array
2. `"System message must be at the beginning."` — system message at any position other than index 0
3. `"No user query found in messages."` — every user message is a `<tool_response>` wrapper with no real human query
4. `"Unexpected message role."` — role is not `"system"`, `"user"`, `"assistant"`, or `"tool"`
5. `"System message cannot contain images."` — image content in a system-role message
6. `"System message cannot contain videos."` — video content in a system-role message
7. `"Unexpected item type in content."` — content array item that isn't image, video, or text
8. `"Unexpected content type."` — content is not a string, array, or null

None of these checks exist in the fork's `ChatHandler`.

**What llama.cpp does at this stage:** The llama.cpp HTTP server validates at `tools/server/server-common.cpp`:
- Messages field is present and is a JSON array (lines 938-944)
- All non-assistant messages have a `content` field (line 947-949)
- No 2 or more consecutive assistant messages at the end of the list (line 1055-1057)
- `enable_thinking` must be boolean, not a quoted string (line 1042-1043)
- Assistant prefill and `enable_thinking` cannot both be active (line 1062-1064)

These are all returned as HTTP 400 responses. Additionally, llama.cpp executes the official Jinja2 template directly, so the template's own 8 `raise_exception()` calls fire naturally during rendering. When a Qwen 3.5 template calls `raise_exception("System message must be at the beginning.")`, the exception propagates through the Jinja2 runtime → gets wrapped in `std::invalid_argument` at `chat.cpp:1532` → gets caught by the HTTP server's `ex_wrapper()` at `server.cpp:43-46` → returns HTTP 400 to the client. (Earlier research incorrectly stated this mapped to HTTP 500 — the wrapping at `chat.cpp:1532` converts it to `std::invalid_argument` first.)

### Stage 2: System Message Prepending (`server/routes.go:2078-2080` and `2189-2192`)

**What this stage does:** If the model has a default system message defined in its Modelfile (the configuration file that specifies model parameters, system prompt, etc.) and the client's first message isn't already a system message, the server prepends the default system message at position 0 of the messages array.

**What validation happens now:** Just a role string comparison — `req.Messages[0].Role != "system"`. If the first message IS a system message, the model's default system message is skipped.

**What validation is missing:** No check for multiple system messages in the array. No check for system messages appearing at positions other than 0. A client can send `[{role: "user", content: "Hi"}, {role: "system", content: "Be concise"}, {role: "assistant", content: "Hello"}]` and the server will prepend its own system message at position 0, creating a conversation with TWO system messages — one at position 0 (the model's default from the Modelfile) and one at position 2 (the client's misplaced system message). The renderer handles the second system message by rendering it as a system-role turn in the middle of the conversation body: `<|im_start|>system\nBe concise<|im_end|>\n`. This is syntactically valid ChatML markup but is a prompt format the model was never trained on — the model expects at most one system message, and always at the very beginning.

### Stage 3: Prompt Construction and Context Fitting (`server/prompt.go`)

**What this stage does:** The `chatPrompt` function at `prompt.go:23-150` converts the message array into a prompt string by calling the model-specific renderer. The `fitsContext` function at `prompt.go:45-71` runs a binary search to find the longest prompt that fits in the model's context window (131,072 tokens for the Qwen 3.5 Modelfile configuration): it tries rendering the full conversation, checks the token count, and if it exceeds the limit, drops the oldest messages and retries. This means `Render()` is called O(log N) times per request, where N is the number of messages.

**What validation happens now:** Only one check — at `prompt.go:115`, if the model is from the `mllama` family (Meta's Llama multimodal models, not Qwen) and a message has more than 1 image, it returns an error. No other validation.

**Why this stage matters for validation architecture:** The binary-search loop calls `Render()` multiple times with progressively shorter message arrays. If validation were embedded inside `Render()`, it would run on every iteration — wasting time re-checking structural properties that don't change between iterations (the message roles, system message position, and tool call structure are the same regardless of which messages are truncated). This is the key architectural reason validation should happen once, before the binary search, not inside the renderer.

### Stage 4: Renderer Execution (`model/renderers/qwen35.go`, the `Render` method)

**What this stage does:** This is the core of prompt construction. The `Qwen35Renderer.Render()` method at `qwen35.go:82-194` walks the message array and emits the exact byte sequence that becomes the model's input:
- System messages → `<|im_start|>system\n[tools JSON]\n[content]<|im_end|>\n`
- User messages → `<|im_start|>user\n[content]<|im_end|>\n`
- Assistant messages → `<|im_start|>assistant\n<think>\n[reasoning]\n</think>\n\n[content][tool calls XML]<|im_end|>\n`
- Tool response messages → grouped under `<|im_start|>user\n<tool_response>\n[result]\n</tool_response>\n...<|im_end|>\n`
- Generation prompt (the final bytes that tell the model "start generating now") → `<|im_start|>assistant\n<think>\n` when thinking is enabled, or `<|im_start|>assistant\n<think>\n\n</think>\n\n` when thinking is disabled

**What validation happens now:** Zero. The method signature is `Render(messages []api.Message, tools []api.Tool, think *api.ThinkValue) (string, error)` — it CAN return an error, but no code path ever does. Every input is silently accepted and rendered into some prompt string, no matter how malformed.

**What goes wrong with specific invalid inputs — concrete prompt traces:**

Each example below shows the exact prompt string the renderer produces. These were traced through the actual code in `qwen35.go`.

**Example 1 — Unknown role `"function"` (silently dropped):**

Client sends:
```json
[
  {"role": "system", "content": "You are helpful"},
  {"role": "user", "content": "Hi"},
  {"role": "function", "content": "result data"},
  {"role": "assistant", "content": "OK"}
]
```

The if/else chain at lines 138-180 checks for `"user"`, `"system"`, `"assistant"`, and `"tool"`. The `"function"` role matches none of these branches. There is no `else` clause. The message produces no output — it is silently dropped from the prompt.

Prompt the model sees:
```
<|im_start|>system
You are helpful<|im_end|>
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
<think>

</think>

OK<|im_end|>
<|im_start|>assistant
<think>
```

The client sent 4 messages. The model sees 3. The "function" result data is invisible. The client has no way to know its message was ignored.

*Official template:* `raise_exception("Unexpected message role.")` — rendering halts with a clear error.
*llama.cpp:* Same — the Jinja2 engine propagates the exception as HTTP 400.

**Example 2 — System message at position 2 (rendered in wrong location):**

Client sends:
```json
[
  {"role": "user", "content": "Hi"},
  {"role": "assistant", "content": "Hello"},
  {"role": "system", "content": "Be concise"},
  {"role": "user", "content": "Tell me a joke"}
]
```

Line 100 checks `messages[0].Role == "system"` — it's `"user"`, so no system preamble is rendered. At the main loop, the system message at index 2 matches the catch-all at line 138: `message.Role == "system" && i != 0`, so it's rendered as a regular conversation turn.

Prompt the model sees:
```
<|im_start|>user
Hi<|im_end|>
<|im_start|>assistant
Hello<|im_end|>
<|im_start|>system
Be concise<|im_end|>
<|im_start|>user
Tell me a joke<|im_end|>
<|im_start|>assistant
<think>
```

A system instruction appears mid-conversation, after the assistant already responded. The model was trained with system messages only at position 0. This out-of-position system message produces undefined model behavior.

*Official template:* `raise_exception("System message must be at the beginning.")` — rendering halts.
*llama.cpp:* Same — HTTP 400.

**Example 3 — Empty tool function name (malformed XML):**

Client sends an assistant message with a tool call where `Function.Name` is an empty string `""`.

Line 159 concatenates `"<function=" + toolCall.Function.Name + ">"` without checking emptiness.

Relevant portion of the prompt:
```
<tool_call>
<function=>
<parameter=location>
Paris
</parameter>
</function>
</tool_call>
```

`<function=>` is malformed XML — no function name. The model was never trained on this pattern. The fork's own grammar-constrained generation pipeline (which constrains the model's output to valid tool call XML) would never produce this. But the renderer accepts it in history without complaint.

*Official template:* Would produce the same malformed XML — the template doesn't validate empty names either. However, in practice this only appears in history (tool calls the model previously made), and a well-functioning grammar would prevent the model from generating an empty name in the first place.
*llama.cpp:* Same — no validation of tool call names in history.

**Example 4 — Two consecutive user messages (out-of-distribution):**

Client sends:
```json
[
  {"role": "user", "content": "Hi"},
  {"role": "user", "content": "Are you there?"},
  {"role": "assistant", "content": "Yes"}
]
```

Both user messages match the `message.Role == "user"` branch at line 138. Both are rendered.

Prompt the model sees:
```
<|im_start|>user
Hi<|im_end|>
<|im_start|>user
Are you there?<|im_end|>
<|im_start|>assistant
Yes<|im_end|>
<|im_start|>assistant
<think>
```

Two consecutive `<|im_start|>user` blocks. The model was trained exclusively on alternating user-assistant turns. This is out-of-distribution.

*Official template:* Does NOT reject this — the template doesn't enforce role alternation either. This is an implicit training assumption, not an explicit template guard.
*llama.cpp:* Does not validate this either, except for consecutive assistant messages at the END of the list (line 1055-1057).

**Example 5 — Tool message without preceding assistant tool call (orphaned):**

Client sends:
```json
[
  {"role": "user", "content": "Hi"},
  {"role": "tool", "content": "some result"},
  {"role": "assistant", "content": "OK"}
]
```

The tool message at index 1 matches line 172: `message.Role == "tool"`. It's wrapped in `<|im_start|>user\n<tool_response>...\n</tool_response>\n<|im_end|>\n`. There is no preceding assistant message with a `<tool_call>` that this result corresponds to.

Prompt the model sees:
```
<|im_start|>user
Hi<|im_end|>
<|im_start|>user
<tool_response>
some result
</tool_response>
<|im_end|>
<|im_start|>assistant
OK<|im_end|>
<|im_start|>assistant
<think>
```

A tool response appears after a user greeting, with no preceding tool call. The model sees a result with no context about what function produced it or why it was called. The conversation structure is semantically broken.

*Official template:* Does NOT reject this — the template groups consecutive tool messages under a `<|im_start|>user` block without checking what precedes them. This is another implicit assumption.
*llama.cpp:* Does not validate this.

**Example 6 — Images in system message (training-absent tokens):**

Client sends a system message with an image in its `Images` array. The `renderContent()` method at lines 47-62 renders `<|vision_start|><|image_pad|><|vision_end|>` tokens for ANY message role, including system.

Prompt the model sees:
```
<|im_start|>system
<|vision_start|><|image_pad|><|vision_end|>Describe this<|im_end|>
<|im_start|>user
What is it?<|im_end|>
<|im_start|>assistant
<think>
```

Vision tokens inside the system prompt. The model was trained with vision tokens only in user and assistant turns, never in system messages.

*Official template:* `raise_exception("System message cannot contain images.")` — rendering halts.
*llama.cpp:* Same — HTTP 400 via Jinja2 execution.

### Stage 5: Grammar-Constrained Tool Call Generation (`server/routes.go:2223-2254`, `llama/sampling_ext.cpp`)

**What this stage does:** This is unique to the fork. When the parser is `"qwen3.5"` and tools are provided, the `ChatHandler` serializes the tool definitions to JSON, passes them to the C++ `tool_call_grammar_from_json()` function (via CGo — Go calling C/C++ code — through `llama/llama.go:ToolCallGrammarFromJSON()`), which builds a GBNF grammar (a formal grammar specification that constrains the model to only produce valid tool call XML with real function names and properly-typed parameters during text generation).

**What validation happens now — this is the fork's strongest validation point:**

The C++ `tool_call_grammar_from_json()` function at `sampling_ext.cpp:411-606` has 12 defensive checks, each returning a specific error code defined in `sampling_ext.h:91-101`:

| Error Code | Name | What It Catches |
|-----------|------|-----------------|
| -1 | `TOOL_GRAMMAR_ERR_NULL_INPUT` | Null pointer passed as input (internal programming error) |
| -2 | `TOOL_GRAMMAR_ERR_NULL_OUTPUT` | Null pointer for output buffer (internal programming error) |
| -3 | `TOOL_GRAMMAR_ERR_ZERO_LENGTH` | Zero-length output buffer (internal programming error) |
| -4 | `TOOL_GRAMMAR_ERR_INVALID_JSON` | Tools JSON can't be parsed (client sent malformed tools) |
| -5 | `TOOL_GRAMMAR_ERR_NOT_ARRAY` | Tools JSON is not an array (client sent wrong structure) |
| -6 | `TOOL_GRAMMAR_ERR_EMPTY_TOOLS` | Tools array is empty (client said "use tools" but provided none) |
| -7 | `TOOL_GRAMMAR_ERR_INVALID_TOOL` | Tool missing `function` object, missing `name`, empty name, null bytes in name, or forbidden characters `>`, `<`, `\n`, `\r` in name |
| -8 | `TOOL_GRAMMAR_ERR_TRUNCATED` | Generated grammar string exceeded the 64KB output buffer |
| -9 | `TOOL_GRAMMAR_ERR_GRAMMAR_BUILD` | The GBNF grammar construction itself failed (e.g., invalid JSON schema in parameters) |
| -10 | `TOOL_GRAMMAR_ERR_DUPLICATE_NAME` | Two tools have the same function name |

The per-tool validation function `validate_tool_json()` at `sampling_ext.cpp:301-388` performs detailed checks: tool must be an object, must have a `"function"` key, function must be an object, must have a `"name"` key, name must be a string, name must be non-empty, name must not contain null bytes, name must not contain characters that would break XML (`>`, `<`) or GBNF rule names (`\n`, `\r`), and if `required` parameters are declared, every required parameter name must exist in the `properties` object.

**These tool-level checks are unique to the fork.** llama.cpp's `common_chat_tools_parse_oaicompat()` at `common/chat.cpp:396-428` only checks: tools is an array, each tool has `type: "function"`, each tool has a `function` object, and each function has a `name` field (presence only — an empty string `""` passes). No duplicate name detection, no forbidden character check, no null byte check, no required-in-properties check.

**What validation is missing at this stage:**
- **No Go-level pre-validation.** The Go wrapper `ToolCallGrammarFromJSON()` at `llama/llama.go:775-797` passes the JSON string straight to C++ with zero pre-checks. If the C++ function returns an error code, it's translated into a Go error like `"tool_call_grammar_from_json (code -7): tools[2]: function name is empty"` — technically descriptive but not formatted for end-user consumption.
- **Wrong HTTP status code.** At `routes.go:2231-2234`, if `ToolCallGrammarFromJSON` returns an error, the server responds with `http.StatusInternalServerError` (HTTP 500). This should be HTTP 400 — the client sent invalid tool definitions, that's the client's fault. HTTP 500 means "Internal Server Error" — it tells the client "something broke on my end, you might want to retry later." Monitoring systems flag HTTP 500 as a server bug, automated retry logic retries the same broken request, and the client's error handling code for "bad input" never fires.

---

## The Architectural Problem and How to Solve It Without Code Duplication

### Why Naive Validation Would Create Code Duplication

The obvious approach — adding a `ValidateMessages()` function that runs before `Render()` — would duplicate the structural analysis that `Render()` already performs. Here is the specific state that both validation and rendering need:

| Structural Property | Where Rendered in `qwen35.go` | What Validation Would Check |
|---|---|---|
| `lastQueryIndex` — position of last non-tool-response user message | Lines 114-127: backward scan calling `renderContent()` on each user message to check `<tool_response>` wrapping | Whether any real user query exists at all (template rejection 3: "No user query found") |
| System message position | Lines 100-112: checks `messages[0].Role == "system"` | Whether system messages appear only at index 0 (rejection 2), whether multiple system messages exist |
| Message role sequence | Lines 138-180: if/else chain dispatches on role | Whether all roles are valid (rejection 4), whether roles alternate correctly |
| Tool call structure | Lines 149-166: iterates `toolCall.Function.Name` and `Arguments.All()` | Whether function names are non-empty |
| Content type and media | Lines 47-62: `renderContent()` processes images | Whether system messages contain images or videos (rejections 5, 6) |
| Tool response grouping | Lines 172-179: consecutive tool messages grouped under one `<|im_start|>user` tag | Whether tool messages appear in valid positions |

A separate `ValidateMessages()` function would need to re-walk the entire message array and recompute `lastQueryIndex` (including calling `renderContent()` on every user message to check for `<tool_response>` wrapping), re-check system message positions, re-scan roles, and re-inspect tool calls. This is not a trivial duplication — the `lastQueryIndex` computation alone requires rendering content to check whether it starts and ends with `<tool_response>` tags.

### Why Embedding Validation Inside `Render()` Is Also Wrong

Embedding validation checks directly inside `Render()` avoids the duplication but creates two new problems:

1. **Repeated work across binary search iterations.** The `fitsContext` function at `server/prompt.go:45-71` calls `Render()` O(log N) times with progressively shorter message arrays. The structural validity of the full message array doesn't change between iterations — only the number of included messages changes. Validation would run on every iteration, wasting time.

2. **Interleaved concerns.** Validation logic ("is this role valid?") becomes tangled with rendering logic ("emit `<|im_start|>` tag"), making both harder to test independently.

### The Right Architecture: Validate in `Render()` Once, Skip on Subsequent Calls

The key insight is that `Render()` already computes the structural properties validation needs — the question is when and how to expose validation without re-walking the messages.

**The approach:** Add validation checks directly into the existing structural analysis code in `Render()`, at the points where the relevant state is already computed. Use a flag or the binary-search call pattern to run validation only on the first `Render()` call, not on subsequent truncation iterations.

Concretely, this means adding checks at these exact locations in `qwen35.go`:

**Role validation — at lines 138-180 (the existing role dispatch):**

The if/else chain already enumerates every valid role. Adding an `else` clause that returns an error is one line of code and zero duplication:

```go
} else if message.Role == "tool" {
    // ... existing tool handling ...
} else {
    return "", fmt.Errorf("message at index %d has invalid role %q (valid roles: system, user, assistant, tool)", i, message.Role)
}
```

This reuses the existing role dispatch. No separate validation walk needed. The check runs inside the main loop that already processes every message.

**System message position — at lines 100-112 (existing system message extraction):**

The renderer already checks `messages[0].Role == "system"`. Extending this to reject non-first system messages requires adding a check inside the main loop:

```go
if message.Role == "system" && i != 0 {
    return "", fmt.Errorf("system message at index %d must be at index 0", i)
}
```

This replaces the current behavior at line 138 where non-first system messages are silently rendered as conversation turns. One line, zero duplication.

**No user query found — at lines 114-127 (existing `lastQueryIndex` computation):**

The renderer already walks backward to find `lastQueryIndex`. After the loop, if `multiStepTool` is still `true` (meaning no non-tool-response user message was found), returning an error instead of proceeding is one line:

```go
if multiStepTool {
    return "", fmt.Errorf("no user query found in messages (all user messages are tool responses)")
}
```

This reuses the existing backward scan. Zero duplication.

**Images in system messages — at lines 47-62 (existing `renderContent()` method):**

The `renderContent()` method already iterates over `content.Images`. Adding a role-based check requires passing the role or an `isSystemContent` flag to `renderContent()`, which is a small signature change:

```go
if isSystemContent && len(content.Images) > 0 {
    return "", 0, fmt.Errorf("system message cannot contain images")
}
```

This reuses the existing image iteration. The only change is adding a parameter to `renderContent()` — which currently doesn't know what role it's rendering for. Note: this changes `renderContent()`'s return signature from `(string, int)` to `(string, int, error)`, which affects the 4 call sites within `qwen35.go`.

**Empty tool function names — at lines 149-166 (existing tool call rendering):**

The renderer already accesses `toolCall.Function.Name` at line 159. Adding an emptiness check is one line:

```go
if toolCall.Function.Name == "" {
    return "", fmt.Errorf("assistant message at index %d has tool call at index %d with empty function name", i, j)
}
```

This reuses the existing tool call iteration. Zero duplication.

### How This Interacts with `fitsContext` Binary Search

The `fitsContext` function at `prompt.go:45-71` calls `Render()` multiple times. The validation checks above run inside `Render()`, so they would run on every binary search iteration.

However, this is acceptable for two reasons:

1. **The checks are O(1) per message.** Each check is a string comparison, an integer comparison, or an array length check inside a loop that already iterates every message. The cost is a few nanoseconds per message per iteration — negligible compared to tokenization and context window size estimation.

2. **Validation errors are deterministic.** If the full message array is invalid (system message at wrong position, unknown role, etc.), it's still invalid after truncation. The first `fitsContext` call will return the error immediately. The binary search never starts.

The only edge case is `lastQueryIndex`: the "no user query found" check runs on the full message array. After truncation, the shorter array might contain a real user query that was missing from the full array. But this is fine — if the full conversation has no user query, that's an error regardless of truncation. The client sent a broken conversation.

### What About the Other Renderers? (Qwen3Coder, Qwen3VL, etc.)

The `Qwen3CoderRenderer` at `model/renderers/qwen3coder.go` has a fundamentally different internal structure: it pre-filters system messages into a separate array (lines 66-78) and works with a filtered message slice. Its `lastQueryIndex` computation is also different (simple backward scan for any user message, no `<tool_response>` checking). The validation checks cannot be shared as-is between renderers.

This is fine. Each renderer is responsible for validating the conventions of its specific model family. The Qwen 3.5 renderer validates Qwen 3.5's conventions; the Qwen3Coder renderer would validate Qwen3Coder's conventions. The checks are small (one-liners at existing code points), so the "duplication" between renderers is minimal and appropriate — each model has its own rules.

What SHOULD be shared is a common error type or error formatting convention, so that all renderers produce error messages with the same structure (identifying the offending message index, the specific violation, and the valid alternatives). This is a one-time decision, not a code-sharing problem.

### The HTTP Status Code Fix

All validation errors from the renderer should surface as HTTP 400 (Bad Request), not HTTP 500 (Internal Server Error). Currently, `Render()` errors would propagate through `chatPrompt()` → `ChatHandler()`, where they need to be caught and returned as HTTP 400.

The grammar pipeline error at `routes.go:2231-2234` should also change from `http.StatusInternalServerError` to `http.StatusBadRequest` — the client sent invalid tool definitions, not a server bug.

---

## Complete Validation Checklist

### Validations to add inside `Qwen35Renderer.Render()` (zero duplication — each reuses existing state):

| # | What to Validate | Where in `qwen35.go` | How | Official Template Equivalent |
|---|---|---|---|---|
| 1 | Unknown message roles | Lines 138-180, add `else` clause | `return "", fmt.Errorf(...)` | Rejection 4: `"Unexpected message role."` |
| 2 | System message not at index 0 | Lines 138, change behavior of `i != 0` case | `return "", fmt.Errorf(...)` | Rejection 2: `"System message must be at the beginning."` |
| 3 | No real user query found | Lines 114-127, check `multiStepTool` after loop | `return "", fmt.Errorf(...)` | Rejection 3: `"No user query found in messages."` |
| 4 | Images in system message | Lines 47-62, add role-aware check in `renderContent()` | `return "", 0, fmt.Errorf(...)` | Rejection 5: `"System message cannot contain images."` |
| 5 | Empty tool function name | Lines 149-166, check before concatenation | `return "", fmt.Errorf(...)` | No equivalent (implicit) |

### Validations to add in `ChatHandler` at `server/routes.go` (before rendering):

| # | What to Validate | Where | How |
|---|---|---|---|
| 6 | Grammar pipeline errors → HTTP 400 | Line 2231-2234 | Change `http.StatusInternalServerError` to `http.StatusBadRequest` |
| 7 | Renderer errors → HTTP 400 | Where `chatPrompt` errors are handled | Check for renderer validation errors, return HTTP 400 |

### Validations that are NOT worth adding (implicit assumptions, not explicit template rejections):

| # | What | Why Skip |
|---|---|---|
| — | Consecutive same-role messages (e.g., two user messages in a row) | The official template doesn't reject this either. It's an implicit training assumption. Adding this check could break legitimate client patterns where two user messages represent different modalities or multi-part input. |
| — | Tool messages not preceded by assistant tool calls | Same — the official template doesn't reject this. The template silently wraps tool messages under `<|im_start|>user` regardless of what precedes them. Rejecting this would require tracking cross-message dependencies (what the previous assistant message's tool calls were), adding significant complexity for a check the official template doesn't make. |
| — | Multiple system messages | Already caught by validation #2 (system message not at index 0). If a second system message appears at index 3, it triggers "system message at index 3 must be at index 0." |

### Where NOT to add validation (to avoid code duplication):

| Don't | Why Not |
|---|---|
| A separate `ValidateMessages()` function before `Render()` | Duplicates the message walk, `lastQueryIndex` computation, `renderContent()` calls, and role dispatch that `Render()` already does |
| A `Validate()` method on the `Renderer` interface | Same duplication, plus forces every renderer to implement a second method that walks the same data structure |
| Validation in `api.Message.UnmarshalJSON()` | Role validation there can't check ordering (it sees one message at a time, not the full array). Would reject `"tool"` role messages that are individually valid but only make sense in context. |
| Validation in `server/prompt.go` `chatPrompt()` | This function is renderer-agnostic — it doesn't know the model's specific conventions. Qwen 3.5's rules (system at position 0, no images in system) are different from what another model might allow. |

---

## Summary

The fork's client input validation is concentrated in the C++ grammar builder (12 error codes for tool definitions, unique to the fork and stronger than llama.cpp's equivalent) and absent everywhere else. The entire path from HTTP endpoint through JSON parsing through system message prepending through prompt rendering has no validation of conversation structure, message roles, message ordering, or content conventions.

The fix is 5 one-liner checks inside `Qwen35Renderer.Render()`, each placed at the exact code point where the relevant state is already computed. No separate validation function, no message re-walking, no code duplication. Plus one HTTP status code change from 500 to 400 for grammar pipeline errors.

llama.cpp avoids this problem architecturally by executing Jinja2 templates directly — the template IS the validator. But llama.cpp's approach is not available to Ollama because Ollama uses hand-written Go renderers instead of Jinja2 templates. The Go renderer approach is faster and more predictable, but it means validation must be added explicitly rather than inherited from the template.
