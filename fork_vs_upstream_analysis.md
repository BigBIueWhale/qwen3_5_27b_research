# Ollama Fork vs Upstream: Precise Analysis

Fork: `BigBIueWhale/ollama` @ `9ec17fc1` (7 commits atop `v0.17.4` merge)
Upstream: `ollama/ollama` master @ `82848a78`
llama.cpp: `ggml-org/llama.cpp` master @ `d969e933` — the canonical C++ inference engine. Ollama vendors GGML (the tensor math library) from llama.cpp but reimplements everything else in Go: chat template rendering is hardcoded Go renderers instead of llama.cpp's Jinja2 engine, sampling is a Go rewrite instead of llama.cpp's `llama-sampler.cpp`, and the model runner is `ollamarunner` instead of llama.cpp's server. This means llama.cpp bugs in GGML affect Ollama, but llama.cpp's correct Jinja2 template handling does NOT help Ollama — Ollama must reimplement the same logic in Go renderers/parsers.
Official reference: Qwen 3.5 Jinja2 template from `Qwen/Qwen3.5-35B-A3B` ([publicly accessible on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/tokenizer_config.json)). **Also verified** against `Qwen/Qwen3.5-27B` ([publicly accessible](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/tokenizer_config.json)) — both repos contain byte-identical `chat_template` fields.
Qwen3-Coder reference: `Qwen/Qwen3-Coder-30B-A3B-Instruct` Jinja2 template ([verified via public access](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)) — confirms XML tool definitions, system-first ordering. Upstream's `Qwen3CoderRenderer` faithfully matches it.

---

## Part 1: What the Fork Should Fix

### 1.1 CRITICAL: `SetInplace` in deltanet.go — crashes Apple Silicon & Vulkan on Ollama's vendored GGML

**Fault: Ollama's stale GGML vendor, not the fork's algorithm.** The fork author used `ggml_set_inplace()` — the same approach llama.cpp uses in its canonical GatedDeltaNet implementation (`delta-net-base.cpp:262` in `d969e933`). llama.cpp master has since added `GGML_OP_SET` support to Metal (`ggml-metal-device.m:1162`, `ggml-metal-ops.cpp:429`) and Vulkan (`ggml-vulkan.cpp:9037-9041`). But Ollama vendors an older GGML snapshot that only has `GGML_OP_SET_ROWS` on these backends — confirmed by searching Ollama's `ml/backend/ggml/ggml/src/ggml-metal/`: zero hits for `GGML_OP_SET`, only `GGML_OP_SET_ROWS`. The upstream Ollama developers worked around this by replacing `SetInplace` with a balanced concat tree (commit `3490e959`).

**File**: `model/models/qwen3next/deltanet.go:466-473`

The fork uses `v.SetInplace()` to assemble chunk outputs in `deltaNetChunked`. This maps to `GGML_OP_SET` via `C.ggml_set_inplace()` -> `ggml_set_impl(..., true)` -> `result->op = GGML_OP_SET`. Backend support **in Ollama's vendored GGML** vs **llama.cpp master (`d969e933`)**:

| Backend | Ollama's GGML `GGML_OP_SET` | llama.cpp master `GGML_OP_SET` | `GGML_OP_CONCAT` (both) |
|---------|---------------------------|-------------------------------|------------------------|
| CPU     | Yes                       | Yes                           | Yes                    |
| CUDA    | Yes                       | Yes                           | Yes                    |
| Metal   | **No**                    | **Yes** (`ggml-metal-device.m:1162`) | Yes           |
| Vulkan  | **No**                    | **Yes** (`ggml-vulkan.cpp:9037`) | Yes               |

The fork's approach is algorithmically correct and matches llama.cpp's reference implementation. The problem is purely that Ollama's vendored GGML is behind llama.cpp master. Until Ollama updates its GGML vendor (or backports the `GGML_OP_SET` Metal/Vulkan kernels), the concat tree workaround is necessary.

**What upstream does instead (commit `3490e959`)**: A balanced binary concat tree using `GGML_OP_CONCAT` (supported on all backends in all GGML versions):

```go
// Lines 496-507: Balanced pairwise reduction
for len(chunks) > 1 {
    merged := make([]ml.Tensor, 0, (len(chunks)+1)/2)
    for i := 0; i < len(chunks); i += 2 {
        if i+1 < len(chunks) {
            merged = append(merged, chunks[i].Concat(ctx, chunks[i+1], 2))
        } else {
            merged = append(merged, chunks[i])
        }
    }
    chunks = merged
}
```

The tree has **O(log N) graph depth** vs the linear chain of `SET` operations. For a 2048-token prompt with chunkSize=64, that's 32 chunks and 5 levels of merging.

**Fix**: Replace with upstream's balanced concat tree until Ollama updates its GGML vendor to include Metal/Vulkan `GGML_OP_SET` support. Alternatively, if Ollama updates GGML first, the fork's current approach would work as-is.

### 1.2 CRITICAL: Tool definition format — Qwen3.5 was trained on JSON, not XML

**Fault: Fork, but with a mitigating factor that is Alibaba Qwen's fault.** The fork author's commit `fbae6976` explicitly states: *"Qwen 3.5 was wired to the Qwen3VL JSON tool pipeline but was trained on the Qwen3-Coder XML format."* This is factually wrong — the Qwen3.5 official template uses JSON tool definitions (`{{ tool | tojson }}`), not XML. But the mistake is understandable because **Alibaba Qwen made the tool CALL format identical** between Qwen3-Coder and Qwen3.5 (both use `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` XML). A developer seeing XML tool calls would reasonably infer XML tool definitions — this is a confusing design choice by the Qwen team. The upstream Ollama developers got this right by creating a dedicated `Qwen35Renderer` that uses JSON definitions with `marshalWithSpaces`. **llama.cpp (`d969e933`) sidesteps this entirely** by executing the Jinja2 template from the GGUF directly (`chat.cpp:631-639`), passing tools as JSON objects to the template context (`chat.cpp:813-820`). The template's `{{ tool | tojson }}` produces correct JSON output with llama.cpp's native `tojson` filter (`value.cpp:157-183`). Ollama cannot benefit from this because it reimplements template rendering in Go instead of using llama.cpp's Jinja2 engine.

**File**: `model/renderers/qwen3coder.go:91-135`

Alibaba Qwen publishes **two different official templates** with a hybrid format split that is easy to miss:

| Template | Tool definitions | Tool calls (model output) | System+tools order |
|----------|-----------------|--------------------------|-------------------|
| **Qwen3-Coder** | XML `<function><name>...` | XML `<tool_call><function=...>` | System first, tools second |
| **Qwen3.5**     | JSON `{{ tool | tojson }}` | XML `<tool_call><function=...>` **(same!)** | Tools first, system appended after |

The tool call format is character-for-character identical. Only the tool *definition* rendering and system message ordering differ. This is **Alibaba Qwen's confusing design decision** — it's the reason the fork author made the wrong call. Verified by fetching both official templates: `Qwen/Qwen3.5-27B/tokenizer_config.json` uses `{{ tool | tojson }}`; `Qwen/Qwen3-Coder-30B-A3B-Instruct/tokenizer_config.json` uses the XML decomposition (`<function>\n<name>...`).

The fork routes `qwen3.5` through `Qwen3CoderRenderer` (at `renderer.go:59-60`: `case "qwen3.5": return &Qwen3CoderRenderer{isThinking: true, emitEmptyThinkOnNoThink: true}`), which uses Qwen3-Coder XML format for **both** models. Correct for `qwen3-coder`, wrong for `qwen3.5`. The model sees a completely different token sequence than what it was trained on:

```
Official:  {"function": {"description": "Get weather", "name": "get_weather", ...}, "type": "function"}
Fork:      <function>\n<name>get_weather</name>\n<description>Get weather</description>\n...
```

**Verified empirically**: Jinja2's `tojson` calls `json.htmlsafe_dumps()` which uses Python's default separators `(', ', ': ')` — **spaced JSON**. It also **sorts keys alphabetically** (confirmed: `"function"` before `"type"` at top level, `"description"` before `"name"` inside `function`). Upstream's `marshalWithSpaces` (at `json.go:8-45`) post-processes Go's `json.Marshal` output to add spaces after `:` and `,` — matching the separator style. However, Go's `json.Marshal` uses **struct declaration order** for keys, not alphabetical: it emits `{"type": "function", "function": {...}}` while tojson emits `{"function": {...}, "type": "function"}`. This key ordering difference is low-impact (LLMs handle JSON key order variability well) but worth noting for exact template fidelity.

The HTML-escaping behavior is identical: both Go's `json.Marshal` and tojson produce `\u003c`/`\u003e`/`\u0026` for `<`/`>`/`&`.

**Fix**: Either create a dedicated `Qwen35Renderer` (upstream's approach) or parameterize `Qwen3CoderRenderer` to emit JSON tool definitions when serving qwen3.5. Use `marshalWithSpaces` for the JSON serialization.

### 1.3 CRITICAL: System+tools ordering reversed for Qwen3.5

**Fault: Fork (collateral damage from 1.2).** This is a direct consequence of sharing the `Qwen3CoderRenderer` between both models. The Qwen3-Coder template genuinely does system-first — the fork author implemented the Coder format correctly, then applied it to the wrong model. The upstream Ollama developers got the ordering right in `Qwen35Renderer` because they created a dedicated renderer from the official Qwen3.5 template. llama.cpp (`d969e933`) gets this right automatically via Jinja2 template execution. Note: **Alibaba Qwen's decision to reverse the ordering between two models in the same family** (Coder = system first; 3.5 = tools first) is an unusual design choice that compounds the confusion from 1.2.

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

The fork renders system message first (`sb.WriteString(systemMessage)` at line 88), then tools (lines 90-136). The Qwen3-Coder template does it the fork's way (system first). So this is correct for qwen3-coder but wrong for qwen3.5.

Upstream's `Qwen35Renderer` does it correctly: tools block first (lines 90-99), system message appended after `</IMPORTANT>` (lines 100-107). Upstream even has a test at `qwen35_test.go:70-74` that explicitly asserts `systemIdx > toolsIdx`.

This matters for attention patterns — tools-first means the model's attention sees the tool schema before the system instruction, which is how it was trained. Reversing this changes the attention context for the system message and alters relative position encodings.

**Fix**: For `qwen3.5`, emit tools block first, append system message content after `</IMPORTANT>`.

### 1.4 MEDIUM: Missing `repeatPenalty <= 0` guard

**Fault: Fork (oversight).** The fork redesigned the sampler from scratch (see 3.1) and added guards for `temperature`, `topP`, `minP`, and `repeatLastN` in `NewSampler` (lines 159-179) but simply missed `repeatPenalty`. The upstream Ollama developers included it at `samplers.go:186-188`. This is a straightforward omission, not a design disagreement.

**File**: `sample/samplers.go`

The API (`api/types.go` `FromMap()`) accepts any float64 for `repeat_penalty` with no range validation. With `repeatPenalty = 0`: division by zero produces `+Inf` logits. With negative: penalty inverts, boosting repeated tokens.

**Fix**: `if repeatPenalty <= 0 { repeatPenalty = 1.0 }`

### 1.5 MEDIUM: Missing nil checks + Validate() for third-party GGUFs

**Fault: Fork (incomplete defensive coding).** The fork author contributed 3 commits specifically fixing third-party GGUF compatibility (ssm_dt.bias tensor name, head_count_kv scalar broadcast, architecture-based defaults) — all correct and valuable. But the fork lacks the defensive validation layer that would catch *other* third-party GGUF problems at load time rather than at inference time. The upstream Ollama developers added this in commit `8da09b1e`. llama.cpp (`d969e933`) achieves equivalent protection differently: `create_tensor()` at `llama-model.cpp:6779` uses flag `0` (REQUIRED), so missing tensors fail at model load time. No runtime nil checks needed — the loader refuses to proceed.

**Files**: `model/models/qwen3next/deltanet.go`, `model/models/qwen3next/model.go`

The fork explicitly targets third-party GGUF compatibility (3 commits fixing Unsloth/llama.cpp GGUFs). But its defensive validation is incomplete:

**Fork's current nil checks** in `GatedDeltaNet.Forward()`: Only `SSMQKV` and `SSMQKVGate` at `deltanet.go:103-105`, plus implicit nil checks in the `SSMBetaAlpha`/`SSMBeta`+`SSMAlpha` switch cases at lines 113 and 130. All other tensors (`SSMDT`, `SSMA`, `SSMConv1D`, `SSMNorm`, `SSMOut`) are accessed without nil guards — a missing tensor causes a nil-pointer panic at runtime (e.g., `alpha.Add(ctx, gdn.SSMDT)` at line 140).

**Fork's Validate()**: Does not exist. No `Validate` method anywhere in the fork's `model.go`. Search confirmed: zero hits for "Validate" in the file.

**Upstream's approach** (commit `8da09b1e`): Two layers of defense:
1. `Validate()` method on `Model` (`model.go:440-478`) — checks Options not nil, layer count matches isRecurrent flag count, then for each recurrent layer: SSMQKV, SSMQKVGate, SSMBetaAlpha (or SSMBeta+SSMAlpha), SSMDT, SSMA, SSMConv1D+Weight, SSMNorm, SSMOut. Implements the `model.Validator` interface, called by `model.New()` after tensor loading — catches problems at model load time, not inference time.
2. Inline nil checks in `GatedDeltaNet.Forward()` (`deltanet.go:138-149`) for `SSMDT`, `SSMA`, `SSMConv1D` (including `.Weight`), `SSMNorm`, `SSMOut` — right before first use at line 152.

Without these, a missing tensor in a third-party GGUF causes a nil-pointer panic at inference time instead of a clean error at model load. This is the difference between a cryptic stack trace and a message like `"qwen3next: layer 3 missing ssm_dt tensor"`.

**Fix**: Port upstream's nil checks (4 additional checks in deltanet.go lines 138-149) and the full `Validate()` method (lines 440-478).

### 1.6 MEDIUM: No image/vision support for Qwen3.5

**Fault: Fork (regression introduced in commit `fbae6976`).** Qwen3.5-27B is a **native multimodal early-fusion model** — it has a 27-layer vision encoder (1152 hidden size, 16 heads, patch size 16) built directly into its architecture, confirmed by `vision_config` in the [official config.json](https://huggingface.co/Qwen/Qwen3.5-27B/raw/main/config.json) with dedicated `vision_start_token_id` (248053), `vision_end_token_id` (248054), and `image_token_id` (248056). The official Ollama `qwen3.5:27b` tag (17GB, Q4_K_M) includes the vision encoder and supports image input. llama.cpp (`d969e933`) has full vision support via `tools/mtmd/models/qwen3vl.cpp` (193 lines) including dual-conv patch embedding, M-RoPE, deepstack feature merging, and multimodal projection.

**The fork actively broke image support that already existed.** At the merge base (`cc90a035`, Ollama `v0.17.4`), `qwen3.5` was routed to `Qwen3VLRenderer` with `useImgTags: RenderImgTags` — this renderer has a `renderContent()` method (lines 17-35 of `qwen3vl.go`) that iterates `content.Images` and inserts `[img-N]` tags or `<|vision_start|><|image_pad|><|vision_end|>` tokens. Image support **worked** at that point.

Commit `fbae6976` (diff to `renderer.go`) rewired `qwen3.5` from `Qwen3VLRenderer` to `Qwen3CoderRenderer`:
```diff
-  renderer := &Qwen3VLRenderer{isThinking: true, emitEmptyThinkOnNoThink: true, useImgTags: RenderImgTags}
-  return renderer
+  return &Qwen3CoderRenderer{isThinking: true, emitEmptyThinkOnNoThink: true}
```
The commit message states: *"Qwen 3.5 was wired to the Qwen3VL JSON tool pipeline but was trained on the Qwen3-Coder XML format."* The tool format fix was necessary (see 1.2), but the collateral damage was total loss of image support: `Qwen3CoderRenderer` has **no** `useImgTags` field, **no** `renderContent()` method, and **zero** references to `message.Images` (confirmed: 0 hits for `useImgTags`, `.Images`, and `renderContent` in `qwen3coder.go`). Images sent to a vision-capable Qwen3.5 GGUF through the fork are silently discarded.

**Upstream took the correct approach.** Commit `82848a78` also replaced `Qwen3VLRenderer` for `qwen3.5` — but with a **dedicated** `Qwen35Renderer` that fixes tool definitions (JSON via `marshalWithSpaces`) and system+tools ordering (tools-first) while **preserving** image support via its own `renderContent()` (lines 47-62 of `qwen35.go`). The `qwen3next` model (`model.go:665-668`) conditionally loads the vision encoder when `vision.block_count > 0` in the GGUF, and `EncodeMultimodal()` (`model.go:297-318`) processes images through `qwen3vl.VisionModel` — this works correctly for vision-capable GGUFs. For text-only GGUFs (where `Vision == nil`), `EncodeMultimodal` returns `ErrNoVisionModel` (`model.go:298-299`).

**File**: `model/renderers/qwen3coder.go` (fork), `model/renderers/renderer.go:59-60` (routing)

The official template handles three content types: `image` items (with `<|vision_start|><|image_pad|><|vision_end|>` and optional `Picture N:` prefix), `video` items (with `<|vision_start|><|video_pad|><|vision_end|>`), and `text` items. It also validates that system messages cannot contain images (raises exception). Upstream's `Qwen35Renderer` supports images but not videos (line 58: `// TODO: support videos`); the fork supports neither.

**Unsloth text-only GGUFs and the split-vision problem.** Ollama and llama.cpp take opposite approaches to vision packaging:
- **Ollama** bundles vision tensors into the main GGUF. The official `qwen3.5:27b` blob has 1307 tensors and `vision.block_count = 27` — the 27-layer vision encoder is embedded alongside the 64-layer text model. The `qwen3next` model loads it via struct tag `Vision *qwen3vl.VisionModel \`gguf:"v"\`` at `model.go:237`.
- **llama.cpp** uses a separate mmproj GGUF loaded via `--mmproj`. Unsloth follows this convention: text-only main GGUFs (851 tensors, no `vision.block_count`) plus separate `mmproj-{BF16,F16,F32}.gguf` files.
- **Ollama's Go engine explicitly rejects split vision**: `llm/server.go:149-152` checks `if len(projectors) == 0` and returns `errors.New("split vision models aren't supported")` when projectors are present. When pulling from HuggingFace (e.g., `FROM hf.co/unsloth/Qwen3.5-27B-GGUF:Q4_K_XL`), Ollama auto-downloads the mmproj alongside the main GGUF, classifies it as `"application/vnd.ollama.image.projector"` at `create.go:653` (because it has `vision.block_count > 0` and `block_count == 0`), adds it to `ProjectorPaths` at `images.go:319`, and then the Go engine rejects the entire load. This is the error the custom Modelfile works around — by using `FROM /tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf` (manually downloaded text-only GGUF), the mmproj is never pulled, `ProjectorPaths` stays empty, and the model loads successfully as text-only.

For text-only Unsloth GGUFs, the model layer behaves correctly: `model.go:572` checks `vision.block_count` (absent → 0), skips `NewVisionModel`, sets `Vision = nil`. If a user somehow sends an image, `EncodeMultimodal` returns `ErrNoVisionModel` at line 298. The renderer is irrelevant for text-only usage — neither the fork's `Qwen3CoderRenderer` (drops images) nor upstream's `Qwen35Renderer` (inserts tokens for images that would fail in `EncodeMultimodal` anyway) affects text-only inference.

**Upstream has secondary architectural issues** (not bugs, but design debt):
- The chat handler (`routes.go`) never checks `CapabilityVision` before accepting images — the error surfaces late in `EncodeMultimodal()` rather than at the API boundary. By contrast, llama.cpp's server validates `allow_image` at the API entry point (`server-common.cpp:970`: `"image input is not supported - hint: you may need to provide the mmproj"`) and advertises vision capability in its `/models` metadata (`server-context.cpp:3414`: `"modalities": {"vision": true}`). This missing capability advertisement is why [issue #14508](https://github.com/ollama/ollama/issues/14508) exists — third-party tools (Dify) can't detect that `qwen3.5:27b` supports vision.
- `sched.go:448` forces `numParallel = 1` for all `qwen35` models. This restriction exists because the hybrid recurrent architecture (GatedDeltaNet) is not safe with concurrent requests (ref: issue #4165), not because of vision — the same restriction applies to non-vision architectures like `lfm2`. However, it applies indiscriminately to text-only GGUFs that might otherwise be safe.

**Fix**: Add a `useImgTags` field and `renderContent()` method to `Qwen3CoderRenderer` (or create a dedicated `Qwen35Renderer` like upstream), and pass `RenderImgTags` when constructing the renderer at `renderer.go:60`. The model layer (`qwen3next/model.go`) already handles vision correctly — only the renderer routing is broken. For text-only Unsloth GGUFs this is a non-issue since no images will be processed.

### 1.7 LOW: Default system message injected when not provided

**Fault: Fork (collateral damage from sharing the Qwen3-Coder renderer).** The Qwen3-Coder official template genuinely injects this default. The Qwen3.5 official template does not — when no system message is provided, the system block contains only the tools text and instructions. The fork inherited the Coder behavior by routing both models through `Qwen3CoderRenderer`. Upstream's `Qwen35Renderer` correctly omits the default. llama.cpp (`d969e933`) gets this right via Jinja2 template execution — no injection.

**File**: `model/renderers/qwen3coder.go:83-85`

The fork injects `"You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."` when tools are present but no system message given. The official Qwen3.5 template simply omits the system message portion — the system block contains only the tools text.

**Fix**: Don't inject a default system message for qwen3.5.

### 1.8 LOW: `</think>` rendering for empty reasoning differs from training template

**Fault: Fork (minor template fidelity gap), partially mitigated by the fork's own `emitEmptyThinkOnNoThink` flag.** The fork does handle the generation-prompt case correctly — when `emitEmptyThinkOnNoThink: true` (set for qwen3.5 at `renderer.go:60`), the final generation prompt emits `<think>\n\n</think>\n\n` when thinking is disabled. The gap is only for **historical assistant messages** in multi-turn conversations where `message.Thinking == ""` — those don't get the empty think wrapper. The official template always emits it unconditionally for all assistant messages after `lastQueryIndex`. llama.cpp (`d969e933`) handles this correctly via the Jinja2 template, which unconditionally emits the think block.

**File**: `model/renderers/qwen3coder.go:163-166`

The fork only emits think blocks when `message.Thinking != ""`:
```go
if isThinking && message.Thinking != "" && i > lastQueryIndex {
    sb.WriteString("<think>\n" + strings.Trim(message.Thinking, "\n") + "\n</think>\n\n")
}
```

Upstream's `Qwen35Renderer` (lines 143-147) **always** emits the think block for eligible assistant messages, even with empty reasoning:
```go
if isThinking && i > lastQueryIndex {
    sb.WriteString(imStartTag + message.Role + "\n<think>\n" + contentReasoning + "\n</think>\n\n" + content)
}
```

The official template also always emits `<think>\n{reasoning}\n</think>\n\n` — unconditionally — for assistant messages after `last_query_index` in the current round. An empty reasoning produces `<think>\n\n</think>\n\n`.

Low practical impact — assistant messages with empty reasoning content are unusual in practice. But for exact template fidelity, the fork should emit the think block unconditionally.
### 1.9 ~~LOW~~ NON-ISSUE: `lastQueryIndex` doesn't check for tool_response wrappers

**Fault: Nobody — the fork's simpler approach is correct for Ollama's protocol.** The official Qwen3.5 Jinja2 template checks for `<tool_response>` wrappers inside user messages because HuggingFace's `transformers` library represents tool results as `role: "user"` messages with `<tool_response>...</tool_response>` content. But Ollama's API uses a distinct `role: "tool"` (confirmed: `api/types.go` line 243 normalizes `"TOOl"` → `"tool"`). The `lastQueryIndex` loop operates on **original `api.Message` roles before rendering** — tool messages have `Role == "tool"`, never `Role == "user"`, so the loop naturally skips them. The `<tool_response>` content check in upstream's `Qwen35Renderer` (`82848a78`, `qwen35.go:121`) and the fork's own `qwen3vl.go` (lines 61-74) is **redundant defensive code** — it guards against a scenario that Ollama's protocol makes impossible. Both the fork (`9ec17fc1`) and upstream (`82848a78`) renderers convert `role: "tool"` → rendered `<|im_start|>user\n<tool_response>...</tool_response><|im_end|>` in the output (fork: `qwen3coder.go:199-215`; upstream: `qwen35.go:172-179`), but this conversion happens AFTER `lastQueryIndex` is already computed.

**File**: `model/renderers/qwen3coder.go:149-155`

The fork's `lastQueryIndex` search finds the last `role == "user"` message, which correctly excludes `role == "tool"` messages without needing to inspect content:
```go
for i := len(filteredMessages) - 1; i >= 0; i-- {
    if filteredMessages[i].Role == "user" {
        lastQueryIndex = i; break
    }
}
```

The official template walks backward and **skips user messages whose content is a `<tool_response>...</tool_response>` wrapper** (with `|trim` before checking). This is necessary for the HuggingFace `transformers` protocol where tool results are user messages. It is unnecessary for Ollama's protocol where tool results are `role: "tool"` messages.

---

## Part 2: What Upstream Should Fix

### 2.1 CRITICAL: Prefill triggers on assistant messages with tool calls

**Fault: Upstream Ollama (`82848a78`).** All three upstream renderers — `qwen35.go`, `qwen3coder.go`, `qwen3vl.go` — have this bug. The fork (`9ec17fc1`) correctly fixed it in commit `fbae6976` by adding `&& len(message.ToolCalls) == 0` to the prefill condition. This is one of the fork's most valuable contributions. llama.cpp (`d969e933`) does not have this bug — it uses `add_generation_prompt` as a separate parameter passed to the Jinja2 template, which never conflates message closure with generation prompt skipping. Prefill is an Ollama-specific concept that llama.cpp does not need.

**Files**: `model/renderers/qwen35.go:136`, `model/renderers/qwen3coder.go:140`, `model/renderers/qwen3vl.go:82`

All three upstream renderers have:
```go
prefill := lastMessage && message.Role == "assistant"
```

This treats any last assistant message as a prefill, **including messages with tool calls**. The consequences differ by renderer:

**In `qwen35.go` and `qwen3vl.go`** — **doubly broken**: The `<|im_end|>` tag is gated by `if !prefill` (qwen35.go:169, qwen3vl.go:122), so when `prefill=true` on an assistant+toolcalls message, BOTH the `<|im_end|>` is omitted AND the generation prompt (`<|im_start|>assistant\n<think>\n`) is skipped. The rendered prompt ends with `</tool_call>` — no closing tag, no new start. The model's view of the conversation is corrupted.

**In `qwen3coder.go`** — the tool-call branch unconditionally writes `<|im_end|>` (line 156), but the post-loop generation prompt at lines 187-189 is gated by `if lastMessage && !prefill`, which is skipped since `prefill=true`. The model gets no `<|im_start|>assistant\n` to begin generating.

**When does this trigger?** Any time a client sends a chat request where the last message is an assistant message with tool calls populated:
- A client replays a conversation from saved history that ended at a tool-call boundary
- An agent framework constructs the message list with the assistant's tool-call message last
- The fork's own test suite (`qwen3coder_test.go:327-376`) proves this is an expected and tested input shape

The official Qwen3.5 Jinja2 template **always** emits `<|im_end|>` after every assistant message, regardless of tool calls. The `add_generation_prompt` flag is a separate concern — it controls whether `<|im_start|>assistant\n<think>\n` is appended at the end, independently of message closures. Ollama's prefill conflates two things (omitting `<|im_end|>` for streaming continuation, and skipping the generation prompt) that should be independent.

Note: Ollama's prefill mechanism is an Ollama-specific feature with no upstream Jinja2 equivalent. The official template has no concept of "incomplete last assistant message." This is fine as an additive feature, but it MUST be gated to only activate on plain assistant messages without tool calls.

The fork correctly adds `&& len(message.ToolCalls) == 0` in `qwen3coder.go:159` and `qwen3vl.go:82`.

### 2.2 HIGH: `formatToolCallArgument` uses compact JSON for object/array parameters

**Fault: Both upstream Ollama (`82848a78`) and the fork (`9ec17fc1`).** The `marshalWithSpaces` function already exists (upstream: `json.go:8-45`; fork: same file) and is used for tool **definitions**, but neither codebase uses it for tool call **arguments**. The fork acknowledged this at `qwen3coder.go:42-44` with a TODO comment but didn't fix it. Upstream never addressed it either. llama.cpp (`d969e933`) gets this right because the Jinja2 template uses `tojson` for tool call arguments too (e.g., `args_value | tojson | safe` for mappings/sequences), producing spaced JSON automatically.

**File**: `model/renderers/qwen3coder.go:235-257` (shared by all renderers via tool call rendering)

When a tool call parameter value is a map, slice, or array, `formatToolCallArgument` uses `json.Marshal()`, which produces **compact JSON**:
```
Go:       {"key":"value","nested":[1,2,3]}
```

The official template uses Jinja2's `tojson` (via `args_value | tojson | safe` for mappings/sequences), which produces **spaced JSON**:
```
Python:   {"key": "value", "nested": [1, 2, 3]}
```

This was verified empirically: Python's `json.dumps()` default separators are `(', ', ': ')`, and `tojson` uses these defaults. Simple string/number arguments render identically. Both the fork and upstream have this bug.

### ~~2.3~~ RETRACTED: Go's `json.Marshal` HTML-escaping

**Previous claim was WRONG.** Both Go's `json.Marshal` and Jinja2's `tojson` (which calls `json.htmlsafe_dumps`) produce the same `\u003c`/`\u003e`/`\u0026` unicode escapes for `<`, `>`, `&`. Verified empirically. The only real format differences are spacing (section 2.2) and key ordering (section 1.2 note).

### 2.4 MEDIUM: `</think>` not closed when content is empty in Qwen3VLRenderer

**Fault: Upstream Ollama (`82848a78`).** The fork (`9ec17fc1`) correctly fixed this in commit `fbae6976` — the `</think>` tag is always emitted after thinking content. Upstream's `Qwen3VLRenderer` conditionally gates it on `content != ""`, leaving an unclosed `<think>` tag for thinking-only responses. llama.cpp (`d969e933`) always closes `</think>` — the parser at `chat-parser-xml-toolcall.cpp:756` enforces closure, and the Jinja2 template itself always emits the closing tag.

**File**: `model/renderers/qwen3vl.go:96-100`

The exact difference — upstream (lines 95-103):
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

Upstream conditionally emits `\n</think>\n\n` only when `content != ""`. For thinking-only assistant messages (thinking content but empty visible content), the `<think>` tag is opened but `</think>` is never written.

This no longer affects qwen3.5 (which now uses `Qwen35Renderer`, which always closes `</think>` at line 144). But `Qwen3VLRenderer` is still used by `qwen3-vl-instruct` and `qwen3-vl-thinking`, where this bug remains live.

### 2.5 MEDIUM: `repeat_last_n` API parameter silently ignored in Go runner

**Fault: Upstream Ollama (`82848a78`).** The Go runner's `NewSampler` simply doesn't accept the parameter — the API field exists, the default exists (`api/types.go:1065`: `RepeatLastN: 64`), but nothing connects it to the sampler. The fork (`9ec17fc1`, commit `ab234955`) fixed this by adding `repeatLastN` as the 9th parameter to `NewSampler` and wiring `req.Options.RepeatLastN` through at `runner.go:899`. llama.cpp (`d969e933`) correctly wires `penalty_last_n` as a user-configurable parameter (`common.h:195`: `int32_t penalty_last_n = 64`) that flows through to `llama_sampler_init_penalties()`.

**File**: `sample/samplers.go`

Upstream's `NewSampler` signature (`sample/samplers.go:159`):
```go
func NewSampler(temperature float32, topK int, topP float32, minP float32,
    repeatPenalty float32, presencePenalty float32, frequencyPenalty float32,
    seed int, grammar *GrammarSampler) Sampler
```

No `repeatLastN` parameter. The constant `DefaultPenaltyLookback = 64` (line 19) is hardcoded in both `Accept()` (line 40: truncates history to 64 entries) and `tokenCounts()` in `transforms.go` (line 34: only looks at last 64). The runner call at lines 897-907 passes `req.Options.RepeatPenalty` etc. but NOT `req.Options.RepeatLastN`.

A user setting `repeat_last_n: 128` in the API has **zero effect** — the value flows through `api/types.go` into `req.Options.RepeatLastN` (default 64 at line 1065) but is never read by the Go sampler. It is a dead field.

**Caveat**: The value IS respected in the legacy C++ `llamarunner` at `runner/llamarunner/runner.go:651` (`RepeatLastN: req.Options.RepeatLastN`), which passes it to llama.cpp's C sampler. So this is a Go-runner-only regression, not a universal API lie. But the Go runner is the default for all new model architectures including qwen3next.

### 2.6 MEDIUM: `mropeInterleaved` defaults to `false` — wrong for third-party qwen35 GGUFs

**Fault: Upstream Ollama (`82848a78`).** The fork (`9ec17fc1`, commit `9ec17fc1`) correctly defaults to `c.Architecture() != "qwen3next"` at `model.go:548`, which evaluates to `true` for qwen35 — matching llama.cpp's approach. llama.cpp (`d969e933`) **hardcodes** the RoPE type per architecture in `llama_model_rope_type()` at `llama-model.cpp:9109-9113`: `case LLM_ARCH_QWEN35: return LLAMA_ROPE_TYPE_IMROPE` (value 40). No GGUF metadata lookup needed — the architecture enum determines the RoPE type at model load. This is a **silent data corruption bug** in upstream Ollama: the model loads, runs, and produces text, but the positional encoding is wrong in every full-attention layer. Users would blame the model quality, not the inference engine.

**File**: `model/models/qwen3next/model.go:641`

Upstream:
```go
mropeInterleaved: c.Bool("rope.mrope_interleaved", c.Bool("mrope_interleaved", false)),
```

Fork:
```go
mropeInterleaved: c.Bool("rope.mrope_interleaved", c.Bool("mrope_interleaved", c.Architecture() != "qwen3next")),
```

Third-party GGUFs (llama.cpp converter, Unsloth) do not write the ollama-internal `rope.mrope_interleaved` key (it is not in llama.cpp's GGUF constants). The actual Qwen3.5-27B model has `mrope_interleaved: true` (matching llama.cpp's hardcoded `LLAMA_ROPE_TYPE_IMROPE`).

**Concrete impact traced through code**: `mropeInterleaved` controls the RoPE type at `model.go:67-75`:
- `true` -> `rope.WithInterleaveMRoPE()` -> sets `opts.Type |= 1<<3 | 1<<5` = **40** = `GGML_ROPE_TYPE_IMROPE`
- `false` -> `rope.WithMRoPE()` -> sets `opts.Type |= 1<<3` = **8** = `GGML_ROPE_TYPE_MROPE`

With upstream's `false` default, all 16 full-attention layers in Qwen 3.5 27B use RoPE type 8 (MROPE) instead of 40 (IMROPE). These are completely different dimensional interleaving patterns for position encoding across the time/height/width sections. The model **silently produces wrong output** — it still runs, but attention patterns are misaligned across every full-attention layer.

Note: `ssm.v_head_reordered` defaults are functionally equivalent between fork (`c.Architecture() != "qwen3next"`) and upstream (`defaultVHeadReordered` which positively matches `"qwen35" || "qwen35moe"`). Both produce `true` for qwen35/qwen35moe and `false` for qwen3next.

### 2.7 LOW: `lastQueryIndex` initialization edge case + upstream Qwen3VL inconsistency

**Fault: Upstream Ollama (`82848a78`).** Two minor bugs in upstream's renderers. Neither the fork nor upstream validates the "no user messages" edge case that the official Qwen3.5 template explicitly rejects with `raise_exception('No user query found in messages.')`.

**File**: `model/renderers/qwen35.go:115`

Upstream initializes `lastQueryIndex := len(messages) - 1`. If there are zero user messages (e.g., `[system, assistant]`), it stays at the last index. The check `i > lastQueryIndex` on line 143 is never true, so no assistant message gets `<think>` blocks.

Low practical impact — conversations without user messages are extremely unusual.

**Additional finding**: Upstream has an inconsistency between its two `lastQueryIndex` implementations. `Qwen35Renderer` (qwen35.go:121) calls `strings.TrimSpace(content)` before checking tool-response prefixes, but `Qwen3VLRenderer` (qwen3vl.go:66) does NOT trim. As established in section 1.9, the tool_response content check is redundant in Ollama's protocol anyway (tool responses use `role: "tool"`, not `role: "user"`), so this inconsistency has zero practical impact.

### 2.8 LOW: `repeat_penalty` defaults to 1.0 — penalty system has zero effect with defaults

**Fault: Both upstream Ollama (`82848a78`) and llama.cpp (`d969e933`).** Both inherited the same design of feeding prompt tokens into the penalty window. Ollama does it via `Accept()` (`runner.go:700` and `runner.go:956`). llama.cpp does it via `common_sampler_accept()` on all prompt tokens during `init_sampler()` (`server-context.cpp:208-212`). Both default to `repeat_penalty = 1.0` (Ollama: `api/types.go:1066`; llama.cpp: `common.h:196`), making the penalty system a no-op. This is a **known anti-pattern**: the original repetition penalty paper (Keskar et al. 2019, CTRL, arXiv:1909.05858) defines the penalty over *"generated tokens g"* — not prompt tokens. HuggingFace's `transformers` library had the same bug and [acknowledged it as problematic](https://github.com/huggingface/transformers/issues/36642) (users reported *"weird strings"* and false penalization of phrases like *"let's think step by step"*), adding an opt-in fix in [PR #37625](https://github.com/huggingface/transformers/pull/37625) (April 2025). llama.cpp users reported the [same problem](https://github.com/ggml-org/llama.cpp/issues/331): *"the repetition penalty is affecting the anti-prompt and response prefix."* OpenAI's `frequency_penalty` and `presence_penalty` almost certainly apply to completion tokens only (community consensus, never officially contradicted).

Both upstream Ollama and llama.cpp are **trapped by this design**: any `repeatPenalty > 1.0` would penalize the model for generating tokens that appeared in the prompt — tool names, parameter names, JSON keys. For Qwen3.5 tool calling, where the system prompt says `get_weather` and the model must generate `get_weather` verbatim, this would actively degrade tool selection. So both default to `1.0` (identity), making the entire penalty infrastructure dead work: `Accept()`/`Reset()` run every forward pass, `applyPenalty()` computes `logit / 1.0 = logit`, zero effect. llama.cpp even comments this explicitly: `float penalty_repeat = 1.00f; // 1.0 = disabled` (`common.h:196`).

The fork (`9ec17fc1`, commit `ab234955`) removed the `Accept()` calls and records only generated tokens via the private `recordToken()` method (called from `Sample()` at lines 62 and 84). This matches the original paper's intent and makes the `1.1` default (`api/types.go:1066`) safe. One remaining nuance: the fork **does** penalize the model's own thinking tokens (they're generated), so if `<think>I need to call get_weather</think>` is shorter than 64 tokens, the `get_weather` tokens are still in the penalty window when the model emits the actual tool call. At `1.1` this is mild (~9% logit reduction), but users who raise the penalty to `1.5+` would break tool calling even with the fork's architecture.

---

## Part 3: Things That Look Neutral But Are Actually Critical

### 3.1 Prompt tokens excluded from penalty — the fork matches the original paper, upstream matches llama.cpp's mistake

**Fault: Both upstream Ollama (`82848a78`) and llama.cpp (`d969e933`). The fork (`9ec17fc1`, commit `ab234955`) is the only one correct per the original academic formulation.**

The repetition penalty was introduced by Keskar et al. (2019) in the CTRL paper (arXiv:1909.05858). The paper's formula operates over *"generated tokens g"* — the set of tokens the model has produced during inference. Prompt tokens are not in `g`. This is not ambiguous; the paper explicitly distinguishes the generation set from the conditioning context.

**Upstream Ollama** feeds prompt tokens into the penalty history via `Accept()` at **two call sites**:

1. **Sequence load** (`runner.go:951-957`): `seq.sampler.Reset()` (line 951) then loops through ALL cached inputs calling `seq.sampler.Accept(inp.Token)` (line 956). The last 64 prompt tokens become part of the penalty window.
2. **Batch processing** (`runner.go:694-701`): As pending input tokens are committed to the cache, each token is `Accept()`-ed (line 700). Both prompt tokens and generated tokens flow through this path (the generated token becomes an input for the next forward pass).

**llama.cpp** (`d969e933`) has the same behavior. The server's `init_sampler()` at `server-context.cpp:208-212` loops through ALL prompt tokens and calls `common_sampler_accept(smpl.get(), id, false)` on each one. This feeds prompt tokens into the penalty ring buffer (`llama-sampler.cpp:2644-2656`), same as Ollama upstream.

This is the same mistake HuggingFace `transformers` made (and [acknowledged as a bug](https://github.com/huggingface/transformers/issues/36642) — users reported *"weird strings"* and false penalization of common phrases, fixed in [PR #37625](https://github.com/huggingface/transformers/pull/37625), April 2025). llama.cpp users reported the [same problem](https://github.com/ggml-org/llama.cpp/issues/331): *"the repetition penalty is affecting the anti-prompt and response prefix."* All three projects (Ollama upstream, llama.cpp, HuggingFace) feed prompt tokens into the penalty window because it's the simplest implementation — you just call `accept()` on every token. The correct implementation requires distinguishing prompt tokens from generated tokens, which is what the fork does.

The fork's `runner.go` has **neither call site**. The sampler records tokens internally via the private `recordToken()` method (called only from `Sample()` at lines 62 and 84). No external code can feed prompt tokens into the penalty window because there is no public API to do so. This matches the original paper's intent and is the **correct** behavior for agentic tool-calling models where the prompt contains tool names the model must reproduce verbatim. The fork is the only implementation among all four (Ollama upstream, llama.cpp, HuggingFace pre-fix, Ollama fork) that gets this right by default.

### 3.2 Ring buffer is independent of KV cache state — this is a correctness property

**Fault: Upstream Ollama (`82848a78`) couples penalty state to KV cache operations; the fork (`9ec17fc1`) correctly decouples them.**

**Fork's architecture** (`sample/samplers.go:19-32`): The `Sampler` struct contains `recentTokens []int32` (ring buffer), `recentTokenPos int` (write cursor), and `repeatLastN int` (window size from API, user-configurable). `recordToken()` (lines 196-206) implements a classic circular buffer: O(1) per token, zero allocations after warmup, permanently bounded at `repeatLastN` entries. The runner's only interaction is `seq.sampler.Sample(logits)` at line 760 — it never touches internal state.

**llama.cpp's architecture** (`d969e933`, `llama-sampler.cpp:23-132`): Uses a proper `ring_buffer<llama_token>` template class with the same design — fixed capacity, O(1) push, modular arithmetic. The ring buffer is initialized with `penalty_last_n` capacity at `llama-sampler.cpp:2763`. The key difference: llama.cpp's `llama_sampler_penalties_accept()` is a **public** function called by external code (the server feeds prompt tokens through it), while the fork's `recordToken()` is **private** and called only from `Sample()`. Both use ring buffers, but the fork's encapsulation prevents the prompt-token contamination problem (see 3.1).

**Upstream's architecture** (`sample/samplers.go:21-32`): The `Sampler` struct contains `history []int32` (an append-then-truncate slice). `Accept()` (lines 38-43) appends the token, then if `len(history) > DefaultPenaltyLookback`, copies the last 64 entries to the front of the slice and truncates — a **slide-and-truncate** pattern (O(n) per trim, allocates on growth). `Reset()` (line 36) sets history to nil. The runner calls these externally at 3 separate call sites (cache load at line 951+956, pending inputs at line 700, reprocess error at line 565).

**Cache shift behavior**: On normal cache shift, neither codebase touches the sampler. On the reprocess error path, upstream calls `seq.sampler.Reset()` (line 565), clearing the entire history — the reprocessed tokens will be `Accept()`-ed back through `pendingInputs`. The fork does nothing to the ring buffer on either path.

**Why not resetting is correct**: Cache shift is a memory management operation, not a semantic boundary. The tokens in the ring buffer are tokens the model actually generated recently. Clearing the ring buffer on cache shift would destroy repetition penalty information at the exact moment it matters most — during long generations that have filled the context window.

An earlier version of this report listed "ring buffer not cleared on cache shift" as a fork bug. This was **wrong**. Not resetting is the correct behavior.

### 3.3 Default RepeatPenalty 1.1 vs 1.0 — enables the penalty system to actually work

This is not a policy preference. With upstream Ollama's and llama.cpp's 1.0 default:
- The penalty math is identity (no effect on any logit)
- The Accept/Reset bookkeeping is wasted work
- The `repeat_last_n` API parameter is doubly useless (hardcoded to 64 in upstream, AND the penalty is 1.0)
- llama.cpp explicitly comments: `float penalty_repeat = 1.00f; // 1.0 = disabled` (`common.h:196`)

With the fork's 1.1 default:
- Repeated positive logits are divided by 1.1 (~9% reduction); negative logits are multiplied by 1.1 (10% further suppression)
- The ring buffer actually serves a purpose
- The `repeat_last_n` parameter controls something real
- This is safe ONLY because the fork excludes prompt tokens (see 3.1) — with prompt tokens in the window (as in llama.cpp and upstream), 1.1 would degrade tool calling

The exact penalty math in all three codebases (fork `transforms.go:134-139`, upstream `transforms.go:50-57`, llama.cpp `llama-sampler.cpp:2690-2696`) is identical:
```
if logit > 0: logit /= repeatPenalty
if logit < 0: logit *= repeatPenalty
```

**Additional difference**: Upstream's `tokenCounts()` (`transforms.go:40`) performs bounds checking on token IDs: `if token < 0 || int(token) >= vocabSize { continue }`. The fork's `applyPenalties()` does NOT validate token IDs — if a corrupted or garbage token ID enters the ring buffer, it creates bogus entries in the penalty map. This is a minor robustness gap.

### 3.4 Unconditional KV emission — prevents a real default-mismatch bug

**Fork converter** (`convert/convert_qwen3next.go`): Always writes both keys unconditionally:
```go
kv["rope.mrope_interleaved"] = q.RopeParameters.MRopeInterleaved   // line 287
kv["ssm.v_head_reordered"] = q.shouldReorderVHeads()                // line 314
```

**Upstream converter**: Only writes when `true`:
```go
if q.RopeParameters.MRopeInterleaved { kv["rope.mrope_interleaved"] = true }  // lines 287-289
if q.shouldReorderVHeads() { kv["ssm.v_head_reordered"] = true }              // lines 316-318
```

For `rope.mrope_interleaved`: upstream's conditional emission is fine for Ollama-converted GGUFs (the converter writes `true` for qwen35, which is correct). The risk is for third-party GGUFs that omit the key — upstream defaults to `false` (wrong for qwen35), the fork defaults based on architecture (correct). See section 2.6.

Note: `ssm.v_head_reordered` and `rope.mrope_interleaved` are **ollama-internal conventions**. They do not exist in llama.cpp's GGUF constants (`gguf-py/gguf/constants.py`, confirmed in `d969e933`). llama.cpp sidesteps the problem entirely: RoPE type is hardcoded per architecture in `llama_model_rope_type()` (`llama-model.cpp:9109-9113`), and V-head reordering is done **physically during GGUF conversion** by `_LinearAttentionVReorderBase` (`convert_hf_to_gguf.py:4782-4791`) — the converter permutes tensor weights so no runtime flag is needed. Third-party GGUFs from Unsloth, bartowski, or llama.cpp's converter will **never** contain these keys.

For `ssm.v_head_reordered`: the runtime defaults are equivalent between fork and upstream (`true` for qwen35/qwen35moe, `false` for qwen3next). The unconditional emission is defense-in-depth.

### 3.5 Recurrent layer inference — fork's formula matches llama.cpp, upstream matches the full logic flow

The fork's 4-line interval computation (`model.go:472-476`):
```go
interval := int(c.Uint("full_attention_interval", 4))
isRecurrent = make([]bool, numLayers)
for i := range numLayers {
    isRecurrent[i] = (i+1)%interval != 0
}
```

This formula `(i+1)%interval != 0` is identical to llama.cpp's (`d969e933`) `((i + 1) % full_attn_interval != 0)` at `llama-model.cpp:2526` (for `LLM_ARCH_QWEN35`). llama.cpp reads `full_attention_interval` from GGUF metadata via `ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false)` with a default of 4, then populates `hparams.recurrent_layer_arr[i]` with the same formula. The fork does NOT replicate llama.cpp's outer guard (`if (recurrent_layer_arr.empty())`) — llama.cpp only falls back to interval computation if the per-layer array wasn't already populated from GGUF metadata.

Upstream's `inferRecurrentLayers()` (52 lines, `model.go:497-549`) is closer to llama.cpp's complete behavior:
1. **Primary**: If `headCountKV` is a per-layer array with a mix of zero (recurrent) and non-zero (full attention) values, use it directly.
2. **Fallback**: If `headCountKV` is uniform (scalar broadcast from third-party GGUFs), fall through to the `full_attention_interval` computation. Also validates edge cases (`interval > numLayers`, `interval <= 0`).

For all real-world GGUFs (Ollama-converted and third-party), both produce identical results:
- **Ollama-converted**: Per-layer `headCountKV` array triggers upstream's primary path, fork's interval computation produces the same pattern.
- **Third-party**: Scalar `headCountKV` triggers upstream's fallback, which uses the same `(i+1)%interval != 0` formula as the fork.

The difference matters only if someone produces a GGUF with a non-uniform `headCountKV` array that deviates from the interval pattern — upstream would honor the array, the fork would override it.

The fork's approach is actually **more robust for third-party GGUFs** because it has zero dependency on `headCountKV` — it goes straight to the interval formula. The upstream's approach adds an extra indirection through `headCountKV` that only matters for Ollama's own GGUFs.

### 3.6 BOS/EOS bug at vocabulary.go:57 — confirmed logging-only, no functional impact

```go
// Line 57 — Should be v.EOS, not v.BOS
if len(ids) > 0 && slices.Contains(v.BOS, ids[len(ids)-1]) {
    slog.Warn("adding eos token to prompt which already has it", "id", v.EOS)
}
```

**Why it is purely cosmetic**: The `if` block containing `slog.Warn` only controls the warning message. The actual `append(ids, v.EOS[0])` at line 61 runs **unconditionally** — the EOS token is always added correctly regardless of the bug.

**Why it never fires for Qwen**: `AddEOS` defaults to `false` for qwen3next (`model.go:686`: `AddEOS: c.Bool("tokenizer.ggml.add_eos_token", false)`), so the entire EOS block at lines 55-62 is never entered.

Both codebases have this bug. Zero priority for either.

---

## Part 4: Tokenizer Performance (All Validated Correct)

1. **Binary search in prompt.go**: O(log N) tokenize calls. Monotonicity assumption valid (dropping messages can only reduce token count).
2. **Special token splitting in special.go**: Equivalent behavior with `strings.Contains` early-out. 13 test cases.
3. **Stack buffer in vocabulary.go Merge()**: 128-byte stack buffer eliminates heap allocations in BPE merge hot path. Leverages Go 1.12+ map-lookup compiler optimization.

---

## Part 5: Architectural Divergence — Parser/Renderer Strategy

Upstream and the fork solved the qwen3.5 parser/renderer problem with fundamentally different strategies:

**Upstream** (commit `82848a78`): Created entirely new dedicated files — `model/parsers/qwen35.go` (`Qwen35Parser`) and `model/renderers/qwen35.go` (`Qwen35Renderer`).

The `Qwen35Parser` is a clean two-layer architecture: it has its own 3-state thinking extraction machine (`CollectingThinking`, `ThinkingDoneEatingWhitespace`, `CollectingContent`), and delegates all post-thinking content (including tool calls) to an embedded `Qwen3CoderParser` initialized with `nil` thinkValue and `nil` lastMessage — this means the inner tool parser always starts in `LookingForToolStart` state with `hasThinkingSupport = false`. Thinking is handled by the outer layer, tool calls by the inner layer.

The `Qwen35Parser.Init()` also checks `assistantPrefill` (`lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != ""`) to skip thinking mode when the last message is an assistant prefill. The fork's `Qwen3CoderParser.Init()` does NOT check for assistant prefill.

The `Qwen35Renderer` includes image support via `renderContent()`, correct tools-first ordering, `marshalWithSpaces` for JSON tool definitions, lastQueryIndex with tool_response filtering and TrimSpace, and unconditional `</think>` closure. The existing `Qwen3CoderParser`/`Qwen3CoderRenderer` are left unchanged — still no thinking support, still XML tool definitions, still system-first ordering. Upstream's `Qwen3CoderParser` has only 2 states (no thinking states at all), and `HasThinkingSupport()` returns `false`.

**Fork**: Extended the existing `Qwen3CoderParser`/`Qwen3CoderRenderer` with thinking fields and states. Both `qwen3.5` and `qwen3-coder` route through the same code, so both benefit from all fixes. The parser uses `hasThinkingSupport: true, defaultThinking: true` for qwen3.5 (at `parsers.go:52-53`). The parser has 4 states (2 extra thinking states: `CollectingThinking`, `ThinkingDoneTransition`) crammed into the same state machine as tool call parsing.

**Practical consequence**: In upstream, fixes to thinking/whitespace/lastQueryIndex exist for qwen3.5 (via `Qwen35Renderer`) but `qwen3-coder` still has wrong whitespace, no lastQueryIndex, and no thinking support. In the fork, both models share all fixes. On the other hand, the fork's shared approach means qwen3.5 gets the wrong tool definition format (XML instead of JSON), wrong system+tools ordering, no image support, and a spurious default system message — collateral damage from sharing code with qwen3-coder.

**Assessment**: The upstream's strategy is architecturally cleaner and results in each model getting exactly the right behavior. The fork's strategy shares improvements but creates template-specific bugs for qwen3.5. For the ideal implementation, create a dedicated `Qwen35Renderer` and `Qwen35Parser` that take the best from both.

**The llama.cpp contrast**: llama.cpp (`d969e933`) avoids this entire problem class by executing the Jinja2 template from the GGUF file directly (`chat.cpp:631-639`). Each model carries its own template, so there is no renderer routing, no shared renderer bugs, no format mismatches. Qwen3-Coder gets XML definitions because its template says so; Qwen3.5 gets JSON definitions because its template says so. The entire Part 1 sections 1.2, 1.3, 1.7, 1.8 — all collateral damage from Ollama's Go renderer architecture — simply do not exist in llama.cpp. Ollama chose to reimplement template rendering in Go (presumably for performance and control), but the cost is exactly this class of bugs: every model that deviates from the Qwen3-Coder format needs a new dedicated renderer. llama.cpp's Jinja2 approach trades some performance for correctness-by-construction.

---

## Part 6: The Ideal Implementation — Best of Both Worlds

**Target model**: Qwen 3.5 27B (`Qwen/Qwen3.5-27B`), running from the Unsloth Dynamic 2.0 `UD-Q4_K_XL` quantized GGUF (`unsloth/Qwen3.5-27B-GGUF`, commit `30d153c8`, 17.6 GB, 851 tensors). This GGUF was downloaded to `/tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf`.

**Target inference engine**: Ollama's Go runner (the default runner for all new model architectures including `qwen3next`/`qwen35`).

**Goal**: An Ollama implementation that produces inference output **identical** to what the Qwen 3.5 27B model would produce if run through its official HuggingFace Jinja2 chat template with correct RoPE configuration, correct layer type classification, correct V-head ordering, correct penalty sampling semantics, and correct thinking/tool-call parsing — taking the best components from both the `BigBIueWhale/ollama` fork (commit `9ec17fc1`) and the `ollama/ollama` upstream (commit `82848a78`).

Every subsection below explains: what the correct behavior is, who implemented it correctly (the fork author, the upstream Ollama developers, or llama.cpp `d969e933`), why the alternative is wrong, and what breaks if you get it wrong — all grounded in the specific metadata and tensor layout of the `Qwen3.5-27B-UD-Q4_K_XL.gguf` file.

---

### 6.1 GatedDeltaNet Chunk Assembly: SetInplace (from llama.cpp) vs Balanced Concat Tree (Ollama workaround)

**Who got it right**: llama.cpp (`d969e933`) and the fork both use `ggml_set_inplace()` (`delta-net-base.cpp:262` and `deltanet.go:466`). The upstream Ollama developers used a balanced concat tree (commit `3490e959`) as a **workaround** for their stale GGML vendor.

**The core issue**: Ollama vendors GGML from llama.cpp but does not track llama.cpp master closely. llama.cpp master has added `GGML_OP_SET` support to Metal (`ggml-metal-device.m:1162`, `ggml-metal-ops.cpp:429`) and Vulkan (`ggml-vulkan.cpp:9037-9041`). Ollama's vendored GGML does NOT have these kernels — confirmed: searching Ollama's `ml/backend/ggml/ggml/src/ggml-metal/` finds only `GGML_OP_SET_ROWS`, not `GGML_OP_SET`.

**Backend support comparison**:

| GGML Backend | Ollama's GGML `GGML_OP_SET` | llama.cpp master `GGML_OP_SET` | `GGML_OP_CONCAT` (both) |
|-----|-----|-----|-----|
| CPU | Yes | Yes | Yes |
| CUDA | Yes | Yes | Yes |
| Metal | **No** | **Yes** (`ggml-metal-device.m:1162`) | Yes |
| Vulkan | **No** | **Yes** (`ggml-vulkan.cpp:9037`) | Yes |

The fork author followed the llama.cpp reference implementation. On Ollama's vendored GGML, this breaks Apple Silicon and Vulkan. The upstream Ollama developers' concat tree workaround is necessary until Ollama updates its GGML vendor.

**What breaks on Ollama's current GGML**: On any Apple Silicon Mac, the Metal backend's `ggml_metal_device_supports_op` returns `false` for `GGML_OP_SET`. The GGML scheduler may fall back to CPU, but the tensor buffer is in Metal GPU memory (`MTLBuffer`). Attempting to dereference it as a CPU pointer causes `EXC_BAD_ACCESS`, silent data corruption, or a "buffer-less intermediate" failure.

**What the balanced concat tree does instead**: Upstream collects chunk outputs into a slice, merges via balanced binary `Concat` tree (lines 496-507). `GGML_OP_CONCAT` is supported on all backends in all GGML versions. O(log N) graph depth vs linear SET chain.

**Fix for the fork**: Either (a) replace with upstream's concat tree until Ollama updates GGML, or (b) backport the `GGML_OP_SET` Metal/Vulkan kernels from llama.cpp into Ollama's vendored GGML. Option (a) is simpler and safer.

---

### 6.2 Tool Definition Format: Spaced JSON Objects (from llama.cpp and Upstream)

**Who got it right**: llama.cpp (`d969e933`) gets this right automatically by executing the Jinja2 template from the GGUF (`chat.cpp:631-639`), passing tools as JSON objects to the template context (`chat.cpp:813-820`). The upstream Ollama developers, in commit `82848a78`, manually reimplemented this in Go with a dedicated `Qwen35Renderer` using `marshalWithSpaces(tool)`.
**Who got it wrong**: The fork, which routes `qwen3.5` through `Qwen3CoderRenderer` (at `renderer.go:59-60`) and renders tool definitions in XML format.

**What the official Qwen 3.5 training template specifies**: The Qwen team at Alibaba published the official Jinja2 chat template in `Qwen/Qwen3.5-35B-A3B/tokenizer_config.json` on HuggingFace (publicly accessible, no authentication required). Line 49 of the template renders each tool as `{{ tool | tojson }}`. Jinja2's `tojson` filter calls Python's `json.htmlsafe_dumps()`, which uses Python's default separators `(', ', ': ')` — producing **spaced JSON** with spaces after every colon and comma. It also **sorts keys alphabetically** and HTML-escapes `<`, `>`, `&` to `\u003c`, `\u003e`, `\u0026`.

The tools block is wrapped in `<tools>...</tools>` XML tags, with the preamble text `"# Tools\n\nYou have access to the following functions:\n\n<tools>"` before the JSON objects and instruction text after `</tools>`. Each tool appears on its own line preceded by `\n`.

Verified empirically by running `tojson` through a Python 3.12 Jinja2 3.1.6 environment (at `/tmp/qwen35-check/`):
```python
# Input:  {"type": "function", "function": {"name": "get_weather", ...}}
# Output: {"function": {"description": "Get weather", "name": "get_weather", ...}, "type": "function"}
#          ^ alphabetical key order     ^ spaces after : and ,
```

**What the fork does instead**: The fork's `Qwen3CoderRenderer` renders tools in a structured XML decomposition (lines 91-135 of `qwen3coder.go`):
```
<function>
<name>get_weather</name>
<description>Get the current weather</description>
<parameters>
<parameter>
<name>location</name>
<type>string</type>
...
```

This XML format is correct for the **Qwen3-Coder** model family (`Qwen/Qwen3-Coder-8B-Instruct`), which was trained on XML tool definitions. But the **Qwen 3.5** model family was trained on the JSON format. These are two completely different Qwen model families with different templates.

**What the upstream `marshalWithSpaces` function does**: At `model/renderers/json.go:8-45`, the upstream's `marshalWithSpaces` takes Go's `json.Marshal` output (compact JSON) and post-processes it to insert a space after every `:` and `,` that appears outside of string values. This matches Jinja2's separator style. Go's `json.Marshal` also HTML-escapes `<`/`>`/`&` to the same `\u003c`/`\u003e`/`\u0026` sequences as Jinja2's `tojson`.

One minor remaining difference: Go's `json.Marshal` serializes struct fields in **declaration order** (the `Tool` struct at `api/types.go:319-323` puts `"type"` before `"function"`), while Jinja2's `tojson` sorts keys **alphabetically** (`"function"` before `"type"`). This key ordering difference is low-impact — large language models handle JSON key order variability well — but it means even upstream's output is not byte-for-byte identical to the training template. It is, however, semantically equivalent, which is sufficient.

**What breaks if you don't fix this**: When the `Qwen3.5-27B-UD-Q4_K_XL.gguf` model processes a tool-calling prompt, it sees XML tokens (`<function>`, `<name>`, `<description>`, `<parameters>`) where it was trained to see JSON tokens (`{"type":`, `"function":`, `{"name":`). The model's attention patterns over the tool schema are misaligned, reducing tool selection accuracy and potentially causing the model to generate malformed tool calls, refuse to use tools, or hallucinate tool names that don't exist. The severity depends on how much the model's tool-calling behavior is sensitive to the exact prompt format — for a 27-billion-parameter model trained on a specific template, format deviations in the tool schema are significant.

---

### 6.3 System Message and Tools Ordering: Tools First, System After (from Upstream)

**Who got it right**: llama.cpp (`d969e933`) via Jinja2 template execution, and the upstream Ollama developers in the `Qwen35Renderer` (lines 90-107 of `qwen35.go`).
**Who got it wrong**: The fork, which uses `Qwen3CoderRenderer` and puts the system message first.

**What the official Qwen 3.5 training template specifies**: When both tools and a system message are present, the template renders a single `<|im_start|>system\n...` block with this exact structure:

```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{...tool 1 JSON...}
{...tool 2 JSON...}
</tools>

If you choose to call a function ONLY reply in the following format...
<IMPORTANT>
Unless you are directly asked by the user, do NOT call any functions...
</IMPORTANT>

[user's system message here, if provided]
<|im_end|>
```

Tools instruction block comes **first**. The user's system message is **appended after** the `</IMPORTANT>` tag, separated by `\n\n`. If no system message was provided by the user, nothing is appended — there is no default system message injected.

**What the fork does instead**: The `Qwen3CoderRenderer` writes `sb.WriteString(systemMessage)` at line 88 — the system message goes first — then appends the tools block starting at line 90 with `\n\n# Tools\n\n...`. Additionally, if no system message is provided but tools are present, the fork injects a default: `"You are Qwen, a helpful AI assistant that can interact with a computer to solve tasks."` (lines 83-85). Neither the ordering nor the default injection matches the Qwen 3.5 training template.

Note: this system-first ordering and default injection IS correct for the Qwen3-Coder template (the `Qwen3CoderRenderer` was originally written for that model). The bug is that the fork shares this renderer for qwen3.5.

**What breaks if you don't fix this**: The ordering matters for the transformer's attention mechanism. The Qwen 3.5 27B model has 24 attention heads in each of its 16 full-attention layers and 16 key-heads in each of its 48 GatedDeltaNet layers. When the model processes the system prompt, each attention head builds key-value representations of the tokens it has seen so far. If the tool schema appears at positions 0-200 and the system message at positions 200-250 (upstream's order), the attention keys encode tool schema information in early positions. If reversed (fork's order), the system message occupies early positions and tools come later. Since the model was trained with tools-first ordering, its learned attention patterns expect to find tool definitions at low position indices within the system block. Reversing this reduces the model's ability to correctly reference the tool schema during generation.

---

### 6.4 Prefill Detection: Must Exclude Tool-Call Messages (from Fork)

**Who got it right**: The fork author, who added `&& len(message.ToolCalls) == 0` to the prefill condition in both `qwen3coder.go:159` and `qwen3vl.go:82`. llama.cpp (`d969e933`) does not have this bug because it uses `add_generation_prompt` as a separate Jinja2 parameter — the template never conflates message closure with generation prompt skipping.
**Who got it wrong**: All three upstream Ollama renderers (`qwen35.go:136`, `qwen3coder.go:140`, `qwen3vl.go:82`), which use `prefill := lastMessage && message.Role == "assistant"` without checking for tool calls.

**What Ollama's prefill mechanism does**: When the last message in a chat completion request is an assistant message, Ollama treats it as a "prefill" — the assistant's content is emitted without a closing `<|im_end|>` tag, and no new `<|im_start|>assistant\n<think>\n` generation prompt is appended. This allows the model to continue generating from where the assistant left off, as if the assistant's message were incomplete. This is an Ollama-specific feature with no equivalent in the official Jinja2 template or llama.cpp.

**Why it must exclude tool calls**: When the last assistant message contains tool calls (e.g., the model called `get_weather` and is waiting for the tool response), Ollama's upstream prefill logic incorrectly treats it as an incomplete assistant message. In the `Qwen35Renderer` and `Qwen3VLRenderer`, this causes **two** things to break simultaneously:
1. The `<|im_end|>` after the assistant's tool call is **skipped** (gated by `if !prefill` at qwen35.go:169 / qwen3vl.go:122)
2. The generation prompt `<|im_start|>assistant\n<think>\n` is **also skipped** (gated by the same `!prefill` check at qwen35.go:175-181 / qwen3vl.go:130)

The rendered prompt ends with raw `</tool_call>` text — no turn closure, no new turn opening. The Qwen 3.5 27B model sees a malformed conversation structure and cannot generate a coherent response.

The official Qwen 3.5 Jinja2 template **always** emits `<|im_end|>` after every assistant message, regardless of whether it contains tool calls. The `add_generation_prompt` flag is a completely separate concern that controls whether `<|im_start|>assistant\n<think>\n` is appended at the very end. The template never conflates turn closure with generation prompt skipping.

**What breaks if you don't fix this**: Any agentic workflow where the client sends a message list ending with the assistant's tool-call message (common in agent frameworks that replay conversation history, or when a client saves and restores a conversation that was interrupted at a tool-call boundary) will produce corrupted prompts. The Qwen 3.5 27B model will receive an unclosed assistant turn and generate garbage, error responses, or refuse to respond.

---

### 6.5 `</think>` Closure: Must Always Close (from Fork's Qwen3VL Fix)

**Who got it right**: The fork author, in the `Qwen3VLRenderer` at `qwen3vl.go:95-103`, where `</think>` is always emitted after thinking content, even when visible content is empty. llama.cpp (`d969e933`) also always closes `</think>` — the Jinja2 template unconditionally emits it, and the parser at `chat-parser-xml-toolcall.cpp:756` enforces closure.
**Who got it wrong**: Upstream's `Qwen3VLRenderer` (`qwen3vl.go:95-103`), which only emits `\n</think>\n\n` when `content != ""`.

**What the official Qwen 3.5 training template specifies**: For assistant messages after `last_query_index`, the template unconditionally emits `<think>\n{reasoning}\n</think>\n\n{content}` — the `</think>` tag is always present, even if reasoning is empty (producing `<think>\n\n</think>\n\n`). There is no path in the template that opens `<think>` without closing it.

**What breaks if you don't fix this**: When an assistant message has thinking content but empty visible content (a thinking-only response), the unclosed `<think>` tag means the Qwen 3.5 27B model's tokenizer receives `<|im_start|>assistant\n<think>\nsome reasoning here` with no `</think>`. The model's understanding of conversation structure is corrupted — it may interpret subsequent messages as still being inside the thinking block, leading to malformed output where thinking content leaks into visible responses.

Note: the `Qwen35Renderer` in upstream (used for qwen3.5 specifically) DOES always close `</think>` at line 144. This bug only remains live in the `Qwen3VLRenderer` used for `qwen3-vl-instruct` and `qwen3-vl-thinking`. The fork's fix is correct and should be preserved when creating the ideal implementation.

---

### 6.6 lastQueryIndex with Tool-Response Filtering and TrimSpace (from Upstream)

**Who got it right**: llama.cpp (`d969e933`) via Jinja2 template execution (the template's `last_query_index` logic runs natively), and the upstream Ollama developers in the `Qwen35Renderer` (lines 114-127 of `qwen35.go`).
**Who got it wrong**: The fork, which uses a simple "find last user message" loop without filtering tool responses (lines 149-155 of `qwen3coder.go`).

**What `lastQueryIndex` controls**: In the Qwen 3.5 chat template, `lastQueryIndex` identifies the last **real user query** (not a tool response wrapped as a user message). Thinking blocks (`<think>...</think>`) are only emitted for assistant messages that appear **after** `lastQueryIndex`. For assistant messages at or before `lastQueryIndex` (i.e., from previous rounds of the conversation), thinking traces are **completely stripped** — only the visible content is rendered. This prevents the model from seeing its own prior thinking, which the Qwen team determined produces better multi-turn performance.

**What the official template does**: The template walks messages in reverse, looking for user messages whose rendered content (after `|trim`) does NOT start with `<tool_response>` and end with `</tool_response>`. Tool-response user messages are skipped. The first non-tool-response user message found is the `lastQueryIndex`. If no such message exists, the template raises an exception: `raise_exception('No user query found in messages.')`.

**What the upstream `Qwen35Renderer` does**: Lines 114-127 faithfully replicate this logic, including `strings.TrimSpace(content)` before the prefix/suffix check.

**What the fork does**: The fork's `Qwen3CoderRenderer` (lines 149-155) simply finds the last message with `Role == "user"`, regardless of content. In Ollama's protocol, tool responses typically use `role: "tool"` rather than `role: "user"`, so the divergence is partially mitigated. However, the fork's approach is less robust and does not match the template.

**What breaks if you don't fix this**: In a multi-step agentic workflow with the Qwen 3.5 27B model — where the conversation goes user → assistant(tool_call) → tool(result) → assistant(response) → user → assistant(tool_call) → tool(result) → assistant(response) — the wrong `lastQueryIndex` could cause thinking blocks from the current round to be stripped (too aggressive) or thinking from a prior round to be included (too permissive). Both degrade multi-turn reasoning quality.

---

### 6.7 Dedicated Qwen35Parser with Two-Layer Architecture (from Upstream)

**Who got it right**: The upstream Ollama developers, who created `Qwen35Parser` in `model/parsers/qwen35.go`. llama.cpp (`d969e933`) handles parsing via `chat-parser-xml-toolcall.cpp`, which detects `<tool_call>` XML markers and has explicit thinking block extraction logic — the parser is model-agnostic since the template itself determines the output format.
**Who got it wrong**: The fork, which routes `qwen3.5` through `Qwen3CoderParser` with thinking flags bolted on (at `parsers.go:52-53`).

**What the `Qwen35Parser` does**: The upstream `Qwen35Parser` (239 lines) uses a clean two-layer architecture:

1. **Outer layer** (the `Qwen35Parser` itself): A 3-state machine (`CollectingThinking`, `ThinkingDoneEatingWhitespace`, `CollectingContent`) that handles the `<think>...</think>` block extraction. It strips an optional leading `<think>` tag (some model checkpoints emit one even when the prompt already opens thinking), collects thinking content until `</think>`, eats inter-block whitespace, then transitions to content mode.

2. **Inner layer** (an embedded `Qwen3CoderParser`): Initialized at `Qwen35Parser.Init()` line 48-49 with `nil` for both `lastMessage` and `thinkValue`, which means the inner parser starts in `LookingForToolStart` state with `hasThinkingSupport = false`. The inner parser handles ONLY XML tool call parsing (`<tool_call>...<function=...>...</function>...</tool_call>`), with zero awareness of thinking blocks.

The `Qwen35Parser.Init()` also checks for assistant prefill at line 56: `assistantPrefill := lastMessage != nil && lastMessage.Role == "assistant" && lastMessage.Content != ""`. When the last message is a prefilled assistant message, thinking mode is skipped entirely — the parser starts in `CollectingContent` state instead of `CollectingThinking`.

**What the fork does instead**: The fork's `Qwen3CoderParser` (in `parsers/qwen3coder.go`) has 4 states in a single state machine: `LookingForToolStart`, `CollectingToolContent`, `CollectingThinking`, `ThinkingDoneTransition`. Thinking and tool call parsing are interleaved in the same `eat()` function. The parser has `hasThinkingSupport` and `defaultThinking` boolean fields, and routes `qwen3.5` with both set to `true`. There is no assistant prefill check.

**What breaks if you don't fix this**: The interleaved state machine is harder to reason about and maintain. More concretely, the missing assistant prefill check means that when a client sends a prefilled assistant message (an incomplete assistant message for the model to continue), the fork's parser starts in `CollectingThinking` state and attempts to parse the model's output as thinking content — even though the model is continuing a non-thinking response. This can cause the parser to misclassify visible content as thinking, or to hold output in a buffer waiting for a `</think>` tag that never arrives.

---

### 6.8 Repetition Penalty Sampler: Ring Buffer Recording Only Generated Tokens (from Fork)

**Who got it right**: The fork author, who redesigned the `Sampler` struct in `sample/samplers.go` to use a self-contained ring buffer that records only tokens produced by `Sample()`.
**Who got it wrong**: Both upstream Ollama and llama.cpp (`d969e933`). Upstream Ollama feeds both prompt and generated tokens via externally-called `Accept()`. llama.cpp does the same: `server-context.cpp:208-212` calls `common_sampler_accept()` on ALL prompt tokens during `init_sampler()`, feeding them into the penalty ring buffer at `llama-sampler.cpp:2644-2656`. The fork is the only implementation that correctly restricts the penalty window to generated tokens only.

**What the Qwen 3.5 27B model needs for agentic tool calling**: The `Qwen3.5-27B-UD-Q4_K_XL.gguf` model is designed for multi-step agentic workflows where the model calls tools (functions) by name and receives structured results. In a typical agentic conversation:

```
System: You have access to get_weather, search_web, run_code...
User: What's the weather in Tokyo?
Assistant: <tool_call><function=get_weather><parameter=location>Tokyo</parameter></function></tool_call>
Tool: {"temperature": 22, "condition": "sunny", "humidity": 65}
Assistant: The weather in Tokyo is 22°C and sunny with 65% humidity.
```

The model MUST reproduce exact tokens from the prompt — tool names (`get_weather`), parameter names (`location`, `temperature`), and structured JSON keys/values — in its generated output. A repetition penalty that penalizes the model for generating tokens that appeared in the prompt directly undermines this.

**How upstream's sampler works**: The upstream `Sampler` struct (`sample/samplers.go:21-32`) has a `history []int32` field — a simple growing slice. External code in `runner/ollamarunner/runner.go` calls the public `Accept(token int32)` method at three separate sites:

1. **Sequence load** (runner.go lines 951-957): When a new completion request arrives and a KV cache slot is loaded, `seq.sampler.Reset()` clears the history, then ALL cached input tokens are replayed via `seq.sampler.Accept(inp.Token)`. For a conversation with 2000 cached tokens, all 2000 get `Accept()`-ed, then the history is trimmed to the last 64.

2. **Batch processing** (runner.go lines 694-701): As each batch of pending input tokens is committed to the KV cache, every token is `Accept()`-ed. This includes both prompt tokens (from the user's message) and previously generated tokens (which become inputs for the next forward pass).

3. **Reprocess on cache shift** (runner.go line 565): When the KV cache runs out of space and shifts, `seq.sampler.Reset()` wipes the history. The reprocessed tokens will be `Accept()`-ed back through the batch processing path.

The result: the penalty window is a mix of the last ~64 prompt tokens and generated tokens, indiscriminately. The hardcoded `DefaultPenaltyLookback = 64` constant (line 19) controls the window size and cannot be changed via the API — the `repeat_last_n` API parameter is accepted but **silently ignored** (the `NewSampler` function at line 159 has no `repeatLastN` parameter).

With `repeatPenalty > 1.0`, tokens like `"temperature"`, `"get_weather"`, `{`, `}`, `:` from the tool result in the prompt are in the penalty window and actively penalized during generation. The model is discouraged from producing the exact tokens it needs for tool calls and structured output.

**How the fork's sampler works**: The fork's `Sampler` struct (`sample/samplers.go:19-32`) has three fields for penalty tracking: `recentTokens []int32` (the ring buffer), `recentTokenPos int` (write cursor), and `repeatLastN int` (window size from API). The `recordToken(id int32)` method (lines 196-206) is **private** — only called from within `Sample()` itself (at lines 62 and 84), after a token is selected. The runner's only interaction with the sampler is `seq.sampler.Sample(logits)` at line 760. There is no `Accept()` method. There is no `Reset()` method. The runner **cannot** feed prompt tokens into the penalty window because there is no public API to do so.

The ring buffer implements classic O(1) circular buffer semantics:
- While the buffer is still growing (`len(recentTokens) < repeatLastN`): tokens are appended (line 200).
- Once full: new tokens overwrite the oldest entry at `recentTokenPos`, cursor advances via `(recentTokenPos + 1) % repeatLastN` (lines 202-205).

Zero allocations after warmup. Permanently bounded at `repeatLastN` entries. The window size is user-configurable via the `repeat_last_n` API parameter, which flows through `api/types.go` (default 64 at line 1065) → `runner.go` line 899 → `NewSampler` parameter 9 → `Sampler.repeatLastN` field.

**Why upstream Ollama and llama.cpp both default `RepeatPenalty` to 1.0**: With prompt tokens in the penalty window, any `repeatPenalty > 1.0` would degrade tool calling. Setting it to 1.0 (identity) makes the entire penalty math a no-op: `logit / 1.0 = logit`. Both projects were forced into this default by their architectural decision to feed prompt tokens into the penalty window. llama.cpp comments this explicitly: `float penalty_repeat = 1.00f; // 1.0 = disabled` (`common.h:196`). The entire `Accept()`/`Reset()` machinery runs but has **zero effect on output** — it is dead code in both projects.

**Why the fork defaults `RepeatPenalty` to 1.1**: Because the fork's ring buffer only contains generated tokens, a non-trivial penalty is safe. The 1.1 value means: for a token the model already generated recently, positive logits are divided by 1.1 (~9.09% reduction: `1 - 1/1.1 = 1/11`) and negative logits are multiplied by 1.1 (pushed 10% further negative). This actively discourages the Qwen 3.5 27B model from degenerating into repetitive loops — a common failure mode in long-context generation with the model's 262,144-token context window.

**What the ideal implementation also needs from upstream**: The upstream's `tokenCounts()` function (`transforms.go:40`) performs bounds checking on token IDs: `if token < 0 || int(token) >= vocabSize { continue }`. The Qwen 3.5 27B model has a vocabulary of 248,320 tokens (confirmed in the GGUF metadata: `tokenizer.ggml.tokens` = 248,320 entries). The fork's `applyPenalties()` does not validate token IDs — if a corrupted token ID (e.g., `-1` or `300000`) enters the ring buffer, it would create a bogus entry in the penalty `counts` map. The upstream's bounds check should be ported. Additionally, the upstream's `repeatPenalty <= 0` guard (`samplers.go:186-188`: `if repeatPenalty <= 0 { repeatPenalty = 1.0 }`) prevents division by zero when a user passes `repeat_penalty: 0` via the API.

**What breaks if you don't fix this**: Using upstream's sampler architecture with any `repeatPenalty > 1.0` on the Qwen 3.5 27B model in an agentic tool-calling scenario causes the model to avoid reproducing tokens from the tool schema and tool results. Tool call accuracy degrades. The model may paraphrase function names (e.g., generating `weather_check` instead of `get_weather`), skip required parameters, or produce malformed JSON in tool call arguments. Using upstream's 1.0 default avoids this but makes repetition penalty completely non-functional — the model can degenerate into repetitive loops during long generations with no mechanism to prevent it.

---

### 6.9 mropeInterleaved Default: Architecture-Based (from Fork)

**Who got it right**: llama.cpp (`d969e933`) hardcodes this per architecture: `case LLM_ARCH_QWEN35: return LLAMA_ROPE_TYPE_IMROPE` at `llama-model.cpp:9109-9113` — no GGUF metadata lookup, no default mismatch possible. The fork author achieves the same result via `model.go:548`: `c.Architecture() != "qwen3next"` evaluates to `true` for qwen35.
**Who got it wrong**: The upstream Ollama developers, at `model.go:641`: `mropeInterleaved: c.Bool("rope.mrope_interleaved", c.Bool("mrope_interleaved", false))`.

**What the `Qwen3.5-27B-UD-Q4_K_XL.gguf` file contains — and does not contain**: The GGUF metadata was inspected in full (49 metadata keys, 851 tensors). The file has `general.architecture = "qwen35"` and `qwen35.rope.dimension_sections = [11, 11, 10, 0]` (the M-RoPE section sizes for time, height, width, and extra dimensions). It does **NOT** contain any of these keys:

- `rope.mrope_interleaved` — **absent**
- `mrope_interleaved` — **absent**
- `qwen35.rope.mrope_interleaved` — **absent**

These keys are **ollama-internal conventions** that do not exist in llama.cpp's GGUF metadata vocabulary (`gguf-py/gguf/constants.py`). The Unsloth converter (and llama.cpp's `convert_hf_to_gguf.py`, and bartowski's quantizations) never write them. llama.cpp handles the equivalent configuration by hardcoding the RoPE type per architecture enum in C++ — `LLM_ARCH_QWEN35` maps to `LLAMA_ROPE_TYPE_IMROPE` (value 40).

**What happens when the key is absent**: Ollama's `model.go` reads the key with a fallback chain: `c.Bool("rope.mrope_interleaved", c.Bool("mrope_interleaved", <default>))`. Both lookups fail (the keys don't exist in the GGUF), so the **default value** is used.

- **Upstream default**: `false`. This selects `rope.WithMRoPE()` at `model.go:70`, which sets the RoPE type to `1 << 3 = 8 = GGML_ROPE_TYPE_MROPE` (defined in `ggml.h:247`).
- **Fork default**: `c.Architecture() != "qwen3next"`. For the `qwen35` architecture, this evaluates to `true`. This selects `rope.WithInterleaveMRoPE()` at `model.go:69`, which sets the RoPE type to `1<<3 | 1<<5 = 40 = GGML_ROPE_TYPE_IMROPE` (defined in `ggml.h:249`).

**What MROPE (type 8) vs IMROPE (type 40) means for the Qwen 3.5 27B model**: The model has 16 full-attention layers (layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63) with 24 query heads and 4 key-value heads per layer. RoPE (Rotary Position Embedding) is applied to the query and key tensors in each full-attention layer. The model uses M-RoPE (Multi-dimensional RoPE) with sections `[11, 11, 10]`, meaning the 32 active dimension pairs (from `rope.dimension_count = 64`, so 32 pairs) are assigned to three position axes: time (11 pairs), height (11 pairs), and width (10 pairs).

The difference between MROPE and IMROPE is **how** these 32 dimension pairs are assigned to the three axes:

- **IMROPE (type 40, correct for Qwen 3.5)**: Dimension pairs cycle through axes in an **interleaved** pattern: T, H, W, T, H, W, T, H, W, ... Each adjacent dimension pair rotates by a different axis, creating a rich multi-frequency positional signal within each attention head.

- **MROPE (type 8, wrong for Qwen 3.5)**: Dimension pairs are assigned to axes in **contiguous blocks**: the first 11 pairs all rotate by time, the next 11 by height, the next 10 by width. The axes are grouped rather than interleaved.

These produce fundamentally different rotation matrices in the GGML RoPE CUDA kernel (at `ggml-cuda/rope.cu:202-224`). The Qwen 3.5 27B model's attention weight matrices were trained with the interleaved pattern — they learned to extract positional information from specific dimension pair locations. Applying the contiguous pattern instead means:
- Dimension pair #1, which the model expects to encode height position (interleaved: pair 1 → height), instead encodes time position (contiguous: pair 1 → time, since pairs 0-10 are all time).
- Dimension pair #2, which the model expects to encode width, instead also encodes time.
- The Q*K^T dot products embed wrong positional relationships. Tokens at different positions may appear to be at the same position, or vice versa.

For **text-only inference** (which is the case for the `Qwen3.5-27B-UD-Q4_K_XL.gguf` since it has no vision tensors — confirmed: zero vision/visual/vit/image tensors in the 851-tensor file), all three position IDs (time, height, width) are typically set to the same value (the sequential token position). In this case, the distinction between MROPE and IMROPE may seem moot — but the model's learned attention weights are still adapted to the specific frequency landscape of the interleaved pattern. The per-dimension frequencies differ between interleaved and contiguous assignments, producing different attention patterns even with identical position IDs.

**What breaks if you don't fix this**: When loading the `Qwen3.5-27B-UD-Q4_K_XL.gguf` file with upstream Ollama, the `mropeInterleaved` default is `false` because the GGUF does not contain the key. All 16 full-attention layers in the model use RoPE type 8 (MROPE) instead of 40 (IMROPE). The positional encoding is wrong in every full-attention layer. The model generates incoherent, degraded output. This is a **silent** failure — no error message, no warning. The model loads, runs, and produces text, but the text quality is significantly degraded because the attention mechanism's positional signal is scrambled. Users would assume the model is simply bad, when in fact the inference engine is applying incorrect position embeddings.

---

### 6.10 V-Head Reordering Default: Architecture-Based (Both Correct)

**Who got it right**: All three. The fork uses `c.Architecture() != "qwen3next"` (line 539); upstream uses `defaultVHeadReordered(c.Architecture())` which returns `arch == "qwen35" || arch == "qwen35moe"` (lines 493-495, 632). Both evaluate to `true` for `qwen35` and `false` for `qwen3next`. llama.cpp (`d969e933`) sidesteps the runtime flag entirely: the converter physically reorders V-head weights during GGUF conversion via `_LinearAttentionVReorderBase._reorder_v_heads()` (`convert_hf_to_gguf.py:4782-4791`). The permuted weights are stored in the GGUF, so no runtime reordering flag is needed.

**What the `Qwen3.5-27B-UD-Q4_K_XL.gguf` file requires**: The GGUF does NOT contain `ssm.v_head_reordered` (confirmed absent from the 49 metadata keys). The architecture is `qwen35`. Both codebases default to `true`, which is correct.

**What `vHeadReordered` controls**: In each of the 48 GatedDeltaNet recurrent layers, the model has `ssm.group_count = 16` key-heads and `ssm.time_step_rank = 48` value-heads — a 3:1 ratio. Each key-head must be repeated to match 3 value-heads. The `vHeadReordered` flag controls which of two repeat strategies is used at `deltanet.go:190-206`:

- **`true` (Path A, correct for Qwen 3.5)**: `Repeat4D` — simple broadcast repeat. K-head j maps to V-heads {3j, 3j+1, 3j+2}. The V-head weights in the GGUF are stored with heads that share a K-group already contiguous.
- **`false` (Path B, correct for legacy qwen3next)**: Reshape-repeat-reshape — interleaved repeat. K-head j maps to V-heads {j, j+16, j+32}. The V-head weights follow a different storage convention.

Getting this wrong silently produces garbage output in all 48 recurrent layers — the Q/K/V alignment is completely scrambled.

---

### 6.11 Recurrent Layer Classification: Interval Formula (from Fork)

**Who got it right**: All three use the same formula. llama.cpp (`d969e933`) reads `full_attention_interval` from GGUF metadata (default 4) and computes `((i + 1) % full_attn_interval != 0)` at `llama-model.cpp:2526`, storing results in `hparams.recurrent_layer_arr[]`.
**Fork**: 4-line inline computation at `model.go:472-476` using `(i+1) % interval != 0` — identical to llama.cpp.
**Upstream**: 52-line `inferRecurrentLayers()` function at `model.go:497-549` with a primary path through `headCountKV` and a fallback to the interval formula.

**What the `Qwen3.5-27B-UD-Q4_K_XL.gguf` contains**: `qwen35.attention.head_count_kv = 4` — a **scalar** UINT32 value, not a per-layer array. And `qwen35.full_attention_interval = 4`.

**Why the fork's approach is more direct**: When `head_count_kv` is a scalar (as in all third-party GGUFs from Unsloth, bartowski, and the llama.cpp converter), Ollama's GGUF reader broadcasts it to a uniform array of 64 identical values `[4, 4, 4, ..., 4]`. The upstream's `inferRecurrentLayers()` function first checks this array for a mix of zero and non-zero values — it finds all non-zero (all 4), so `hasZero = false` and `hasFull = true`. It falls through to the compatibility path (line 533), which uses the same `(i+1) % interval != 0` formula as the fork.

The fork skips the `headCountKV` indirection entirely and goes straight to the interval formula. For the `Qwen3.5-27B-UD-Q4_K_XL.gguf`, both produce identical results: layers 3, 7, 11, ..., 63 are full-attention; all others are recurrent. The formula matches llama.cpp's `llama-model.cpp:2291`: `hparams.recurrent_layer_arr[i] = ((i + 1) % 4 != 0)`.

---

### 6.12 Unconditional GGUF KV Emission in Converter (from Fork)

**Who got it right**: The fork, which always writes both `rope.mrope_interleaved` and `ssm.v_head_reordered` to the GGUF regardless of their value. This is defense-in-depth: it makes the GGUF self-describing.
**Who got it wrong**: Upstream, which only writes these keys when `true`, meaning they are absent for architectures where they are `false`.
**How llama.cpp avoids the problem entirely**: llama.cpp (`d969e933`) does NOT write `rope.mrope_interleaved` or `ssm.v_head_reordered` to GGUFs at all — these are ollama-internal conventions. Instead, llama.cpp hardcodes RoPE type per architecture in C++ and physically reorders V-heads during conversion. This means the GGUF doesn't need to carry these flags, and there's no default-mismatch risk.

**Why this matters for the `Qwen3.5-27B-UD-Q4_K_XL.gguf`**: This GGUF was NOT produced by Ollama's converter — it was produced by Unsloth's quantization pipeline. So neither fork's nor upstream's converter behavior directly affects this file. However, the fork's unconditional emission matters for GGUFs that ARE produced by Ollama's own `ollama create` command. If a user creates an Ollama GGUF from Qwen 3.5 SafeTensors, the fork's converter writes `rope.mrope_interleaved = true` explicitly, making the GGUF self-describing. Upstream's converter also writes `true` for qwen35 (since the condition `if q.RopeParameters.MRopeInterleaved` is true), so for this specific architecture, both produce the same output. The fork's defense-in-depth matters more for hypothetical architectures where the value is `false` and needs to be explicitly recorded.

---

### 6.13 Model Validation and Nil Checks (from Upstream)

**Who got it right**: The upstream Ollama developers, who added `Validate()` (`model.go:440-478`) and inline nil checks (`deltanet.go:138-149`). llama.cpp (`d969e933`) achieves the same protection differently: `create_tensor()` at `llama-model.cpp:6779` uses flag `0` (REQUIRED) for all recurrent layer tensors — if any tensor is missing, model loading fails immediately with a clear error. No runtime nil checks needed.
**Who got it wrong**: The fork, which has neither `Validate()` nor the additional nil checks.

**What the `Qwen3.5-27B-UD-Q4_K_XL.gguf` contains that matters**: Each of the 48 GatedDeltaNet recurrent layers has these tensors (confirmed by inspecting the GGUF):

| Tensor | Present? | Example Shape |
|--------|----------|---------------|
| `blk.{i}.attn_qkv.weight` (maps to `SSMQKV`) | Yes | [5120, 10240] |
| `blk.{i}.attn_gate.weight` (maps to `SSMQKVGate`) | Yes | [5120, 6144] |
| `blk.{i}.ssm_a` (maps to `SSMA`) | Yes | [48] |
| `blk.{i}.ssm_alpha.weight` (maps to `SSMAlpha`) | Yes | [5120, 48] |
| `blk.{i}.ssm_beta.weight` (maps to `SSMBeta`) | Yes | [5120, 48] |
| `blk.{i}.ssm_conv1d.weight` (maps to `SSMConv1D.Weight`) | Yes | [4, 10240] |
| `blk.{i}.ssm_dt.bias` (maps to `SSMDT` — note: `.bias` not `.weight`) | Yes | [48] |
| `blk.{i}.ssm_norm.weight` (maps to `SSMNorm`) | Yes | [128] |
| `blk.{i}.ssm_out.weight` (maps to `SSMOut`) | Yes | [6144, 5120] |

All tensors are present in this particular GGUF. But a different third-party GGUF (e.g., one produced by an older version of the llama.cpp converter, or a partially-exported model) might be missing one. The fork checks only `SSMQKV` and `SSMQKVGate` (lines 103-105 of `deltanet.go`). If `SSMA`, `SSMDT`, `SSMConv1D`, `SSMNorm`, or `SSMOut` are missing, the fork panics with a nil-pointer dereference at runtime — e.g., `alpha.Add(ctx, gdn.SSMDT)` at line 140 when `gdn.SSMDT` is nil.

Upstream's `Validate()` method checks all 8 tensor groups at model load time, producing clean error messages like `"qwen3next: layer 3 missing ssm_dt tensor"` instead of a cryptic nil-pointer stack trace.

Notable: the `Qwen3.5-27B-UD-Q4_K_XL.gguf` uses `ssm_dt.bias` (not `ssm_dt.weight`) for the dt tensor in all 48 recurrent layers. This was a compatibility fix contributed by the fork in a prior commit. Both codebases now handle this tensor name.

---

### 6.14 Image/Vision Support (from Upstream)

**Who got it right**: The upstream Ollama developers, whose `Qwen35Renderer` includes `renderContent()` (lines 47-62 of `qwen35.go`) for image rendering. llama.cpp (`d969e933`) handles vision via its `mtmd.cpp` multimodal tool, which inserts `<|vision_start|>` / `<|vision_end|>` tokens (at `mtmd.cpp:289-290`). The Jinja2 template itself renders image content via `render_content()` macros.
**Who got it wrong**: The fork, whose `Qwen3CoderRenderer` has no image handling.

**What the `Qwen3.5-27B-UD-Q4_K_XL.gguf` supports**: This specific GGUF is **text-only** — it contains 851 tensors, none of which are vision-related (no `visual.*`, `vit.*`, or `image_*` tensors). The vision encoder is distributed separately as `mmproj-BF16.gguf` (931 MB), `mmproj-F16.gguf` (928 MB), or `mmproj-F32.gguf` (1.84 GB) in the same Unsloth HuggingFace repository. However, the Qwen 3.5 27B model is **architecturally multimodal** — the model's code in upstream's `model.go` conditionally initializes vision support (lines 663-668), configures vision tokens (image_token_id=151655, vision_start=151652, vision_end=151653 at lines 695-700), and the scheduler at `sched.go:448` restricts `qwen35` to `numParallel = 1` (the multimodal pattern).

When a user loads the `Qwen3.5-27B-UD-Q4_K_XL.gguf` AND a separate `mmproj` file together, the model gains vision capabilities. The renderer must insert `<|vision_start|><|image_pad|><|vision_end|>` tokens for each image in the prompt. The fork's `Qwen3CoderRenderer` silently drops all image content — users who pass images get no error and no visual understanding.

For text-only use (the common case with this specific GGUF), the missing image support has no impact. But it prevents users from upgrading to the full multimodal experience when the mmproj file is available.

---

### 6.15 Tokenizer Performance Optimizations (from Fork)

**Who got it right**: The fork author, who contributed three tokenizer performance optimizations that upstream does not have.

These optimizations do not affect correctness — they affect speed. The `Qwen3.5-27B-UD-Q4_K_XL.gguf` has a vocabulary of **248,320 tokens** (confirmed in the GGUF metadata: `tokenizer.ggml.tokens` = 248,320 STRING entries). This is one of the largest vocabularies of any currently-available LLM (for comparison, Llama 3 has 128,256 tokens, GPT-4 has ~100,000). The large vocabulary makes tokenization performance more impactful:

1. **Binary search prompt truncation** (`server/prompt.go`): When a prompt exceeds `num_ctx`, Ollama must find how many messages to drop. The fork uses binary search over the message count (O(log N) tokenize calls), while upstream uses linear search. With a 262,144-token context window and a 248,320-token vocabulary, tokenization is expensive — binary search reduces the number of tokenize calls from O(N) to O(log N) where N is the number of messages.

2. **Special token splitting early-out** (`tokenizer/special.go`): The fork adds a `strings.Contains` check before attempting to split on each special token. With 248,320 vocabulary entries (many of which are special tokens), this avoids running the full split algorithm on tokens that don't appear in the input text.

3. **Stack buffer in BPE merge** (`tokenizer/vocabulary.go`): The fork uses a 128-byte stack buffer for string concatenation in the BPE merge hot path, eliminating heap allocations. The BPE merge loop runs O(V) times where V is the number of merges in the vocabulary — with 248,320 tokens, this is a significant number of iterations.

---

## Priority Action Items for the Fork

| Priority | Item | Effort | Section |
|----------|------|--------|---------|
| **P0** | Replace `SetInplace` with balanced concat tree (workaround for Ollama's stale GGML vendor — llama.cpp master supports `GGML_OP_SET` on Metal/Vulkan) | Small | 1.1 |
| **P0** | Use JSON tool definitions for qwen3.5 (keep XML for qwen3-coder) | Medium | 1.2 |
| **P0** | Swap system+tools ordering for qwen3.5 | Small | 1.3 |
| **P1** | Fix `formatToolCallArgument` to use spaced JSON for objects/arrays | Small | 2.2 |
| **P2** | Add `repeatPenalty <= 0` guard | Trivial | 1.4 |
| **P2** | Add nil checks + `Validate()` to qwen3next | Small | 1.5 |
| **P2** | Add image/vision support for qwen3.5 | Medium | 1.6 |
| **P2** | Add token ID bounds checking to penalty sampler | Trivial | 3.3 |
| **P3** | Remove default system message injection for qwen3.5 | Trivial | 1.7 |
| **P3** | Emit `</think>` unconditionally for empty reasoning | Trivial | 1.8 |
| **P3** | Add tool_response check to lastQueryIndex | Small | 1.9 |

## What Upstream Ollama Should Fix (for reference)

| Priority | Item | llama.cpp status | Section |
|----------|------|-----------------|---------|
| **P0** | Fix prefill detection to exclude tool-call messages (ALL 3 renderers) | N/A — llama.cpp uses `add_generation_prompt`, no conflation | 2.1 |
| **P1** | Fix `formatToolCallArgument` to use spaced JSON | Correct via `tojson` filter | 2.2 |
| **P1** | Always close `</think>` in Qwen3VLRenderer | Correct via Jinja2 template | 2.4 |
| **P1** | Actually honor `repeat_last_n` API parameter in Go runner | Correct — `penalty_last_n` wired through | 2.5 |
| **P1** | Fix mropeInterleaved default for third-party qwen35/qwen35moe GGUFs | Correct — hardcoded per arch | 2.6 |
| **P2** | Fix TrimSpace inconsistency in Qwen3VLRenderer lastQueryIndex | Correct via Jinja2 template | 2.7 |
| **P2** | Consider recording only generated tokens in penalty window | **Same bug as llama.cpp** — both feed prompt tokens | 3.1 |
| **P2** | Consider making repeat_penalty default > 1.0 — requires fixing prompt-token inclusion first | **Same 1.0 default as llama.cpp** | 2.8, 3.1 |
| **P2** | Emit `ssm.v_head_reordered` / `rope.mrope_interleaved` unconditionally in converter | N/A — llama.cpp doesn't use these keys | 3.4 |
