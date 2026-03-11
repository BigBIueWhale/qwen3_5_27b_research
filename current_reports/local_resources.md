# Local Resources for Qwen 3.5 Ollama Fork Development

## Fork Repository

| Path | Description |
|------|-------------|
| `/home/user/shared_vm/ollama/` | **BigBIueWhale/ollama** fork @ `4044b63f` — 11 commits atop `v0.17.4` merge (tag `v0.17.4-bbl.5`). Merge base: `cc90a035`. Remote `origin` = `https://github.com/BigBIueWhale/ollama.git`, remote `upstream` = `https://github.com/ollama/ollama.git` |

### Key fork source files

| Path | What it does |
|------|-------------|
| `model/models/qwen3next/deltanet.go` | GatedDeltaNet recurrent layer — contains `SetInplace` bug (lines 445-484) |
| `model/models/qwen3next/model.go` | Qwen3Next model definition — recurrent layer inference, defaults, missing `Validate()` |
| `model/renderers/qwen35.go` | Dedicated `Qwen35Renderer` — JSON tool defs, tools-first ordering, image support, `lastQueryIndex`, unconditional `<think>` wrapping in history (matches official Jinja2 template), prefill bug fix |
| `model/renderers/qwen3coder.go` | Shared renderer for qwen3-coder — tool definitions (XML), thinking, lastQueryIndex, Qwen-local structured tool argument serializer with spaced JSON and no HTML escaping |
| `model/renderers/qwen3vl.go` | Qwen3VL renderer — prefill fix, `</think>` closure fix |
| `model/parsers/qwen35.go` | Dedicated `Qwen35Parser` — thinking extraction, delegates tool calls to embedded `Qwen3CoderParser` |
| `model/parsers/qwen3coder.go` | Shared parser for qwen3-coder — thinking support, XML tool call parsing |
| `model/parsers/parsers.go` | Parser routing — maps `"qwen3.5"` to `Qwen35Parser`, `"qwen3-coder"` to `Qwen3CoderParser` |
| `model/renderers/renderer.go` | Renderer routing — maps `"qwen3.5"` to `Qwen35Renderer`, `"qwen3-coder"` to `Qwen3CoderRenderer` |
| `sample/samplers.go` | Ring buffer penalty sampler — `recordToken()`, `repeatLastN` parameter. Grammar sampler with lazy activation support (`NewGrammarSamplerLazy`), dead-end detection (all-tokens-rejected error) |
| `sample/transforms.go` | Penalty application — `applyPenalties()` |
| `runner/ollamarunner/runner.go` | Runner — removed `Accept()`/`Reset()`, passes `RepeatLastN` |
| `convert/convert_qwen3next.go` | GGUF converter — unconditional KV emission |
| `server/prompt.go` | Binary search prompt truncation (perf) |
| `tokenizer/special.go` | Special token splitting with `strings.Contains` early-out (perf) |
| `tokenizer/vocabulary.go` | Stack buffer in `Merge()` BPE hot path (perf) |
| `api/types.go` | `RepeatPenalty: 1.1` default (line 1066) |
| `llama/sampling_ext.h` | C bridge header — `grammar_init_lazy()` (dormant grammar with regex triggers), `tool_call_grammar_from_json()` + 11 error codes |
| `llama/sampling_ext.cpp` | C bridge impl — `grammar_init_lazy()` (~35 lines), `tool_call_grammar_from_json()` (~230 lines): trie-based exclusion patterns, GBNF grammar builder for Qwen 3.5 XML tool calls. Null-properties fix for nil `*ToolPropertiesMap` |
| `llama/llama.go` | Go wrapper — `NewGrammarLazy()` (lazy grammar creation), `ToolCallGrammarFromJSON()` (grammar from tools JSON), consolidated `newGrammar()` shared helper |
| `llm/server.go` | `CompletionRequest` with `Grammar` + `GrammarTriggerPatterns` fields, `ToolCallGrammarFromJSON()` passthrough, Format→Grammar guard (`if req.Grammar == ""`) |
| `server/routes.go` | `ChatHandler` builds tool grammar for `qwen3.5` parser with mode-dependent trigger patterns (thinking: require `</think>` before `<tool_call>`; non-thinking: trigger on first `<tool_call>`) |
| `runner/ollamarunner/runner.go` | Lazy vs eager grammar branch based on `GrammarTriggerPatterns` |

---

## Upstream Clone

| Path | Description |
|------|-------------|
| `/tmp/ollama-master/` | **ollama/ollama** master @ `82848a78` (`model: fix renderer and parser for qwen3.5 (#14605)`). Full clone with history. |

### Key upstream-only files (not in fork)

| Path | What it does |
|------|-------------|
| `/tmp/ollama-master/model/renderers/qwen35.go` | Dedicated `Qwen35Renderer` — JSON tool definitions, tools-first ordering, image support, lastQueryIndex, correct `</think>` closure. Has prefill bug at line 136. |
| `/tmp/ollama-master/model/parsers/qwen35.go` | Dedicated `Qwen35Parser` — thinking extraction, delegates tool calls to embedded `Qwen3CoderParser` |
| `/tmp/ollama-master/model/renderers/json.go` | `marshalWithSpaces` function — JSON with spaces after `:` and `,`. Has HTML-escaping bug (`<>&` → `\u003c` etc.) |

---

## llama.cpp Clone

| Path | Description |
|------|-------------|
| `/tmp/llama-cpp-master/` | `ggml-org/llama.cpp` master @ `d969e933` (`d969e933e172821b4519f66aa4b660bc0846b320`). Shallow clone (`--depth=1`). Used for comparing Ollama's Qwen 3.5 implementation against the canonical C++ inference engine. Key files: `src/models/delta-net-base.cpp` (GatedDeltaNet with `ggml_set_inplace`), `src/llama-model.cpp` (arch-hardcoded IMROPE at line 9109), `src/llama-sampler.cpp` (ring buffer penalty sampler), `common/chat.cpp` (Jinja2 template execution), `convert_hf_to_gguf.py` (V-head reordering in converter), `tools/server/server-context.cpp` (prompt tokens fed into sampler at line 212). |

---

## Latest Upstream Clone

| Path | Description |
|------|-------------|
| `/tmp/ollama-latest/` | **ollama/ollama** master @ `9896e36` (4 commits newer than `82848a78`). Full clone with history. |

## Latest llama.cpp Clone

| Path | Description |
|------|-------------|
| `/tmp/llama-cpp-latest/` | `ggml-org/llama.cpp` master @ `a0ed91a`. Full clone. Contains CUDA async copy optimization (`2cd20b7`), M-RoPE `can_shift()` guard (`99bd67c`), KDA chunk size 16 vs GDA chunk size 64, and speculative decoding compatibility checks. |

## Newest llama.cpp Clone

| Path | Description |
|------|-------------|
| `/tmp/llama_cpp_up_to_date/` | `ggml-org/llama.cpp` master @ `c5a7788` ("ggml: add GATED_DELTA_NET op"). Shallow clone (`--depth=1`). Trie upgraded to `uint32_t` Unicode codepoints in `peg-parser.cpp`. Tool grammar builder refactored from `chat.cpp` into new `chat-peg-parser.cpp` with parameterized markers. Public API (`json-schema-to-grammar.h`) byte-for-byte identical to vendored version. |

## Stale Upstream Bare Clone

| Path | Description |
|------|-------------|
| `/tmp/ollama-upstream/` | Bare clone of `ollama/ollama` @ `8207e55e` (older than `/tmp/ollama-master/`). Created earlier in the session. **Stale — use `/tmp/ollama-latest/` instead.** |

---

## Research Documents

### Current reports (this folder)

| File | Lines | Description |
|------|-------|-------------|
| `fork_vs_upstream_analysis.md` | 348 | Fork vs upstream analysis pinned to upstream `82848a78`. Covers what the fork should fix (P0-P3), what upstream should fix, and critical architectural differences (penalty sampling, ring buffer, KV emission). |
| `fork_vs_latest_upstream_and_llama_cpp.md` | 724 | Fork vs latest upstream Ollama (`9896e36`) and llama.cpp (`a0ed91a`). Covers CUDA async copy, M-RoPE can_shift, speculative decoding, parallelism, vision, sampler, thinking/non-thinking modes, and correctness. |
| `grammar_constrained_tool_calls_plan.md` | ~530 | Implementation plan for grammar-constrained tool call generation — **ALL 5 STEPS DONE** (commit `4044b63f`). Covers why (Ollama trusts model, llama.cpp doesn't), full infrastructure audit, constraint comparison table (llama.cpp vs fork), all error handling paths, step-by-step implementation with actual code details, mode-dependent trigger patterns, dead-end detection, renderer fix. 8 files changed, ~212 lines added. Pinned to llama.cpp reference `a0ed91a`. |

### Other research documents (parent directory)

| Path | Lines | Description |
|------|-------|-------------|
| `../qwen3.5_27b_inference_report.md` | 1668 | Inference report for Qwen 3.5 27B — model behavior, performance observations |
| `../third_party_gguf_compatibility.md` | 182 | Third-party GGUF compatibility notes — Unsloth, llama.cpp converter quirks |
| `../ollama_issue.md` | 105 | Draft GitHub issue content |
| `../tommy_pr_comment.md` | 14 | PR comment draft |
| `../solution.md` | 386 | Solution notes |
| `../example_math_olympiad_unsloth_ud.md` | 1409 | Example inference output — math olympiad problem with Unsloth UD GGUF |

---

## Modelfile

| Path | Description |
|------|-------------|
| `/home/user/Desktop/vibe_web_terminal/ollama-models/qwen3.5-custom.Modelfile` | Custom Modelfile for Qwen 3.5 27B. Uses Unsloth Dynamic 2.0 `UD-Q4_K_XL` GGUF (17.6 GB, 851 tensors, text-only). Pinned to HuggingFace commit `d0cb6e8` (March 2, 2026 iteration 2 — fixes MXFP4 bug). Source: `unsloth/Qwen3.5-27B-GGUF`. Parameters: `num_ctx 131072`, `num_predict 81920`, `temperature 1.0`, `top_k 20`, `top_p 0.95`, `repeat_penalty 1.0`, `presence_penalty 1.5`. Uses `RENDERER qwen3.5` / `PARSER qwen3.5`. |

### GGUF download command (from Modelfile)
```bash
wget -O /tmp/Qwen3.5-27B-UD-Q4_K_XL.gguf \
  https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/d0cb6e84962d23e5b86e65f7c21e003faf0f0d68/Qwen3.5-27B-UD-Q4_K_XL.gguf
```

---

## Python Environment

| Path | Description |
|------|-------------|
| `/tmp/qwen35-check/` | Minimal uv project (`Python >=3.12`) with `transformers>=5.2.0` and `jinja2>=3.1.6`. Created for fetching/validating Qwen 3.5 chat templates from HuggingFace. |
| `/tmp/olmo3-check/` | Scratch directory used to verify OLMo tool serialization against official HuggingFace sources. Contains `allenai/Olmo-3-7B-Instruct` `chat_template.jinja`, `config.json`, tokenizer config, model API metadata, and `allenai/Olmo-3.1-32B-Instruct` `chat_template.jinja`. Verification result: the shared tool-definition HTML-escaping fix is **correct in principle** for all current `marshalWithSpaces` callers (`qwen3.5`, `qwen3-vl`, `olmo3`) because the official templates use `tools | tojson` for tool definitions. **Important:** this fix is still NOT implemented in the fork as of 2026-03-11, and failed implementation attempts showed that changing `marshalWithSpaces` alone is insufficient — nested/custom `MarshalJSON` boundaries in `api/types.go` still HTML-escape nested tool-definition fields. |

---

## Official Qwen 3.5 Template (Remote)

| URL | Access | Description |
|-----|--------|-------------|
| `https://huggingface.co/Qwen/Qwen3.5-35B-A3B/raw/main/tokenizer_config.json` | Public (no auth) | Official Jinja2 chat template. Ground truth for JSON tool definitions, tools-first ordering, `</think>` closure, tool call whitespace, and `lastQueryIndex` logic. |
| `https://huggingface.co/Qwen/Qwen3-Coder-8B-Instruct/raw/main/tokenizer_config.json` | Gated (401) | Qwen3-Coder template. Uses XML tool definitions, system-first ordering. Not directly accessible — upstream's `Qwen3CoderRenderer` serves as a reference proxy. |
| `https://huggingface.co/unsloth/Qwen3.5-27B-GGUF` | Public | Unsloth UD GGUF source. Pinned commit: `d0cb6e8`. |
