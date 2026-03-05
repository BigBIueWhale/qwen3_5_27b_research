# Qwen 3.5 27B — Ollama Bug Research & Fix

Source-level analysis of critical bugs in Ollama's Qwen 3.5 27B integration (v0.17.1 through v0.17.4), and a working fork that fixes all four plus three tokenizer performance bugs from a [separate investigation](https://github.com/BigBIueWhale/ollama_perf_bug_report).

## Current Reports

The `current_reports/` folder contains the active research deliverables and a resource index:

- [**Fork vs Upstream Analysis**](current_reports/fork_vs_upstream_analysis.md) — Fork vs upstream comparison pinned to upstream `82848a78`. Covers what the fork should fix (P0-P3), what upstream should fix, and critical architectural differences (penalty sampling, ring buffer, KV emission).
- [**Fork vs Latest Upstream & llama.cpp**](current_reports/fork_vs_latest_upstream_and_llama_cpp.md) — Comparison against latest upstream Ollama (`9896e36`) and llama.cpp (`a0ed91a`). Covers CUDA async copy, M-RoPE can_shift guard, speculative decoding impossibility, parallelism restrictions, vision multimodal robustness, sampler architecture, thinking/non-thinking mode correctness, and all 5 critical fork fixes still unfixed upstream.
- [**Local Resources**](current_reports/local_resources.md) — Index of all local clones, source files, research documents, and remote references.

## Other Documents

- [**Solution**](solution.md) — Working fork with all fixes, change table, API examples, and recommended parameter profiles. Start here.
- [**Inference Report**](qwen3.5_27b_inference_report.md) — Full source-level analysis of four bugs: wrong tool calling format (with missing thinking support in the correct pipeline), silently ignored repetition penalties, unclosed `</think>` tags in multi-turn prompts, and missing generation prompt after tool call turns.
- [**Third-Party GGUF Compatibility**](third_party_gguf_compatibility.md) — Three ollama-internal GGUF metadata keys (`ssm.v_head_reordered`, `rope.mrope_interleaved`, `isRecurrent` computation) that broke inference on Unsloth/bartowski/llama.cpp-generated GGUFs.
- [**Ollama Issue**](ollama_issue.md) — GitHub issue text filed against ollama/ollama ([#14493](https://github.com/ollama/ollama/issues/14493)).
- [**PR #14167 Comment**](tommy_pr_comment.md) — Comment on [TommyBoiss's unmerged PR](https://github.com/ollama/ollama/pull/14167).
- [**Example: Math Olympiad Proof**](example_math_olympiad_unsloth_ud.md) — Qwen 3.5 27B (Unsloth UD-Q4_K_XL) produces a rigorous, self-contained proof of non-existence for a constrained decagon problem.
