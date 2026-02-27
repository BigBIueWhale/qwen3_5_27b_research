# Qwen 3.5 27B — Ollama Bug Research & Fix

Source-level analysis of four critical bugs in Ollama's Qwen 3.5 27B integration (v0.17.1 through v0.17.4), and a working fork that fixes all four plus three tokenizer performance bugs from a [separate investigation](https://github.com/BigBIueWhale/ollama_perf_bug_report).

## Documents

- [**Client Usage Guide**](client_usage.md) — How to use Qwen 3.5 27B for agentic tool calling with thinking via our [Ollama fork](https://github.com/BigBIueWhale/ollama/commit/ab234955). Includes a full change table covering both Qwen 3.5 fixes and performance fixes, working API examples, all four recommended parameter profiles from the model card, default parameter origins, and VRAM considerations for RTX 5090.
- [**Inference Report**](qwen3.5_27b_inference_report.md) — Full source-level analysis of four bugs: wrong tool calling format (with missing thinking support in the correct pipeline), silently ignored repetition penalties, unclosed `</think>` tags in multi-turn prompts, and missing generation prompt after tool call turns.
- [**Ollama Issue**](ollama_issue.md) — GitHub issue text filed against ollama/ollama ([#14493](https://github.com/ollama/ollama/issues/14493)).
- [**PR #14167 Comment**](tommy_pr_comment.md) — Comment on [TommyBoiss's unmerged PR](https://github.com/ollama/ollama/pull/14167), noting that his implementation correctly wired the tool calling format where the official `main` branch got it wrong.
