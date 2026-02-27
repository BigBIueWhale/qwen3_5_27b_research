# Qwen 3.5 27B — Ollama Bug Research

Source-level analysis of critical bugs in Ollama's Qwen 3.5 27B integration (v0.17.1 through master).

## Documents

- [**Inference Report**](qwen3.5_27b_inference_report.md) — Full source-level analysis of three bugs: wrong tool calling format, silently ignored repetition penalties, and unclosed `</think>` tags in multi-turn prompts.
- [**Ollama Issue**](ollama_issue.md) — GitHub issue text filed against ollama/ollama.
- [**PR #14167 Comment**](tommy_pr_comment.md) — Comment on [TommyBoiss's unmerged PR](https://github.com/ollama/ollama/pull/14167), noting that his implementation correctly wired the tool calling format where the official `main` branch got it wrong.
