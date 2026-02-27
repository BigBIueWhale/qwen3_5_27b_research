# Qwen 3.5 27B — Ollama Bug Research & Fix

Source-level analysis of four critical bugs in Ollama's Qwen 3.5 27B integration (v0.17.1 through v0.17.4), and a working fork that fixes all four plus three tokenizer performance bugs from a [separate investigation](https://github.com/BigBIueWhale/ollama_perf_bug_report).

## Documents

- [**Solution**](solution.md) — Working fork with all fixes, change table, API examples, and recommended parameter profiles. Start here.
- [**Inference Report**](qwen3.5_27b_inference_report.md) — Full source-level analysis of four bugs: wrong tool calling format (with missing thinking support in the correct pipeline), silently ignored repetition penalties, unclosed `</think>` tags in multi-turn prompts, and missing generation prompt after tool call turns.
- [**Ollama Issue**](ollama_issue.md) — GitHub issue text filed against ollama/ollama ([#14493](https://github.com/ollama/ollama/issues/14493)).
- [**PR #14167 Comment**](tommy_pr_comment.md) — Comment on [TommyBoiss's unmerged PR](https://github.com/ollama/ollama/pull/14167), noting that his implementation correctly wired the tool calling format where the official `main` branch got it wrong.
