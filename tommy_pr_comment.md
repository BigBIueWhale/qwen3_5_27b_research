Hey @TommyBoiss — just wanted to give you a heads up since this PR has been sitting here with zero comments for 3 weeks.

Official Qwen 3.5 support landed on `main` while this was open:

- [`da70c32`](https://github.com/ollama/ollama/commit/da70c32) (Feb 24) — "model: support for qwen3.5 architecture (#14378)"
- [`7f9efd5`](https://github.com/ollama/ollama/commit/7f9efd5) (Feb 25) — "model: add support for qwen3.5-27b model (#14415)"

The implementation lives in the `qwen3next` package. It's more refined than your PR in several areas (V-head reordering, separate alpha/beta projections, MoE support, shared cache via `kvcache.Recurrent`, deepstack vision, SetInplace-based DeltaNet chunking). So in terms of the model architecture and inference engine, `main` supersedes this PR.

**However** — I did a detailed source-level comparison of your PR against `main`, and I noticed something important: your PR wires `"qwen3.5"` to `Qwen3CoderRenderer`/`Qwen3CoderParser`, while `main` wires it to `Qwen3VLRenderer`/`Qwen3Parser`. **You got the tool calling format right and `main` got it wrong.** The [HuggingFace ground truth template](https://huggingface.co/Qwen/Qwen3.5-27B/blob/main/tokenizer_config.json) confirms Qwen 3.5 was trained on the Qwen3-Coder XML format (`<function=name>...</function>`), not the Qwen 3 Hermes JSON format (`<tool_call>{"name": ...}</tool_call>`) that `main` currently sends. This means tool calling on `main` is completely non-functional for Qwen 3.5 — the model receives prompts in a format it was never trained on.

I researched the current state of Qwen 3.5 support on Ollama's `main` branch extensively, and as of v0.17.4 (the newest stable release) there are still three critical bugs that make the model's agentic capabilities completely non-functional. I've filed a detailed issue with source-level analysis covering all three: [ollama/ollama#14493](https://github.com/ollama/ollama/issues/14493)

Appreciate you putting in the work on this — you were two weeks ahead of the official implementation and got the tool calling right where they didn't.
