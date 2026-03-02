# Third-Party GGUF Compatibility: Three Ollama-Internal Metadata Defaults

**Updated:** March 2, 2026 UTC
**Status:** All three bugs fixed in [`BigBIueWhale/ollama@9ec17fc`](https://github.com/BigBIueWhale/ollama/commit/9ec17fc)

## Summary

Third-party GGUFs (Unsloth, bartowski, anything from llama.cpp's `convert_hf_to_gguf.py`) are missing three ollama-internal GGUF metadata keys. Ollama defaulted all three incorrectly, causing garbage output, wrong positional encodings, or load failures when using these GGUFs. llama.cpp doesn't read any of these keys — it infers the correct behavior from the architecture name.

| Key | What it controls | Ollama default (broken) | Ollama default (fixed) | llama.cpp equivalent |
|---|---|---|---|---|
| `ssm.v_head_reordered` | Q/K repeat strategy in GDN | `false` | `c.Architecture() != "qwen3next"` | Hardcoded per-architecture: Path A in `qwen35.cpp`, Path B in `qwen3next.cpp` |
| `rope.mrope_interleaved` | Interleaved vs standard MRoPE | `false` | `c.Architecture() != "qwen3next"` | Hardcoded `LLAMA_ROPE_TYPE_IMROPE` for qwen35 in `llama_model_rope_type()` |
| *(implicit)* `full_attention_interval` | Recurrent vs full-attention layer classification | Derived from `head_count_kv` array heuristic | Direct `(i+1) % interval != 0` | Direct computation in `llama-model.cpp:2497-2527` |

---

## Bug 1: `ssm.v_head_reordered` — Garbage Output from Unsloth GGUFs

### Problem

The Unsloth GGUF for Qwen 3.5 27B loads successfully but produces garbage output — minimal thinking, empty content, immediate EOS. The ollama registry GGUF for the same model produces correct output.

| GGUF source | `ssm.v_head_reordered` value | Output quality |
|---|---|---|
| Ollama registry | `true` (explicitly set) | Correct |
| Unsloth (HuggingFace) | **Missing** → defaulted to `false` | Garbage |

### Root Cause

Qwen 3.5's GDN layers have asymmetric head counts: 16 K-heads vs 48 V-heads. Q and K must be repeated (broadcast) to match V's head count. **How** they're repeated determines whether Q/K/V alignment is correct.

Ollama has two code paths in [`deltanet.go:191-206`](https://github.com/BigBIueWhale/ollama/blob/main/model/models/qwen3next/deltanet.go#L191):

- **Path A** (`vHeadReordered=true`): Direct `Repeat4D` — tile semantics. K-head `j` maps to V-heads `{3j, 3j+1, 3j+2}` (contiguous groups).
- **Path B** (`vHeadReordered=false`): Reshape-repeat-reshape — interleave semantics. K-head `j` maps to V-heads `{j, j+16, j+32}` (strided pattern).

### Why Two Paths Exist (Both Are Correct — For Different Architectures)

llama.cpp has **two separate model implementations** with different Q/K repeat strategies:

| llama.cpp file | Architecture | Q/K repeat strategy | Converter reorders V weights? |
|---|---|---|---|
| [`qwen35.cpp:324-329`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp) | `qwen35`, `qwen35moe` | Direct `ggml_repeat_4d` (**Path A**) | **Yes** — `_LinearAttentionVReorderBase` in `convert_hf_to_gguf.py:4768-4842` |
| [`qwen3next.cpp:408-427`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen3next.cpp) | `qwen3next` | Reshape-repeat-reshape (**Path B**) | **No** — `Qwen3NextModel` base class skips reorder |

The HuggingFace reference (`modeling_qwen3_5.py:589`) uses `repeat_interleave`, which is Path B's semantics. llama.cpp's converter compensates by physically reordering V-head weights from grouped to tiled layout for Qwen 3.5, so that `ggml_repeat_4d` (tile semantics, Path A) produces the correct alignment. For Qwen3Next, no reorder happens and the reshape-repeat-reshape (Path B) matches the original weight layout directly.

Ollama unifies both architectures into one model with `vHeadReordered` as the switch. The default must match the architecture: `true` for qwen35/qwen35moe (reordered weights, Path A), `false` for qwen3next (non-reordered weights, Path B).

### Fix

```go
// model/models/qwen3next/model.go
vHeadReordered: c.Bool("ssm.v_head_reordered", c.Architecture() != "qwen3next"),
```

Explicit GGUF values still override the default. The `ssm.v_head_reordered` key does not exist in llama.cpp's GGUF spec — it is ollama-internal.

---

## Bug 2: `rope.mrope_interleaved` — Wrong Positional Encoding

### Problem

Qwen 3.5 uses Interleaved Multi-scale RoPE (IMRoPE) for its 16 full-attention layers. The HuggingFace reference calls `apply_interleaved_mrope()` unconditionally (`modeling_qwen3_5.py:240`). Qwen3Next uses standard NeoX RoPE with no MRoPE sections.

| GGUF source | `rope.mrope_interleaved` value | RoPE behavior |
|---|---|---|
| Ollama registry | `true` (explicitly set) | Correct (IMRoPE) |
| Unsloth (HuggingFace) | **Missing** → defaulted to `false` | **Wrong** (non-interleaved MRoPE) |

### Root Cause

`rope.mrope_interleaved` is an ollama-internal convention. It does not exist in llama.cpp's GGUF constants (`gguf-py/gguf/constants.py`). llama.cpp's converter does not write it. llama.cpp hardcodes `LLAMA_ROPE_TYPE_IMROPE` for the `qwen35` architecture in `llama_model_rope_type()` at [`llama-model.cpp:9111-9113`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp):

```cpp
case LLM_ARCH_QWEN35:
case LLM_ARCH_QWEN35MOE:
    return LLAMA_ROPE_TYPE_IMROPE;
```

With the `false` default, ollama's RoPE code at [`model.go:67-72`](https://github.com/BigBIueWhale/ollama/blob/main/model/models/qwen3next/model.go#L67) selects `rope.WithMRoPE()` instead of `rope.WithInterleaveMRoPE()`, applying position encodings in the wrong dimensional order across all 16 full-attention layers.

For Qwen3Next, `mropeSections` is empty (no MRoPE), so the flag is irrelevant — the code falls through to `rope.WithTypeNeoX()`.

### Fix

```go
// model/models/qwen3next/model.go
mropeInterleaved: c.Bool("rope.mrope_interleaved", c.Bool("mrope_interleaved", c.Architecture() != "qwen3next")),
```

---

## Bug 3: `isRecurrent` — Fragile Layer Type Detection

### Problem

The previous fix ([`bebddccd`](https://github.com/BigBIueWhale/ollama/commit/bebddccd)) used a 30-line `allSame` heuristic on `head_count_kv` to reconstruct per-layer types. Third-party GGUFs store `head_count_kv` as a scalar (broadcast to all layers by `fs/ggml/ggml.go:HeadCountKV`), losing the per-layer signal.

### Root Cause

llama.cpp never uses `head_count_kv` for layer type detection. It computes recurrent layers directly from `full_attention_interval` (default 4) via `(i+1) % interval != 0` at [`llama-model.cpp:2497-2527`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp):

```cpp
for (uint32_t il = 0; il < n_layer; ++il) {
    if (recurrent_layer_arr.empty()) {
        hparams.recurrent_layer_arr.push_back(
            (full_attention_interval > 0)
                ? ((int(il) + 1) % full_attention_interval != 0)
                : false);
    }
}
```

### Fix

Replace the 30-line heuristic with 4 lines matching llama.cpp:

```go
// model/models/qwen3next/model.go
interval := int(c.Uint("full_attention_interval", 4))
isRecurrent = make([]bool, numLayers)
for i := range numLayers {
    isRecurrent[i] = (i+1)%interval != 0
}
```

---

## Converter: Always Write Keys Explicitly

The ollama converter (`convert/convert_qwen3next.go`) previously only wrote `ssm.v_head_reordered` and `rope.mrope_interleaved` when `true`, omitting them for qwen3next. Now both are always written explicitly so the GGUF is self-describing regardless of loading-code defaults:

```go
kv["ssm.v_head_reordered"] = q.shouldReorderVHeads()
kv["rope.mrope_interleaved"] = q.RopeParameters.MRopeInterleaved
```

This is defense-in-depth — the `model.go` defaults are the primary fix.

---

## Verification

All three architectures work correctly with both ollama registry and third-party GGUFs:

| Architecture | Ollama registry GGUF | Unsloth/third-party GGUF |
|---|---|---|
| `qwen35` | Reads explicit `true` for both keys | Defaults to `true` via architecture check |
| `qwen35moe` | Reads explicit `true` for both keys | Defaults to `true` via architecture check |
| `qwen3next` | Reads explicit `false` for both keys | Defaults to `false` via architecture check |

---

## DeltaNet Numerics: Confirmed Correct

Cross-referencing the ollama Go implementation against llama.cpp's C++ and HuggingFace's Python reference confirms the DeltaNet/GDN computation is numerically sound and functionally equivalent across all three:

| Component | Ollama (Go) | llama.cpp (C++) | HuggingFace (Python) |
|---|---|---|---|
| L2 norm on Q, K | `q.L2Norm(ctx, eps)` | `ggml_l2_norm(ctx0, q_conv, eps_norm)` | `l2norm(query, dim=-1, eps=1e-6)` |
| Scale factor | `1/sqrt(headVDim)` | `1.0f/sqrtf(S_k)` | implicit in kernel |
| Gate computation | `softplus(alpha + dt_bias) * -A` | same | `F.softplus(alpha + dt_bias) * -A.exp()` |
| Gated norm | `RMSNorm(out) * SiLU(z)` | `build_norm(out) * ggml_silu(z)` | `self.norm(out, z)` |
| Norm weight +1 convention | `addOne` excludes `ssm_norm` | `data_torch + 1` excludes `linear_attn.norm` | `Qwen3_5RMSNorm` uses `(1 + weight)`, `Qwen3_5RMSNormGated` uses `weight` directly |
| Chunk size | 64 | 64 | 64 |
| Autoregressive threshold | `nSeqTokens == 1` | `n_seq_tokens == 1` | N/A (uses fused kernels) |

## Sources

- llama.cpp qwen35 forward pass: [`src/models/qwen35.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)
- llama.cpp qwen3next forward pass: [`src/models/qwen3next.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen3next.cpp)
- llama.cpp V-head reorder in converter: [`convert_hf_to_gguf.py:4768-4842`](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) (`_LinearAttentionVReorderBase`)
- llama.cpp rope type hardcoding: [`llama-model.cpp:9111-9113`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)
- llama.cpp recurrent layer computation: [`llama-model.cpp:2497-2527`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp)
- HuggingFace Q/K repeat_interleave: [`modeling_qwen3_5.py:589`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- HuggingFace IMRoPE: [`modeling_qwen3_5.py:240`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
- `ggml_repeat_4d` tile semantics: [`ggml-cpu/ops.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ops.cpp)
- Ollama DeltaNet forward pass: [`model/models/qwen3next/deltanet.go:191-206`](https://github.com/BigBIueWhale/ollama/blob/main/model/models/qwen3next/deltanet.go#L191)
- Ollama RoPE selection: [`model/models/qwen3next/model.go:67-72`](https://github.com/BigBIueWhale/ollama/blob/main/model/models/qwen3next/model.go#L67)
