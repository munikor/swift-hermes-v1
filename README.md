# Swift-1.5-Qwen3.8-27B-hermes-v1 — build recipe

**Weights live on Hugging Face:**
[muni501/Swift-1.5-Qwen3.8-27B-hermes-v1-GPTQ-Int4](https://huggingface.co/muni501/Swift-1.5-Qwen3.8-27B-hermes-v1-GPTQ-Int4)

This repository contains only the **build recipe** — the exact scripts that
produced the checkpoint — plus the license/provenance files mirrored from the
HF repo. No weights are stored here.

## What the model is

Swift 1.5 (Qwen3.8-27B, UkisAI) with the
[harness-hub hermes-adapted LoRA](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted)
merged into the body, re-quantized to GPTQ INT4 (group 128, symmetric), while
**preserving byte-identical** Swift's calibrated GPTQ bake of the lm_head and
MTP linears and the INT8 embedding side file. Built for TensorFold XPU serving
on Intel Arc Pro B70.

The full model card (derivation chain, quality caveats, license terms,
serving command) is the README on the HF repo — kept identical there and in
`MODEL_CARD.md` in this repo.

## Pipeline (3 stages)

```
Swift baked GPTQ INT4 ──dequant──> BF16 ──merge LoRA──> merged BF16 ──body quant──> swift-hermes-v1
        (Wrapzii)                  (peft)                (gptqmodel)
```

| Stage | Script | What it does |
|---|---|---|
| 1 | `merge_swift_lora.py` | `gptqmodel.utils.model_dequant.dequantize_model` → BF16; PEFT `merge_and_unload` of the hermes adapter; save |
| 2 | `quant_swift_hermes.py` | gptqmodel quantize, bits=4 g128 sym, **dynamic excludes** for `visual`, `in_proj_a/b`, `embed_tokens`, `mtp`, `lm_head` so the baked head survives dense |
| 3 | (in `transplant.md`) | Replace dense `lm_head` + re-add `mtp.*` from `model-bake-int4.safetensors` byte-identical; copy INT8 embedding side file; rewrite index; restore source `quantize_config.json` (lm_head=true) |

`quant_log.csv` in the HF repo carries the per-layer quantization log.

## Gotchas we hit (for the next person)

- **PEFT cannot merge into gptqmodel qlinear modules** (`NotImplementedError`)
  and refuses quantized models outright — dequant to BF16 **first**, then merge.
- **`transformers` silently drops the MTP block** when loading/saving a
  Qwen3_5 checkpoint: the model class doesn't register the `mtp.*` submodule.
  The merged BF16 dir had zero MTP tensors; they must be re-added in the
  transplant stage from the original bake.
- gptqmodel's `dequantize_model` requires all four GPTQ buffers
  (`qweight/qzeros/scales/g_idx`) **co-located in the same shard**. Swift's
  `compat_g_idx.safetensors` side file breaks format detection (falls back to
  AWQ math and explodes) — merge the g_idx tensors into their home shards
  before dequantizing, and translate the `dynamic` negative patterns into
  `ignored_layers` in `quantization_config.json`.
- gptqmodel 7.3.2 on CUDA needs `ninja` for Marlin JIT; either install it or
  force `backend=BACKEND.GPTQ_TORCH`.
- The vision preprocessor configs must be present in the merged dir or
  `AutoTokenizer`/processor loading fails at quant time.

## Reproduce

```bash
# env: python 3.12, torch 2.15+cu, gptqmodel 7.3.2, peft, transformers 5.24, CUDA GPU
python merge_swift_lora.py     # ~5 min on a GB10 + 121GB unified memory
python quant_swift_hermes.py   # ~29 min on CUDA
# transplant stage per transplant.md, then verify index integrity
```

Inputs: `Swift-1.5-baked-v1/` (Wrapzii checkpoint) and `hermes-adapter/`
(harness-hub LoRA) alongside the scripts.

## License

Same dual license as the weights — see `LICENSE` (Swift Open License v1.0,
<$1M commercial threshold), `LICENSE-APACHE-2.0` (Qwen base, hermes adapter),
and `NOTICE`. Not Apache-only; the Swift contribution's commercial condition
travels with any derivative.
