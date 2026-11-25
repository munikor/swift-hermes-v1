---
license: other
license_name: swift-open-license-v1.0
license_link: LICENSE
base_model: ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound
tags:
- gptq
- int4
- intel-xpu
- tensorfold
- qwen3_8
- hermes-agent
- mtp
- quantized
library_name: safetensors
language:
- en
---

# Swift-1.5-Qwen3.8-27B-hermes-v1-GPTQ-Int4

A GPTQ INT4 quantization of **Swift 1.5 (Qwen3.8-27B)** with the
[harness-hub/Qwen3.8-27B-hermes-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted)
LoRA merged into the body, built for serving behind
[TensorFold](https://github.com/) on Intel Arc Pro B70 (XPU).

## Derivation chain

| Layer | Source | License |
|---|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Alibaba Cloud) | Apache-2.0 |
| Swift 1.5 post-training | [ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AutoRound) (UkisAI) | Swift Open License v1.0 |
| Head/MTP GPTQ bake + INT8 embedding | [Wrapzii/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8](https://huggingface.co/Wrapzii/Swift-1.5-Qwen3.8-27b-GPTQ-Int4-baked-v1-embed-int8) | SOL v1.0 (derivative) |
| Hermes agent adapter | [harness-hub/Qwen3.8-27B-hermes-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-hermes-adapted) | Apache-2.0 |
| **This repo** | LoRA merge into Swift body + body re-quantization | see below |

## What changed (this repo)

1. **Body**: Swift's GPTQ INT4 body linears were dequantized to BF16, the
   hermes-adapted LoRA was merged (`peft.merge_and_unload`), and the body was
   re-quantized with `gptqmodel 7.3.2` — INT4, group size 128, symmetric, no
   activation ordering. Calibration: 192 wikitext-2 samples + 64 agent/tool-call
   shaped transcripts (256 total, 2048-token concat), on CUDA.
2. **lm_head + MTP (9 linears)**: Swift's calibrated GPTQ bake
   (`model-bake-int4.safetensors`) transplanted **byte-identical** — NOT
   re-quantized. Merge did not touch these (the adapter has no head/MTP
   weights, and the MTP block was re-added from the bake).
3. **Embedding**: dense BF16 in shards, plus Wrapzii's INT8 side file
   (`model-embed-int8.safetensors`) for the patched TP-aware runtime loader.

See `MODIFICATIONS.json`, `BAKE.json`, `SOURCE.json` for machine-readable
provenance, and `NOTICE` for the full license chain.

## Quality caveats (read before using)

- The body re-quantization re-introduces GPTQ error on top of Swift's
  AutoRound body quantization. No perplexity or downstream benchmark has been
  re-established for this checkpoint. The hermes adapter's published
  SWE-bench Pro / Terminal-Bench numbers were measured on the **dense base**
  and do not automatically transfer to this INT4 build.
- Tested only with TensorFold XPU serving on 2× Intel Arc Pro B70. Checkpoint
  compatibility outside this runtime is not established (the INT8 embedding
  side file requires the patched loader).

## Serving (TensorFold XPU, B70)

```bash
docker run -d --name swift-hermes --restart unless-stopped -p 8202:8000 \
  --device /dev/dri -e ZE_AFFINITY_MASK=0 -e XPU_KV_INT8=1 \
  -v /path/to/this/repo:/models/model:ro \
  -v /path/to/DFlash2-drafter:/drafter:ro \
  tensorfold-xpu-b70:hermes-2026.11.24 \
  serve /models/model --host 0.0.0.0 --port 8000 --context 200000 \
  --prompt-cache-gib 1 --drafter /drafter --no-thinking
```

Measured locally (2× B70, one instance per card, 200k context): short-prompt
requests ~0.2–0.3 s end-to-end; long-context behavior in progress.

## License

This model is **not Apache-only**:

- **Swift Contribution** (and this derivative of it): [Swift Open License v1.0](LICENSE) —
  commercial use permitted **only** while gross revenue is below the USD 1M
  Threshold (Section 5); above it, a Swift Enterprise License is required.
- **Qwen base material + hermes adapter**: [Apache-2.0](LICENSE-APACHE-2.0).

The full upstream NOTICE is retained in `NOTICE`. This is not an official
UkisAI, Qwen, or harness-hub release.
