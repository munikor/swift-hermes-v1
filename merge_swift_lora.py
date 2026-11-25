import os, time, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from peft import PeftModel
from gptqmodel.utils.model_dequant import dequantize_model

BASE='/home/adminmike/swift-hermes-build/Swift-1.5-baked-v1'
BF16='/home/adminmike/swift-hermes-build/swift-bf16'
ADAPTER='/home/adminmike/hermes-adapter'
OUT='/home/adminmike/swift-hermes-build/merged-bf16'

t0=time.time()
dequantize_model(BASE, BF16, target_dtype=torch.bfloat16)
print(f'dequantized in {time.time()-t0:.0f}s', flush=True)

tok = AutoTokenizer.from_pretrained(BF16, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(BF16, dtype=torch.bfloat16, device_map='cpu',
                                                    low_cpu_mem_usage=True, trust_remote_code=True)
print(f'loaded in {time.time()-t0:.0f}s', flush=True)
pm = PeftModel.from_pretrained(model, ADAPTER)
model = pm.merge_and_unload()
print(f'merged in {time.time()-t0:.0f}s', flush=True)
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True, max_shard_size='5GB')
tok.save_pretrained(OUT)
print('saved', OUT, flush=True)
print('DONE-MERGE', flush=True)
