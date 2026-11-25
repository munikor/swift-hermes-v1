import json, time, random
from datasets import load_dataset
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig

SRC='/home/adminmike/swift-hermes-build/merged-bf16'
OUT='/home/adminmike/swift-hermes-build/swift-hermes-gptq-int4'

tok=AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)

texts=[]
try:
    ds=load_dataset('wikitext','wikitext-2-raw-v1',split='train')
    texts+=[t for t in ds['text'] if len(t.strip())>400][:192]
    print('wikitext samples:',len(texts),flush=True)
except Exception as e:
    print('wikitext unavailable:',e,flush=True)

agent_prompts=[
 'List the running services on the box and report which LLM port is listening.',
 'Check whether port 8200 answers /health and report the served model id.',
 'Write a systemd user unit that serves a GGUF on port 8093 with 4 slots.',
 'Find the note that explains how the brain was rolled back from 173 to 170.',
 'Summarise the last 20 lines of the engine log and say whether it crashed.',
 'Create a backup tarball of the vault and verify its checksum.',
 'Explain why vLLM refuses to start when KV cache memory is below max_model_len.',
 'Give me the exact command to restart the sparkdash user service.',
]
tool_json=json.dumps({'name':'shell','arguments':{'cmd':'ss -ltn | grep 8888'}})
for p in agent_prompts:
    for _ in range(6):
        texts.append(f'<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{tool_json}\n</tool_call><|im_end|>\n')
        texts.append(f'<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\nI checked the box: port 8888 is listening and the engine reports healthy.<|im_end|>\n')
random.seed(0); random.shuffle(texts)
texts=texts[:256]
print('calibration samples:',len(texts),flush=True)

qc=QuantizeConfig(
    bits=4, group_size=128, sym=True, desc_act=False,
    dynamic={
        '-:.*visual.*': {},
        '-:.*in_proj_a.*': {},
        '-:.*in_proj_b.*': {},
        '-:.*embed_tokens.*': {},
        '-:.*mtp.*': {},
        '-:lm_head': {},
    },
    offload_to_disk=True,
)
t0=time.time()
model=GPTQModel.load(SRC, qc)
print(f'loaded for quant in {time.time()-t0:.0f}s',flush=True)
model.quantize(texts, batch_size=1)
print(f'quantized in {time.time()-t0:.0f}s',flush=True)
model.save(OUT)
print('saved',OUT,flush=True)
print('DONE-QUANT',flush=True)
