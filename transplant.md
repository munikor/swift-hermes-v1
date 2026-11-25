# Transplant stage (3): bake Swift's head/MTP into the quant output

After `quant_swift_hermes.py` saves `swift-hermes-gptq-int4/`, the lm_head and
MTP tensors are **dense BF16** (excluded from quantization). Swift's calibrated
GPTQ bake must replace them. The merge stage also silently dropped the MTP
block entirely (transformers does not register `mtp.*` on Qwen3_5), so MTP is
*added*, not replaced.

```python
import json, os, shutil, safetensors.torch as st

SRC = 'Swift-1.5-baked-v1/'    # Wrapzii checkpoint (read-only)
DST = 'swift-hermes-gptq-int4/'  # quant output

bake = st.load_file(SRC + 'model-bake-int4.safetensors')   # 36 GPTQ tensors: lm_head + mtp.*

didx = json.load(open(DST + 'model.safetensors.index.json'))
wm = didx['weight_map']

# 1. swap dense lm_head.weight for the bake's GPTQ quartet (+ mtp.* additions)
target = wm['lm_head.weight']
d = st.load_file(DST + target)
del d['lm_head.weight']
d.update(bake)
st.save_file(d, DST + target + '.new', metadata={'format': 'pt'})
os.replace(DST + target + '.new', DST + target)
for k in bake: wm[k] = target
wm.pop('lm_head.weight', None)
json.dump(didx, open(DST + 'model.safetensors.index.json', 'w'))

# 2. INT8 embedding side file + provenance files
for f in ['model-embed-int8.safetensors', 'MODIFICATIONS.json', 'NOTICE',
          'BAKE.json', 'SOURCE.json', 'COMPATIBILITY.json', 'chat_template_low.jinja']:
    if os.path.exists(SRC + f): shutil.copy(SRC + f, DST + f)

# 3. restore source quantize_config.json (lm_head=true, mtp dynamic negatives),
#    keeping the new build's meta block (calibration provenance)
qc = json.load(open(SRC + 'quantize_config.json'))
dst_qc = json.load(open(DST + 'quantize_config.json'))
qc['meta'] = dst_qc.get('meta', {})
json.dump(qc, open(DST + 'quantize_config.json', 'w'), indent=2)

# 4. verify: every weight_map entry physically present, per shard
for f in set(wm.values()):
    with st.safe_open(DST + f, framework='pt') as fh: keys = set(fh.keys())
    assert {k for k, v in wm.items() if v == f} <= keys, f
print('transplant OK,', len(wm), 'tensors')
```

Result: 2,419 tensors (2,384 dense/quant body + 36 bake − 1 dense lm_head),
`lm_head.qweight` mapped, index integrity verified.
