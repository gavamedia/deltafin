import json, urllib.request, numpy as np, sys
sys.path.insert(0, 'tools')
from mxfp4 import dequant_mxfp4, _E2M1
import model_source
INV = json.load(open('k3-meta/tensor_inventory_offsets.json'))
BASE = model_source.base_url()
def fetch(name):
    t = INV[name]
    start = 8 + t['hlen'] + t['offsets'][0]; size = t['offsets'][1]-t['offsets'][0]
    req = urllib.request.Request(BASE+t['shard'], headers={'Range': f'bytes={start}-{start+size-1}'})
    with urllib.request.urlopen(req, timeout=120) as r: buf = r.read()
    return np.frombuffer(buf, dtype=np.uint8).reshape(t['shape'])

pfx = 'language_model.model.layers.1.block_sparse_moe.experts.0.w1'
packed = fetch(pfx+'.weight_packed'); scale = fetch(pfx+'.weight_scale')
print('packed', packed.shape, 'scale', scale.shape)
w = dequant_mxfp4(packed, scale)
print('dequant shape', w.shape, 'dtype', w.dtype)
print(f'stats: mean {w.mean():.6f} std {w.std():.6f} min {w.min():.4f} max {w.max():.4f}')
print(f'zeros: {(w==0).mean()*100:.2f}%  |  scale byte range: {scale.min()}-{scale.max()} (2^{scale.min()-127}..2^{scale.max()-127})')
# cross-check e2m1 table against ml_dtypes
import ml_dtypes
ref = np.arange(16, dtype=np.uint8).view(ml_dtypes.float4_e2m1fn).astype(np.float32)
ours = _E2M1
print('e2m1 table matches ml_dtypes:', np.array_equal(ref, ours), '| table:', ours.tolist())
# nibble-order check: reconstruct histogram sanity — a trained weight matrix should be roughly symmetric
print('sym check |mean|/std:', abs(w.mean())/w.std())
