import torch
import sys

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt['model'] if 'model' in ckpt else ckpt
for k in sorted(sd.keys()):
    if 'head' in k or 'query' in k or 'norm' in k or 'attn' in k:
        print(k)
