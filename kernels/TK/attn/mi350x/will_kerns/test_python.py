import torch
import tk_kernel
import tk_golden
import random
import time
import math
from torch.nn.functional import scaled_dot_product_attention
from aiter.ops.triton.mha import flash_attn_func

profiling = True
using_aiter = True

torch.manual_seed(0)
random.seed(0)


torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

# Inputs
B = 16
H = 16
N = 4096
D = 64
causal = False
dtype = torch.bfloat16
q = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)

out = torch.zeros(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
out_ref_pytorch = scaled_dot_product_attention(q, k, v, is_causal=causal)

tk_kernel.dispatch_micro(q, k, v, out)

print("out")
print(out[0, 0, 0:16, 0:16])
print("out_ref_pytorch")
print(out_ref_pytorch[0, 0, 0:16, 0:16])

diff = out - out_ref_pytorch
print("diff")
print(diff[0, 0, 0:16, 0:16])

max_error = diff.max().item()
print("max_error")
print(max_error)



    