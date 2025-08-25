import torch
import random
import math
import tk_kernel

torch.set_printoptions(
    precision=3,        
    sci_mode=False,     
    linewidth=220,      
    threshold=float("inf")  
)

random.seed(0)
torch.manual_seed(0)

b = 1
h = 1
n = 512
d = 32

# pytorch
x = torch.randn((b, h, n, d), dtype=torch.float32, device='cuda')
vec = torch.randn((b, h, 1, d), dtype=torch.float32, device='cuda')
# y = x - vec
y = x.sum(dim=-1, keepdim=True)

# tk
y_tk = torch.zeros_like(y)
tk_kernel.dispatch_micro(x, vec, y_tk)

# check
diff = (y - y_tk).abs()
max_diff = diff.max()
print(y.shape, vec.shape, x.shape)
print(f"diff: {max_diff}")


print(y[0, 0, 0:2, :8])
print(y_tk[0, 0, 0:2, :8])
print()

print(y[0, 0, 8:10, :8])
print(y_tk[0, 0, 8:10, :8])


