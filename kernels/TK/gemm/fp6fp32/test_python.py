import torch
import tk_kernel
import random
import time

torch.manual_seed(0)
random.seed(0)

# Inputs
N = 64
inputs = torch.randn(N, N, dtype=torch.float, device='cuda') / 10.0  

C_ref = inputs.clone()
C_ref = torch.ones_like(inputs)

# Kernel matmul
output = torch.zeros(N, N, dtype=torch.float, device='cuda')
tk_kernel.dispatch_micro(inputs, output)


# Compare against reference
C_float = output.float()
C_ref_float = C_ref.float()
diff = (C_float - C_ref_float).abs()
max_error = diff.max().item()
mean_error = diff.mean().item()
error_count = (diff > 0.1).sum().item()
print(f"Max error between kernel and reference: {max_error}")
print(f"Max error: {max_error}")
print(f"Mean error: {mean_error}")
print(f"Number of large errors (>0.1): {error_count}\n")


print("output sample:")
print(output[:32, :32])
print("reference sample:")
print(C_ref[:32, :32])

breakpoint()