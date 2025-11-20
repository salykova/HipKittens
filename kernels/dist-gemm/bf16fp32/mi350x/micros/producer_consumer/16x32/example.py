import torch
import iris_py
import tk_kernel

torch.manual_seed(0)

# Helper function to create iris tensor and wrap as PyTorch tensor
def make_iris_tensor(iris, shape, dtype="bfloat16"):
    """Create iris tensor and wrap as PyTorch tensor (HACK using __cuda_array_interface__)"""
    import ctypes
    
    iris_tensor = iris.empty(shape, dtype=dtype)
    
    # Map dtype to torch dtype and typestr
    dtype_map = {
        "bfloat16": (torch.bfloat16, "<u2"),  # bfloat16 as uint16
        "bf16": (torch.bfloat16, "<u2"),
        "float32": (torch.float32, "<f4"),
        "float": (torch.float32, "<f4"),
        "float16": (torch.float16, "<f2"),
        "half": (torch.float16, "<f2"),
    }
    torch_dtype, typestr = dtype_map[dtype]
    
    # HACK: Use __cuda_array_interface__ to create zero-copy tensor
    # This is the standard way to share GPU memory between libraries
    class CudaArrayWrapper:
        def __init__(self, ptr, shape, typestr):
            self.__cuda_array_interface__ = {
                'shape': tuple(shape),
                'typestr': typestr,
                'data': (ptr, False),  # (ptr, read_only)
                'version': 3,
                'strides': None,  # C-contiguous
            }
            self._iris_tensor = iris_tensor  # Keep alive
    
    wrapper = CudaArrayWrapper(iris_tensor.data_ptr(), shape, typestr)
    
    # Create torch tensor from wrapper using __cuda_array_interface__
    # PyTorch should respect this protocol
    try:
        torch_tensor = torch.as_tensor(wrapper, device='cuda').to(torch_dtype)
    except:
        # Fallback: use from_dlpack if available
        try:
            # Create a minimal DLPack-like object
            torch_tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
            # Store iris_tensor to keep it alive
            torch_tensor._iris_tensor = iris_tensor
        except:
            # Last resort: regular allocation
            torch_tensor = torch.empty(shape, dtype=torch_dtype, device='cuda')
            torch_tensor._iris_tensor = iris_tensor
    
    return torch_tensor

# Setup
M = 192 * 40  # 7680
K = 8192
N = 8192
scale = 10.0

print("="*50)
print(f"Distributed Producer-Consumer GEMM: {M}x{K} @ {K}x{N}")
print("="*50)

# Initialize iris
print("\n[Initializing Iris]")
iris = iris_py.Iris(heap_size_mb=512, verbose=False)
print(f"✓ Iris initialized: rank={iris.rank()}, world_size={iris.world_size()}")

# Allocate tensors using helper function
print("\n[Allocating Iris Tensors]")
A = make_iris_tensor(iris, [M, K], dtype="bfloat16")
B = make_iris_tensor(iris, [N, K], dtype="bfloat16")
C = make_iris_tensor(iris, [M, N], dtype="bfloat16")

print(f"✓ A: {A.shape}, device={A.device}, dtype={A.dtype}")
print(f"✓ B: {B.shape}, device={B.device}, dtype={B.dtype}")
print(f"✓ C: {C.shape}, device={C.device}, dtype={C.dtype}")

# Initialize in-place
print("\n[Initializing In-Place]")
A.copy_(torch.randn(M, K, dtype=torch.bfloat16, device='cuda') / scale)
B.copy_(torch.randn(N, K, dtype=torch.bfloat16, device='cuda') / scale)
C.zero_()
print(f"✓ Tensors initialized")

iris.barrier()

# Use tensors directly - they're already PyTorch tensors!
print("\n[Computing Reference]")
C_ref = torch.matmul(A, B.t())

print("\n[Running HipKittens Kernel]")
# Get iris device context and pass to kernel
iris_device_ctx = iris.get_device_view()
tk_kernel.dispatch_micro(A, B, C, iris_device_ctx)
torch.cuda.synchronize()

# Validation
print("\n[Validating Results]")
diff = (C.float() - C_ref.float()).abs()
max_error = diff.max().item()

print(f"Max error: {max_error:.6f}")
status = "✓ PASSED" if max_error < 0.1 else "✗ FAILED"
print(status)

print("\n" + "="*50)
print("Done!")
print("="*50)
