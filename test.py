import torch
import triton

n = 2048

a = torch.randn((n, n), device="cuda", dtype=torch.float16)
b = torch.randn((n, n), device="cuda", dtype=torch.float16)


ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
perf = lambda ms: 2 * n * n * n * 1e-12 / (ms * 1e-3)

print(ms)
print(perf(ms))
