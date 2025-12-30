# ML Framework Benchmark

- Created: 2025.12.22

## Update

| Date       | Description                                                                      |
|------------|----------------------------------------------------------------------------------|
| 2025.12.29 | Added MacOS Torch MPS / MLX / JAX Metal Comparison                               |
| 2025.12.29 | Added correctness checks and torch.compile optimizations for both CUDA and macOS |
| 2025.12.30 | Added CUDA NVIDIA Nsight Systems tracing                                         |

A benchmarking tool to compare performance of different ML frameworks and kernels.

## Comparison Targets

### CUDA (Linux)

- Torch (cuBLAS)
- Triton
- Custom CUDA fused ops

### macOS

- MLX
- PyTorch MPS
- JAX Metal

## Usage

Run CUDA benchmarks (Linux):

```bash
uv run -m src.envs.cuda.runner
```

Nsight Systems tracing (Linux/CUDA):

```bash
nsys profile --trace=cuda,nvtx --sample=none -o result/cuda_bench -- uv run -m src.envs.cuda.runner --nsys
```

Run macOS benchmarks:

```bash
uv run -m src.envs.macos.runner
```

## Operations

### Matrix multiplication (matmul)

Computes `C = A @ B` where `A[M,K]`, `B[K,N]`, `C[M,N]` are FP16 CUDA tensors.

- `torch`: `a @ b` (cuBLAS/cuBLASLt).
- `triton`: tiled GEMM kernel with FP32 accumulation, stored as FP16.

### Residual connection + RMSNorm

Computes a fused “residual add + RMSNorm”:

- `z = x + residual`
- `y = (z * rsqrt(mean(z^2) + eps)) * weight`

### SwiGLU activation

Computes SwiGLU:

- Split `x[..., 2*hid]` into `a, b`
- `y = silu(a) * b` where `silu(a) = a * sigmoid(a)`

## Benchmark Result

| Date       | Environment                        | Processor        | Doc                                    |
|------------|------------------------------------|------------------|----------------------------------------|
