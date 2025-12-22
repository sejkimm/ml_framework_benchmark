# ML Framework Benchmark

- Created: 2025.12.22

A benchmarking tool to compare performance of different ML frameworks and kernels.

## Comparison Targets

- Torch (cuBLAS)
- Triton
- Custom CUDA fused ops

## Operations

- Matrix multiplication (matmul)
- Residual connection + RMSNorm
- SwiGLU activation

## Environment

Tested on NVIDIA A6000 with `nvcr.io/nvidia/pytorch:25.11-py3` container

## Usage

Run the benchmark:

```bash
python3 main.py
```
