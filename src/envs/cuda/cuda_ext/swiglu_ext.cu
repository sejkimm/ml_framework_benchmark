#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void swiglu_fwd_kernel(const half* __restrict__ x,
                                 half* __restrict__ y,
                                 int hidden,
                                 int hidden2,
                                 int64_t total) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (; idx < total; idx += stride) {
    int64_t row = idx / hidden;
    int64_t col = idx - row * hidden;
    int64_t base = row * hidden2 + col;
    float a = __half2float(x[base]);
    float b = __half2float(x[base + hidden]);
    float out = (a * sigmoid(a)) * b;
    y[idx] = __float2half_rn(out);
  }
}

}  // namespace

torch::Tensor swiglu_forward(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(x.size(1) % 2 == 0, "x.size(1) must be even");

  int64_t rows = x.size(0);
  int64_t hidden2 = x.size(1);
  int64_t hidden = hidden2 / 2;
  int64_t total = rows * hidden;

  auto y = torch::empty({rows, hidden}, x.options());
  constexpr int kThreads = 256;
  int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  blocks = blocks > 65535 ? 65535 : blocks;

  swiglu_fwd_kernel<<<blocks, kThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      static_cast<int>(hidden),
      static_cast<int>(hidden2),
      total);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

