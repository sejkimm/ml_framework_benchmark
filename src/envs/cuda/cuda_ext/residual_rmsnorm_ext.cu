#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

__inline__ __device__ float warp_reduce_sum(float val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int wid = threadIdx.x >> 5;

  val = warp_reduce_sum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  int warps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < warps) ? shared[lane] : 0.0f;
  if (wid == 0) {
    val = warp_reduce_sum(val);
  }
  return val;
}

__global__ void residual_rmsnorm_fwd_kernel(const half* __restrict__ x,
                                           const half* __restrict__ residual,
                                           const half* __restrict__ weight,
                                           half* __restrict__ y,
                                           int N,
                                           float eps) {
  int row = blockIdx.x;
  const half* x_row = x + row * N;
  const half* r_row = residual + row * N;
  half* y_row = y + row * N;

  float sum_sq = 0.0f;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float z = __half2float(x_row[col]) + __half2float(r_row[col]);
    sum_sq += z * z;
  }
  sum_sq = block_reduce_sum(sum_sq);

  __shared__ float inv_rms;
  if (threadIdx.x == 0) {
    float mean_sq = sum_sq / static_cast<float>(N);
    inv_rms = rsqrtf(mean_sq + eps);
  }
  __syncthreads();

  float inv = inv_rms;
  for (int col = threadIdx.x; col < N; col += blockDim.x) {
    float z = __half2float(x_row[col]) + __half2float(r_row[col]);
    float w = __half2float(weight[col]);
    y_row[col] = __float2half_rn(z * inv * w);
  }
}

}  // namespace

torch::Tensor residual_rmsnorm_forward(torch::Tensor x,
                                       torch::Tensor residual,
                                       torch::Tensor weight,
                                       double eps) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(residual.is_cuda(), "residual must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be float16");
  TORCH_CHECK(residual.dtype() == x.dtype(), "residual dtype must match x");
  TORCH_CHECK(weight.dtype() == x.dtype(), "weight dtype must match x");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D");
  TORCH_CHECK(residual.sizes() == x.sizes(), "residual shape must match x");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(weight.size(0) == x.size(1), "weight must have shape (hidden,)");

  int64_t M = x.size(0);
  int64_t N64 = x.size(1);
  TORCH_CHECK(N64 <= 1'000'000, "hidden dim is unexpectedly large");
  int N = static_cast<int>(N64);

  auto y = torch::empty_like(x);
  constexpr int kThreads = 256;
  residual_rmsnorm_fwd_kernel<<<static_cast<unsigned int>(M),
                                kThreads,
                                0,
                                at::cuda::getDefaultCUDAStream()>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<half*>(y.data_ptr<at::Half>()),
      N,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

