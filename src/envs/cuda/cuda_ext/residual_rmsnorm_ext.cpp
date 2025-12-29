#include <torch/extension.h>

torch::Tensor residual_rmsnorm_forward(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &residual_rmsnorm_forward, "Residual + RMSNorm forward (CUDA)");
}

