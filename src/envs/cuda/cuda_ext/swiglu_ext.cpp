#include <torch/extension.h>

torch::Tensor swiglu_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &swiglu_forward, "SwiGLU forward (CUDA)");
}

