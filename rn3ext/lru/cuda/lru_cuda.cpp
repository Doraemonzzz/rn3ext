#include <torch/extension.h>

std::vector<torch::Tensor> lru_forward_cuda(
    torch::Tensor& x_real,
    torch::Tensor& x_imag,
    torch::Tensor& lambda_real,
    torch::Tensor& lambda_imag
);

std::vector<torch::Tensor> lru_backward_cuda(
    torch::Tensor& x_real,
    torch::Tensor& x_imag,
    torch::Tensor& lambda_real,
    torch::Tensor& lambda_imag,
    torch::Tensor& hidden_states_real,
    torch::Tensor& hidden_states_imag,
    torch::Tensor& grad_output_real,
    torch::Tensor& grad_output_imag
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lru_forward_cuda, "lru forward (CUDA)");
  m.def("backward", &lru_backward_cuda, "lru backward (CUDA)");
}

TORCH_LIBRARY(lru_cuda, m) {
    m.def("forward", lru_forward_cuda);
    m.def("backward", lru_backward_cuda);
}