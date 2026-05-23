#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <tuple>

std::tuple<torch::stable::Tensor, torch::stable::Tensor> ctc_alpha_cuda(const torch::stable::Tensor& log_probs,
                                                                        const torch::stable::Tensor& targets,
                                                                        const torch::stable::Tensor& in_lens,
                                                                        const torch::stable::Tensor& tgt_lens);

torch::stable::Tensor ctc_grad_cuda(const torch::stable::Tensor& alpha, const torch::stable::Tensor& log_Z,
                                    const torch::stable::Tensor& log_probs, const torch::stable::Tensor& targets,
                                    const torch::stable::Tensor& in_lens, const torch::stable::Tensor& tgt_lens,
                                    const torch::stable::Tensor& grad_loss, bool zero_infinity);

STABLE_TORCH_LIBRARY(asr, m) {
    m.def("ctc_alpha(Tensor log_probs, Tensor targets, Tensor in_lens, Tensor tgt_lens) -> (Tensor, Tensor)");

    m.def(
        "ctc_grad(Tensor alpha, Tensor log_Z, Tensor log_probs, Tensor targets, Tensor in_lens, Tensor tgt_lens, "
        "Tensor grad_loss, bool zero_infinity) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(asr, CUDA, m) {
    m.impl("ctc_alpha", TORCH_BOX(&ctc_alpha_cuda));
    m.impl("ctc_grad", TORCH_BOX(&ctc_grad_cuda));
}