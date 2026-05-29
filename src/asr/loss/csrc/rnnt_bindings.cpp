// Generated with Claude Code (https://claude.com/claude-code). Opus 4.8 xhigh.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <tuple>

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor,
           torch::stable::Tensor>
rnnt_forward_cuda(const torch::stable::Tensor& encoder, const torch::stable::Tensor& decoder,
                  const torch::stable::Tensor& joint_W, const torch::stable::Tensor& targets,
                  const torch::stable::Tensor& in_lens, const torch::stable::Tensor& tgt_lens, int64_t blank_idx,
                  int64_t tf32);

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor> rnnt_backward_cuda(
    const torch::stable::Tensor& encoder, const torch::stable::Tensor& decoder, const torch::stable::Tensor& joint_W,
    const torch::stable::Tensor& targets, const torch::stable::Tensor& in_lens, const torch::stable::Tensor& tgt_lens,
    int64_t blank_idx, const torch::stable::Tensor& alpha, const torch::stable::Tensor& log_P_blank,
    const torch::stable::Tensor& log_P_y, const torch::stable::Tensor& log_prob, const torch::stable::Tensor& grad_loss,
    int64_t tf32);

// `asr` is already claimed by ctc_bindings.cpp's STABLE_TORCH_LIBRARY, so extend it via FRAGMENT.
STABLE_TORCH_LIBRARY_FRAGMENT(asr, m) {
    m.def(
        "rnnt_forward(Tensor encoder, Tensor decoder, Tensor joint_W, Tensor targets, Tensor in_lens, "
        "Tensor tgt_lens, int blank_idx, int tf32) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "rnnt_backward(Tensor encoder, Tensor decoder, Tensor joint_W, Tensor targets, Tensor in_lens, "
        "Tensor tgt_lens, int blank_idx, Tensor alpha, Tensor log_P_blank, Tensor log_P_y, Tensor log_prob, "
        "Tensor grad_loss, int tf32) -> (Tensor, Tensor, Tensor)");
}

STABLE_TORCH_LIBRARY_IMPL(asr, CUDA, m) {
    m.impl("rnnt_forward", TORCH_BOX(&rnnt_forward_cuda));
    m.impl("rnnt_backward", TORCH_BOX(&rnnt_backward_cuda));
}
