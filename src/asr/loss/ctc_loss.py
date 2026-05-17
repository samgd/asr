import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__forceinline__ __device__ int64_t log_probs_idx(
    int T, int V,
    int b, int t, int v)
{
    return ((int64_t)b*T + t)*V + v;
}

__forceinline__ __device__ int64_t targets_idx(
    int S_max,
    int b, int s_raw)
{
    return (int64_t)b*S_max + s_raw;
}

__forceinline__ __device__ int64_t alpha_idx(
    int T, int Lp_max, 
    int b, int t, int s_ext)
{
    return ((int64_t)b*T + t) * Lp_max + s_ext;
}

__device__ __forceinline__ float logaddexpf(float a, float b) {
    float m = fmaxf(a, b);
    if (isinf(m) && m < 0.0f) return -INFINITY;
    float d = -fabsf(a - b);
    return m + log1pf(expf(d));
}

__global__ void ctc_alpha_kernel(
    const float* __restrict__ log_probs,   // (B, T, V)
    const int*   __restrict__ targets,     // (B, S_max)
    const int*   __restrict__ in_lens,     // (B,)
    const int*   __restrict__ tgt_lens,    // (B,)
    float* __restrict__ alpha,             // (B, T, L'_max)
    float* __restrict__ log_Z,             // (B,)     
    int T_max, int Lp_max, int V, int S_max) 
{
    const int b = blockIdx.x;   // batch index
    const int Lp_b = 2*tgt_lens[b] + 1;
    
    if (tgt_lens[b] == 0) {
        return;
    }

    for (int t = 0; t < in_lens[b]; ++t) {
        for (int s_ext = threadIdx.x; s_ext < Lp_b; s_ext += blockDim.x) {
            const int s_raw = s_ext / 2;
            const bool is_blank = s_ext % 2 == 0;
            const bool non_repeat = (
                !is_blank && 
                (s_raw > 0 && targets[targets_idx(S_max, b, s_raw)] != targets[targets_idx(S_max, b, s_raw - 1)])
            );
            
            if (t == 0) {
                // base case
                if (s_ext <= 1) {
                    int tgt = s_ext == 0 ? 0 : targets[targets_idx(S_max, b, 0)];
                    alpha[alpha_idx(T_max, Lp_max, b, t, s_ext)] = log_probs[log_probs_idx(T_max, V, b, t, tgt)];
                }
            } else {
                // recursive/inductive step
                
                // stay always allowed
                float acc = alpha[alpha_idx(T_max, Lp_max, b, t - 1, s_ext)];
                
                // previous advance always allowed
                if (s_ext > 0) {
                    acc = logaddexpf(acc, alpha[alpha_idx(T_max, Lp_max, b, t - 1, s_ext - 1)]);
                }
                
                // skip advance only allowed if labels differ
                if (non_repeat) {
                    acc = logaddexpf(acc, alpha[alpha_idx(T_max, Lp_max, b, t - 1, s_ext - 2)]);
                }
                
                int tgt = is_blank ? 0 : targets[targets_idx(S_max, b, s_raw)];
                acc += log_probs[log_probs_idx(T_max, V, b, t, tgt)];

                alpha[alpha_idx(T_max, Lp_max, b, t, s_ext)] = acc;
            }
        }
        __syncthreads();
    }
}

std::vector<torch::Tensor> ctc_alpha_cuda(
    torch::Tensor log_probs, 
    torch::Tensor targets,
    torch::Tensor in_lens, 
    torch::Tensor tgt_lens)
{
  TORCH_CHECK(log_probs.is_cuda() && log_probs.dtype() == torch::kFloat32);
  TORCH_CHECK(log_probs.is_contiguous(), "log_probs must be contiguous");
  TORCH_CHECK(targets.is_contiguous(), "targets must be contiguous");
  
  const int B = log_probs.size(0), T = log_probs.size(1), V = log_probs.size(2);
  const int S_max = targets.size(1);
  const int Lp_max = 2 * S_max + 1;

  auto opts = log_probs.options();
  auto alpha = torch::full({B, T, Lp_max}, -INFINITY, opts);
  auto log_Z = torch::empty({B}, opts);

  const int threads = std::min(Lp_max, 256);
  ctc_alpha_kernel<<<B, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      log_probs.data_ptr<float>(), 
      targets.data_ptr<int>(),
      in_lens.data_ptr<int>(), 
      tgt_lens.data_ptr<int>(),
      alpha.data_ptr<float>(), 
      log_Z.data_ptr<float>(),
      T, Lp_max, V, S_max
  );
  return {alpha, log_Z};
}

torch::Tensor ctc_beta_grad_cuda(
    torch::Tensor alpha, 
    torch::Tensor log_Z,
    torch::Tensor log_probs, 
    torch::Tensor targets,
    torch::Tensor in_lens, 
    torch::Tensor tgt_lens,
    torch::Tensor grad_loss) 
{
  return torch::empty({1});
}
"""

cpp_source = r"""
std::vector<torch::Tensor> ctc_alpha_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
    
torch::Tensor ctc_beta_grad_cuda(
    torch::Tensor, 
    torch::Tensor, 
    torch::Tensor, 
    torch::Tensor,
    torch::Tensor, 
    torch::Tensor, 
    torch::Tensor
);
"""

ctc_ext = load_inline(
    name="ctc_ext",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["ctc_alpha_cuda", "ctc_beta_grad_cuda"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    verbose=True,
)


class CTCLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, in_lens, tgt_lens):
        alpha, log_Z = ctc_ext.ctc_alpha_cuda(log_probs, targets, in_lens, tgt_lens)
        ctx.save_for_backward(log_probs, alpha, log_Z, targets, in_lens, tgt_lens)
        # return -log_Z
        return alpha

    @staticmethod
    def backward(ctx, grad_loss):
        log_probs, alpha, log_Z, targets, in_lens, tgt_lens = ctx.saved_tensors
        grad_logits = ctc_ext.ctc_beta_grad_cuda(
            alpha, log_Z, log_probs, targets, in_lens, tgt_lens, grad_loss.contiguous()
        )
        return grad_log_probs, None, None, None
