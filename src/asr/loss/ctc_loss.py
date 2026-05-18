import torch
from torch.utils.cpp_extension import load_inline

cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_FLOAT32(x) \
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be float32")

#define CHECK_INT(x) \
  TORCH_CHECK(x.scalar_type() == torch::kInt, #x " must be int")

#define CHECK_F32_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_FLOAT32(x)

#define CHECK_INT_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x); \
  CHECK_INT(x)
  

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
    if (tgt_lens[b] == 0) {
        return;
    }
    const int Lp_b = 2*tgt_lens[b] + 1;
    const int T_b = in_lens[b];

    for (int t = 0; t < T_b; ++t) {
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
    
    if (threadIdx.x == 0) {
        log_Z[b] = logaddexpf(
            alpha[alpha_idx(T_max, Lp_max, b, T_b - 1, Lp_b - 2)],
            alpha[alpha_idx(T_max, Lp_max, b, T_b - 1, Lp_b - 1)]
        );
    }  
}

__global__ void ctc_beta_kernel(
    const float* __restrict__ log_probs,   // (B, T, V)
    const int*   __restrict__ targets,     // (B, S_max)
    const int*   __restrict__ in_lens,     // (B,)
    const int*   __restrict__ tgt_lens,    // (B,)
    const float* __restrict__ alpha,       // (B, T, L'_max)
    const float* __restrict__ log_Z,       // (B,)
    const float* __restrict__ grad_loss,   // (B,)
    float* __restrict__ grad_logits,       // (B, T, V)
    int T_max, int Lp_max, int V, int S_max)
{
    const int b = blockIdx.x;   // batch index
    if (tgt_lens[b] == 0) {
        return;
    }
    const int Lp_b = 2*tgt_lens[b] + 1;
    const float grad_loss_b = grad_loss[b];

    extern __shared__ float smem[];
    float* beta_cur = smem;
    float* beta_nxt = smem + Lp_max;

    // initialize beta_cur to terminate in final blank or final symbol
    for (int s_ext = threadIdx.x; s_ext < Lp_b; s_ext += blockDim.x) {
        beta_cur[s_ext] = -INFINITY;
    }
    if (threadIdx.x == 0) {
        beta_cur[Lp_b - 1] = 0.0;
        beta_cur[Lp_b - 2] = 0.0;
    }

    __syncthreads();

    for (int t = in_lens[b] - 1; t >= 0; --t) {
        // compute posteriors
        // beta_nxt is stale; reuse for posterior computation
        float* posts = beta_nxt;
        for (int s_ext = threadIdx.x; s_ext < Lp_b; s_ext += blockDim.x) {
            posts[s_ext] = expf(alpha[alpha_idx(T_max, Lp_max, b, t, s_ext)] + beta_cur[s_ext] - log_Z[b]);
        }
        __syncthreads();

        // scatter-add posteriors to vocabulary indices
        for (int s_ext = threadIdx.x; s_ext < Lp_b; s_ext += blockDim.x) {
            const int s_raw = s_ext / 2;
            const bool is_blank = s_ext % 2 == 0;
            const int tgt = is_blank ? 0 : targets[targets_idx(S_max, b, s_raw)];
            atomicAdd(&grad_logits[log_probs_idx(T_max, V, b, t, tgt)], posts[s_ext]);
        }
        __syncthreads();

        // compute gradient
        for (int v = threadIdx.x; v < V; v += blockDim.x) {
            int64_t idx = log_probs_idx(T_max, V, b, t, v);
            grad_logits[idx] = grad_loss_b * (expf(log_probs[idx]) - grad_logits[idx]);
        }
        __syncthreads();
        
        if (t == 0) {
            break;
        }
        
        // compute beta_nxt
        for (int s_ext = threadIdx.x; s_ext < Lp_b; s_ext += blockDim.x) {
            const int s_raw = s_ext / 2;
            const bool is_blank = s_ext % 2 == 0;
            const bool non_repeat = (
                !is_blank && 
                (s_raw < tgt_lens[b] - 1) && 
                (targets[targets_idx(S_max, b, s_raw)] != targets[targets_idx(S_max, b, s_raw + 1)])
            );
    
            // stay always allowed
            int tgt = is_blank ? 0 : targets[targets_idx(S_max, b, s_raw)];
            float acc = beta_cur[s_ext] + log_probs[log_probs_idx(T_max, V, b, t, tgt)];

            // previous advance always allowed
            if (s_ext < Lp_b - 1) {
                int tgt = is_blank ? targets[targets_idx(S_max, b, s_raw)] : 0;
                acc = logaddexpf(acc, beta_cur[s_ext + 1] + log_probs[log_probs_idx(T_max, V, b, t, tgt)]);
            }

            // skip advance only allowed if labels differ
            if (s_ext < Lp_b - 2 && non_repeat) {
                int tgt = targets[targets_idx(S_max, b, s_raw + 1)];
                acc = logaddexpf(acc, beta_cur[s_ext + 2] + log_probs[log_probs_idx(T_max, V, b, t, tgt)]);
            }

            beta_nxt[s_ext] = acc;
        }
        __syncthreads();

        // switch buffers
        float* beta_tmp = beta_cur;
        beta_cur = beta_nxt;
        beta_nxt = beta_tmp;
    }
}

std::vector<torch::Tensor> ctc_alpha_cuda(
    torch::Tensor log_probs, 
    torch::Tensor targets,
    torch::Tensor in_lens, 
    torch::Tensor tgt_lens)
{
    CHECK_F32_INPUT(log_probs);
    CHECK_INT_INPUT(targets);
    CHECK_INT_INPUT(in_lens);
    CHECK_INT_INPUT(tgt_lens);

    const int B = log_probs.size(0);
    const int T = log_probs.size(1);
    const int V = log_probs.size(2);
    const int S_max = targets.size(1);
    const int Lp_max = 2 * S_max + 1;
    
    auto opts = log_probs.options();
    auto alpha = torch::empty({B, T, Lp_max}, opts);
    alpha.select(1, 0).fill_(-INFINITY);
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
    CHECK_F32_INPUT(alpha);
    CHECK_F32_INPUT(log_Z);
    CHECK_F32_INPUT(log_probs);
    CHECK_INT_INPUT(targets);
    CHECK_INT_INPUT(in_lens);
    CHECK_INT_INPUT(tgt_lens);
    CHECK_F32_INPUT(grad_loss);
    
    const int B = log_probs.size(0);
    const int T = log_probs.size(1);
    const int V = log_probs.size(2);
    const int S_max = targets.size(1);
    const int Lp_max = 2 * S_max + 1;
    
    auto opts = log_probs.options();
    auto grad_logits = torch::zeros({B, T, V}, opts);
    
    const int threads = std::min(Lp_max, 256);
    ctc_beta_kernel<<<B, threads, 2 * Lp_max * sizeof(float), at::cuda::getCurrentCUDAStream()>>>(
        log_probs.data_ptr<float>(),
        targets.data_ptr<int>(),
        in_lens.data_ptr<int>(),
        tgt_lens.data_ptr<int>(),
        alpha.data_ptr<float>(),
        log_Z.data_ptr<float>(),
        grad_loss.data_ptr<float>(),
        grad_logits.data_ptr<float>(),
        T, Lp_max, V, S_max
    );
    
    return grad_logits;
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
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)


class CTCLossFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_probs, targets, in_lens, tgt_lens):
        alpha, log_Z = ctc_ext.ctc_alpha_cuda(log_probs, targets, in_lens, tgt_lens)
        ctx.save_for_backward(log_probs, alpha, log_Z, targets, in_lens, tgt_lens)
        return -log_Z

    @staticmethod
    def backward(ctx, grad_loss):
        log_probs, alpha, log_Z, targets, in_lens, tgt_lens = ctx.saved_tensors
        grad_logits = ctc_ext.ctc_beta_grad_cuda(
            alpha, log_Z, log_probs, targets, in_lens, tgt_lens, grad_loss.contiguous()
        )
        return grad_logits, None, None, None


class CTCLoss(torch.nn.Module):
    def __init__(self, reduction="mean", zero_infinity=False):
        super().__init__()
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(self, log_probs, targets, in_lens, tgt_lens):
        loss = CTCLossFn.apply(log_probs, targets, in_lens, tgt_lens)
        if self.zero_infinity:
            loss[loss.isinf()] = 0.0
        if self.reduction == "mean":
            return (loss / tgt_lens).mean()
        elif self.reduction == "none":
            return loss
        raise NotImplementedError(f"unknown reduction {self.reduction}")
