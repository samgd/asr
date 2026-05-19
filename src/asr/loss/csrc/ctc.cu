#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <tuple>

#define CHECK_CUDA(x) STD_TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) STD_TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define CHECK_FLOAT32(x) STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Float, #x " must be float32")

#define CHECK_INT(x) STD_TORCH_CHECK(x.scalar_type() == torch::headeronly::ScalarType::Int, #x " must be int")

#define CHECK_F32_INPUT(x) \
    CHECK_CUDA(x);         \
    CHECK_CONTIGUOUS(x);   \
    CHECK_FLOAT32(x)

#define CHECK_INT_INPUT(x) \
    CHECK_CUDA(x);         \
    CHECK_CONTIGUOUS(x);   \
    CHECK_INT(x)

__forceinline__ __device__ int64_t log_probs_idx(int T, int V, int b, int t, int v) {
    return ((int64_t)b * T + t) * V + v;
}

__forceinline__ __device__ int64_t targets_idx(int S_max, int b, int s_raw) { return (int64_t)b * S_max + s_raw; }

__forceinline__ __device__ int64_t alpha_idx(int T, int Lp_max, int b, int t, int s_ext) {
    return ((int64_t)b * T + t) * Lp_max + s_ext;
}

__device__ __forceinline__ float logaddexpf(float a, float b) {
    float m = fmaxf(a, b);
    if (isinf(m) && m < 0.0f) return -INFINITY;
    float d = -fabsf(a - b);
    return m + log1pf(expf(d));
}

cudaStream_t get_stream(const torch::stable::Tensor& t) {
    // For now, we rely on the raw shim API to get the current CUDA stream.
    // This will be improved in a future release.
    // When using a raw shim API, we need to use TORCH_ERROR_CODE_CHECK to
    // check the error code and throw an appropriate runtime_error otherwise.
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream_ptr));
    return static_cast<cudaStream_t>(stream_ptr);
}

__global__ void ctc_alpha_kernel(const float* __restrict__ log_probs,  // (B, T, V)
                                 const int* __restrict__ targets,      // (B, S_max)
                                 const int* __restrict__ in_lens,      // (B,)
                                 const int* __restrict__ tgt_lens,     // (B,)
                                 float* __restrict__ alpha,            // (B, T, L'_max)
                                 float* __restrict__ log_Z,            // (B,)
                                 int T_max, int Lp_max, int V, int S_max) {
    const int b = blockIdx.x;  // batch index
    if (tgt_lens[b] == 0) {
        return;
    }
    const int Lp_b = 2 * tgt_lens[b] + 1;
    const int T_b = in_lens[b];

    for (int t = 0; t < T_b; ++t) {
        for (int s_ext = threadIdx.x; s_ext < Lp_b; s_ext += blockDim.x) {
            const int s_raw = s_ext / 2;
            const bool is_blank = s_ext % 2 == 0;
            const bool non_repeat = (!is_blank && (s_raw > 0 && targets[targets_idx(S_max, b, s_raw)] !=
                                                                    targets[targets_idx(S_max, b, s_raw - 1)]));

            if (t == 0) {
                // base case
                if (s_ext <= 1) {
                    int tgt = s_ext == 0 ? 0 : targets[targets_idx(S_max, b, 0)];
                    alpha[alpha_idx(T_max, Lp_max, b, t, s_ext)] = log_probs[log_probs_idx(T_max, V, b, t, tgt)];
                } else {
                    alpha[alpha_idx(T_max, Lp_max, b, t, s_ext)] = -INFINITY;
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
        log_Z[b] = logaddexpf(alpha[alpha_idx(T_max, Lp_max, b, T_b - 1, Lp_b - 2)],
                              alpha[alpha_idx(T_max, Lp_max, b, T_b - 1, Lp_b - 1)]);
    }
}

__global__ void ctc_beta_kernel(const float* __restrict__ log_probs,  // (B, T, V)
                                const int* __restrict__ targets,      // (B, S_max)
                                const int* __restrict__ in_lens,      // (B,)
                                const int* __restrict__ tgt_lens,     // (B,)
                                const float* __restrict__ alpha,      // (B, T, L'_max)
                                const float* __restrict__ log_Z,      // (B,)
                                const float* __restrict__ grad_loss,  // (B,)
                                float* __restrict__ grad_logits,      // (B, T, V)
                                int T_max, int Lp_max, int V, int S_max) {
    const int b = blockIdx.x;  // batch index
    if (tgt_lens[b] == 0) {
        return;
    }
    const int Lp_b = 2 * tgt_lens[b] + 1;
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
            const bool non_repeat =
                (!is_blank && (s_raw < tgt_lens[b] - 1) &&
                 (targets[targets_idx(S_max, b, s_raw)] != targets[targets_idx(S_max, b, s_raw + 1)]));

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

std::tuple<torch::stable::Tensor, torch::stable::Tensor> ctc_alpha_cuda(const torch::stable::Tensor& log_probs,
                                                                        const torch::stable::Tensor& targets,
                                                                        const torch::stable::Tensor& in_lens,
                                                                        const torch::stable::Tensor& tgt_lens) {
    CHECK_F32_INPUT(log_probs);
    CHECK_INT_INPUT(targets);
    CHECK_INT_INPUT(in_lens);
    CHECK_INT_INPUT(tgt_lens);

    auto shape = log_probs.sizes();
    const int B = shape[0];
    const int T = shape[1];
    const int V = shape[2];
    const int S_max = targets.sizes()[1];
    const int Lp_max = 2 * S_max + 1;

    auto alpha = torch::stable::new_empty(log_probs, {B, T, Lp_max});
    auto log_Z = torch::stable::new_empty(log_probs, {B});

    const int threads = std::min(Lp_max, 256);
    ctc_alpha_kernel<<<B, threads, 0, get_stream(log_probs)>>>(
        log_probs.const_data_ptr<float>(), targets.const_data_ptr<int>(), in_lens.const_data_ptr<int>(),
        tgt_lens.const_data_ptr<int>(), alpha.mutable_data_ptr<float>(), log_Z.mutable_data_ptr<float>(), T, Lp_max, V,
        S_max);
    return {std::move(alpha), std::move(log_Z)};
}

torch::stable::Tensor ctc_grad_cuda(const torch::stable::Tensor& alpha, const torch::stable::Tensor& log_Z,
                                    const torch::stable::Tensor& log_probs, const torch::stable::Tensor& targets,
                                    const torch::stable::Tensor& in_lens, const torch::stable::Tensor& tgt_lens,
                                    const torch::stable::Tensor& grad_loss) {
    CHECK_F32_INPUT(alpha);
    CHECK_F32_INPUT(log_Z);
    CHECK_F32_INPUT(log_probs);
    CHECK_INT_INPUT(targets);
    CHECK_INT_INPUT(in_lens);
    CHECK_INT_INPUT(tgt_lens);
    CHECK_F32_INPUT(grad_loss);

    auto shape = log_probs.sizes();
    const int B = shape[0];
    const int T = shape[1];
    const int V = shape[2];
    const int S_max = targets.sizes()[1];
    const int Lp_max = 2 * S_max + 1;

    auto grad_logits = torch::stable::new_zeros(log_probs, {B, T, V});

    const int threads = std::min(Lp_max, 256);

    ctc_beta_kernel<<<B, threads, 2 * Lp_max * sizeof(float), get_stream(log_probs)>>>(
        log_probs.const_data_ptr<float>(), targets.const_data_ptr<int>(), in_lens.const_data_ptr<int>(),
        tgt_lens.const_data_ptr<int>(), alpha.const_data_ptr<float>(), log_Z.const_data_ptr<float>(),
        grad_loss.const_data_ptr<float>(), grad_logits.mutable_data_ptr<float>(), T, Lp_max, V, S_max);

    return grad_logits;
}