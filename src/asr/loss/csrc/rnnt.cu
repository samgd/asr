// Generated with Claude Code (https://claude.com/claude-code). Opus 4.8 xhigh.
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <algorithm>
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

#define CUBLAS_CHECK(call)                                                       \
    do {                                                                         \
        cublasStatus_t status_ = (call);                                         \
        STD_TORCH_CHECK(status_ == CUBLAS_STATUS_SUCCESS, "cuBLAS call failed"); \
    } while (0)

// Finite "log zero", matching the pure-PyTorch reference (asr/loss/rnnt.py).
#define NEG (-1e30f)
// Per-node lattice kernels stride over V/d with this many threads; power of two for block reductions.
#define NODE_THREADS 128
// Elementwise kernels.
#define EW_THREADS 256
// Joint matmul is tiled over this many lattice nodes at a time (bounds transient memory).
// 32768 balances cuBLAS GEMM efficiency against transient (chunk * (V + 2d)) footprint.
#define NODE_CHUNK 32768

__device__ __forceinline__ float logaddexpf(float a, float b) {
    float m = fmaxf(a, b);
    if (isinf(m) && m < 0.0f) return -INFINITY;
    float d = -fabsf(a - b);
    return m + log1pf(expf(d));
}

static cudaStream_t get_stream(const torch::stable::Tensor& t) {
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream_ptr));
    return static_cast<cudaStream_t>(stream_ptr);
}

static cublasHandle_t get_cublas() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) CUBLAS_CHECK(cublasCreate(&handle));
    return handle;
}

// FP32 by default; opt into TF32 tensor cores (~2x GEMM throughput, ~1e-2 loss error).
// The cuBLAS handle is shared across calls, so set the mode explicitly every time.
static void set_math_mode(cublasHandle_t handle, bool tf32) {
    CUBLAS_CHECK(cublasSetMathMode(handle, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
}

// ---- index helpers -------------------------------------------------------
__forceinline__ __device__ int64_t act_idx(int L, int d, int b, int i, int k) {
    return ((int64_t)b * L + i) * d + k;  // encoder (L=T) / decoder (L=U) layout (B, L, d)
}
__forceinline__ __device__ int64_t node_idx(int T, int U, int b, int t, int u) {
    return ((int64_t)b * T + t) * U + u;  // (B, T, U)
}
__forceinline__ __device__ int tgt_idx(int S, int b, int u) { return b * S + u; }

__forceinline__ __device__ void decode_node(int n, int T, int U, int& b, int& t, int& u) {
    b = n / (T * U);
    int r = n % (T * U);
    t = r / U;
    u = r % U;
}

// block-wide reductions over `red` (length == blockDim.x, a power of two)
__device__ __forceinline__ float block_reduce_max(float* red, float val) {
    red[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float out = red[0];
    __syncthreads();
    return out;
}
__device__ __forceinline__ float block_reduce_sum(float* red, float val) {
    red[threadIdx.x] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float out = red[0];
    __syncthreads();
    return out;
}

// ---- joint network, tiled over lattice nodes -----------------------------
// hidden[i, k] = tanh(enc[b,t,k] + dec[b,u,k]) for the chunk of nodes [c0, c0+Cn).
__global__ void rnnt_build_hidden(const float* __restrict__ encoder,  // (B, T, d)
                                  const float* __restrict__ decoder,  // (B, U, d)
                                  float* __restrict__ hidden,         // (Cn, d)
                                  int T, int U, int d, int c0, int Cn) {
    const int64_t total = (int64_t)Cn * d;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int i = idx / d;
        const int k = idx % d;
        int b, t, u;
        decode_node(c0 + i, T, U, b, t, u);
        hidden[idx] = tanhf(encoder[act_idx(T, d, b, t, k)] + decoder[act_idx(U, d, b, u, k)]);
    }
}

// log-softmax over V, keeping only the two slim edge log-probs the lattice needs.
// One block per chunk node; threads stride over V; logits are discarded.
__global__ void rnnt_logsoftmax_slim(const float* __restrict__ logits,  // (Cn, V)
                                     const int* __restrict__ targets,   // (B, S)
                                     const int* __restrict__ tgt_lens,  // (B,)
                                     float* __restrict__ log_P_blank,   // (B, T, U)
                                     float* __restrict__ log_P_y,       // (B, T, U)
                                     int T, int U, int V, int S, int blank, int c0, int Cn) {
    const int i = blockIdx.x;
    if (i >= Cn) return;
    int b, t, u;
    decode_node(c0 + i, T, U, b, t, u);
    const float* row = logits + (int64_t)i * V;

    extern __shared__ float red[];
    float local_max = -INFINITY;
    for (int v = threadIdx.x; v < V; v += blockDim.x) local_max = fmaxf(local_max, row[v]);
    const float M = block_reduce_max(red, local_max);
    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) local_sum += expf(row[v] - M);
    const float lse = M + logf(block_reduce_sum(red, local_sum));

    if (threadIdx.x == 0) {
        const int y = (u < tgt_lens[b]) ? targets[tgt_idx(S, b, u)] : -1;
        log_P_blank[node_idx(T, U, b, t, u)] = row[blank] - lse;
        log_P_y[node_idx(T, U, b, t, u)] = (y >= 0) ? row[y] - lse : NEG;
    }
}

// d loss / d logit, written in place over the logits buffer. For each node:
//   glogit[v] = grad_loss * (softmax[v]*P - blank_post*[v==blank] - label_post*[v==y]).
// One block per chunk node; threads stride over V.
__global__ void rnnt_build_glogit(float* __restrict__ logits,            // (Cn, V) in -> glogit out
                                  const float* __restrict__ blank_post,  // (B, T, U)
                                  const float* __restrict__ label_post,  // (B, T, U)
                                  const float* __restrict__ grad_loss,   // (B,)
                                  const int* __restrict__ targets,       // (B, S)
                                  const int* __restrict__ tgt_lens,      // (B,)
                                  int T, int U, int V, int S, int blank, int c0, int Cn) {
    const int i = blockIdx.x;
    if (i >= Cn) return;
    int b, t, u;
    decode_node(c0 + i, T, U, b, t, u);
    float* row = logits + (int64_t)i * V;

    const float bp = blank_post[node_idx(T, U, b, t, u)];
    const float lp = label_post[node_idx(T, U, b, t, u)];
    if (bp == 0.0f && lp == 0.0f) {  // padded / zero-posterior node contributes nothing
        for (int v = threadIdx.x; v < V; v += blockDim.x) row[v] = 0.0f;
        return;
    }
    const float P = bp + lp;
    const float gl = grad_loss[b];
    const int y = (u < tgt_lens[b]) ? targets[tgt_idx(S, b, u)] : -1;

    extern __shared__ float red[];
    float local_max = -INFINITY;
    for (int v = threadIdx.x; v < V; v += blockDim.x) local_max = fmaxf(local_max, row[v]);
    const float M = block_reduce_max(red, local_max);
    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) local_sum += expf(row[v] - M);
    const float lse = M + logf(block_reduce_sum(red, local_sum));

    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float g = gl * expf(row[v] - lse) * P;
        if (v == blank) g -= gl * bp;
        if (v == y) g -= gl * lp;
        row[v] = g;
    }
}

// grad_pre = grad_hidden * (1 - hidden^2), scattered to encoder/decoder (low-contention atomics).
__global__ void rnnt_scatter_act(const float* __restrict__ grad_hidden,  // (Cn, d)
                                 const float* __restrict__ hidden,       // (Cn, d)
                                 float* __restrict__ grad_encoder,       // (B, T, d)
                                 float* __restrict__ grad_decoder,       // (B, U, d)
                                 int T, int U, int d, int c0, int Cn) {
    const int64_t total = (int64_t)Cn * d;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        const int i = idx / d;
        const int k = idx % d;
        const float h = hidden[idx];
        const float gpre = grad_hidden[idx] * (1.0f - h * h);
        int b, t, u;
        decode_node(c0 + i, T, U, b, t, u);
        atomicAdd(&grad_encoder[act_idx(T, d, b, t, k)], gpre);
        atomicAdd(&grad_decoder[act_idx(U, d, b, u, k)], gpre);
    }
}

// ---- forward: alpha (antidiagonal wavefront) -----------------------------
__global__ void rnnt_alpha(const float* __restrict__ log_P_blank,  // (B, T, U)
                           const float* __restrict__ log_P_y,      // (B, T, U)
                           const int* __restrict__ in_lens,        // (B,)
                           const int* __restrict__ tgt_lens,       // (B,)
                           float* __restrict__ alpha,              // (B, T, U)
                           float* __restrict__ log_prob,           // (B,)
                           float* __restrict__ loss,               // (B,)
                           int T, int U) {
    const int b = blockIdx.x;
    for (int diag = 0; diag <= T + U - 2; ++diag) {
        for (int u = threadIdx.x; u < U; u += blockDim.x) {
            const int t = diag - u;
            if (t < 0 || t >= T) continue;
            if (t == 0 && u == 0) {
                alpha[node_idx(T, U, b, 0, 0)] = 0.0f;
                continue;
            }
            float term_h = NEG, term_v = NEG;
            if (t > 0) term_h = alpha[node_idx(T, U, b, t - 1, u)] + log_P_blank[node_idx(T, U, b, t - 1, u)];
            if (u > 0) term_v = alpha[node_idx(T, U, b, t, u - 1)] + log_P_y[node_idx(T, U, b, t, u - 1)];
            alpha[node_idx(T, U, b, t, u)] = logaddexpf(term_h, term_v);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int t_term = in_lens[b] - 1;
        const int u_term = tgt_lens[b];  // U_b - 1
        const float lp = alpha[node_idx(T, U, b, t_term, u_term)] + log_P_blank[node_idx(T, U, b, t_term, u_term)];
        log_prob[b] = lp;
        loss[b] = -lp;
    }
}

// ---- backward: beta on the fly -> slim edge posteriors -------------------
// grid = B, threads over u, reverse antidiagonal sweep. beta lives in a (U+1)
// ping-pong shared buffer and is never stored; only the two edge posteriors are.
__global__ void rnnt_betapost(const float* __restrict__ log_P_blank,  // (B, T, U)
                              const float* __restrict__ log_P_y,      // (B, T, U)
                              const float* __restrict__ alpha,        // (B, T, U)
                              const float* __restrict__ log_prob,     // (B,)
                              const int* __restrict__ in_lens,        // (B,)
                              const int* __restrict__ tgt_lens,       // (B,)
                              float* __restrict__ blank_post,         // (B, T, U)
                              float* __restrict__ label_post,         // (B, T, U)
                              int T, int U) {
    const int b = blockIdx.x;
    const int T_b = in_lens[b];
    const int U_b = tgt_lens[b] + 1;
    const float lp_total = log_prob[b];

    extern __shared__ float smem[];
    float* bcur = smem;            // U + 1
    float* bnxt = smem + (U + 1);  // U + 1

    for (int u = threadIdx.x; u < U + 1; u += blockDim.x) {
        bcur[u] = NEG;
        bnxt[u] = NEG;
    }
    __syncthreads();

    for (int diag = T + U - 2; diag >= 0; --diag) {
        for (int u = threadIdx.x; u < U; u += blockDim.x) {
            const int t = diag - u;
            float val = NEG;
            if (t >= 0 && t < T) {
                const int64_t nidx = node_idx(T, U, b, t, u);
                const bool node_valid = (t < T_b) && (u < U_b);
                if (node_valid) {
                    const bool is_term = (t == T_b - 1) && (u == U_b - 1);
                    const bool can_blank = (t < T_b - 1);
                    const bool can_label = (u < U_b - 1);
                    const float lp_b = log_P_blank[nidx];
                    const float lp_y = log_P_y[nidx];
                    const float beta_t1 = bnxt[u];      // beta[t+1, u]
                    const float beta_u1 = bnxt[u + 1];  // beta[t, u+1]

                    const float blank_term = is_term ? lp_b : (can_blank ? lp_b + beta_t1 : NEG);
                    const float label_term = can_label ? lp_y + beta_u1 : NEG;
                    val = logaddexpf(blank_term, label_term);

                    const float a = alpha[nidx];
                    float bp =
                        is_term ? expf(a + lp_b - lp_total) : (can_blank ? expf(a + lp_b + beta_t1 - lp_total) : 0.0f);
                    float lp = can_label ? expf(a + lp_y + beta_u1 - lp_total) : 0.0f;
                    blank_post[nidx] = bp;
                    label_post[nidx] = lp;
                } else {
                    blank_post[nidx] = 0.0f;
                    label_post[nidx] = 0.0f;
                }
            }
            bcur[u] = val;
        }
        __syncthreads();
        float* tmp = bcur;
        bcur = bnxt;
        bnxt = tmp;
        __syncthreads();
    }
}

// ---- host wrappers -------------------------------------------------------
std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor,
           torch::stable::Tensor>
rnnt_forward_cuda(const torch::stable::Tensor& encoder, const torch::stable::Tensor& decoder,
                  const torch::stable::Tensor& joint_W, const torch::stable::Tensor& targets,
                  const torch::stable::Tensor& in_lens, const torch::stable::Tensor& tgt_lens, int64_t blank_idx,
                  int64_t tf32) {
    CHECK_F32_INPUT(encoder);
    CHECK_F32_INPUT(decoder);
    CHECK_F32_INPUT(joint_W);
    CHECK_INT_INPUT(targets);
    CHECK_INT_INPUT(in_lens);
    CHECK_INT_INPUT(tgt_lens);

    const int B = encoder.sizes()[0];
    const int T = encoder.sizes()[1];
    const int d = encoder.sizes()[2];
    const int U = decoder.sizes()[1];
    const int V = joint_W.sizes()[0];
    const int S = targets.sizes()[1];
    const int N = B * T * U;

    auto log_P_blank = torch::stable::new_empty(encoder, {B, T, U});
    auto log_P_y = torch::stable::new_empty(encoder, {B, T, U});
    auto alpha = torch::stable::new_empty(encoder, {B, T, U});
    auto log_prob = torch::stable::new_empty(encoder, {B});
    auto loss = torch::stable::new_empty(encoder, {B});

    cudaStream_t stream = get_stream(encoder);
    cublasHandle_t handle = get_cublas();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    set_math_mode(handle, tf32 != 0);

    const int chunk = std::min(N, NODE_CHUNK);
    auto hidden = torch::stable::new_empty(encoder, {chunk, d});
    auto logits = torch::stable::new_empty(encoder, {chunk, V});
    const float one = 1.0f, zero = 0.0f;
    const size_t red_smem = NODE_THREADS * sizeof(float);

    for (int c0 = 0; c0 < N; c0 += chunk) {
        const int Cn = std::min(chunk, N - c0);
        const int hid_blocks = (int)(((int64_t)Cn * d + EW_THREADS - 1) / EW_THREADS);
        rnnt_build_hidden<<<hid_blocks, EW_THREADS, 0, stream>>>(encoder.const_data_ptr<float>(),
                                                                 decoder.const_data_ptr<float>(),
                                                                 hidden.mutable_data_ptr<float>(), T, U, d, c0, Cn);
        // logits (Cn, V) = hidden (Cn, d) @ joint_W^T (d, V)  [column-major: L_cm = Wᵀ · H]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, V, Cn, d, &one, joint_W.const_data_ptr<float>(), d,
                                 hidden.const_data_ptr<float>(), d, &zero, logits.mutable_data_ptr<float>(), V));
        rnnt_logsoftmax_slim<<<Cn, NODE_THREADS, red_smem, stream>>>(
            logits.const_data_ptr<float>(), targets.const_data_ptr<int>(), tgt_lens.const_data_ptr<int>(),
            log_P_blank.mutable_data_ptr<float>(), log_P_y.mutable_data_ptr<float>(), T, U, V, S, (int)blank_idx, c0,
            Cn);
    }

    const int alpha_threads = std::min(U, 256);
    rnnt_alpha<<<B, alpha_threads, 0, stream>>>(log_P_blank.const_data_ptr<float>(), log_P_y.const_data_ptr<float>(),
                                                in_lens.const_data_ptr<int>(), tgt_lens.const_data_ptr<int>(),
                                                alpha.mutable_data_ptr<float>(), log_prob.mutable_data_ptr<float>(),
                                                loss.mutable_data_ptr<float>(), T, U);

    return {std::move(loss), std::move(alpha), std::move(log_P_blank), std::move(log_P_y), std::move(log_prob)};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor> rnnt_backward_cuda(
    const torch::stable::Tensor& encoder, const torch::stable::Tensor& decoder, const torch::stable::Tensor& joint_W,
    const torch::stable::Tensor& targets, const torch::stable::Tensor& in_lens, const torch::stable::Tensor& tgt_lens,
    int64_t blank_idx, const torch::stable::Tensor& alpha, const torch::stable::Tensor& log_P_blank,
    const torch::stable::Tensor& log_P_y, const torch::stable::Tensor& log_prob, const torch::stable::Tensor& grad_loss,
    int64_t tf32) {
    CHECK_F32_INPUT(encoder);
    CHECK_F32_INPUT(decoder);
    CHECK_F32_INPUT(joint_W);
    CHECK_INT_INPUT(targets);
    CHECK_INT_INPUT(in_lens);
    CHECK_INT_INPUT(tgt_lens);
    CHECK_F32_INPUT(alpha);
    CHECK_F32_INPUT(log_P_blank);
    CHECK_F32_INPUT(log_P_y);
    CHECK_F32_INPUT(log_prob);
    CHECK_F32_INPUT(grad_loss);

    const int B = encoder.sizes()[0];
    const int T = encoder.sizes()[1];
    const int d = encoder.sizes()[2];
    const int U = decoder.sizes()[1];
    const int V = joint_W.sizes()[0];
    const int S = targets.sizes()[1];
    const int N = B * T * U;

    auto blank_post = torch::stable::new_empty(encoder, {B, T, U});
    auto label_post = torch::stable::new_empty(encoder, {B, T, U});
    auto grad_encoder = torch::stable::new_zeros(encoder, {B, T, d});
    auto grad_decoder = torch::stable::new_zeros(encoder, {B, U, d});
    auto grad_joint_W = torch::stable::new_zeros(joint_W, {V, d});

    cudaStream_t stream = get_stream(encoder);
    cublasHandle_t handle = get_cublas();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    set_math_mode(handle, tf32 != 0);

    const int beta_threads = std::min(U, 256);
    const size_t beta_smem = (size_t)2 * (U + 1) * sizeof(float);
    rnnt_betapost<<<B, beta_threads, beta_smem, stream>>>(
        log_P_blank.const_data_ptr<float>(), log_P_y.const_data_ptr<float>(), alpha.const_data_ptr<float>(),
        log_prob.const_data_ptr<float>(), in_lens.const_data_ptr<int>(), tgt_lens.const_data_ptr<int>(),
        blank_post.mutable_data_ptr<float>(), label_post.mutable_data_ptr<float>(), T, U);

    const int chunk = std::min(N, NODE_CHUNK);
    auto hidden = torch::stable::new_empty(encoder, {chunk, d});
    auto glogit = torch::stable::new_empty(encoder, {chunk, V});  // logits in, d loss / d logit out
    auto grad_hidden = torch::stable::new_empty(encoder, {chunk, d});
    const float one = 1.0f, zero = 0.0f;
    const size_t red_smem = NODE_THREADS * sizeof(float);

    for (int c0 = 0; c0 < N; c0 += chunk) {
        const int Cn = std::min(chunk, N - c0);
        const int act_blocks = (int)(((int64_t)Cn * d + EW_THREADS - 1) / EW_THREADS);
        rnnt_build_hidden<<<act_blocks, EW_THREADS, 0, stream>>>(encoder.const_data_ptr<float>(),
                                                                 decoder.const_data_ptr<float>(),
                                                                 hidden.mutable_data_ptr<float>(), T, U, d, c0, Cn);
        // logits (Cn, V) = hidden @ joint_W^T  (recompute; never stored across chunks)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, V, Cn, d, &one, joint_W.const_data_ptr<float>(), d,
                                 hidden.const_data_ptr<float>(), d, &zero, glogit.mutable_data_ptr<float>(), V));
        rnnt_build_glogit<<<Cn, NODE_THREADS, red_smem, stream>>>(
            glogit.mutable_data_ptr<float>(), blank_post.const_data_ptr<float>(), label_post.const_data_ptr<float>(),
            grad_loss.const_data_ptr<float>(), targets.const_data_ptr<int>(), tgt_lens.const_data_ptr<int>(), T, U, V,
            S, (int)blank_idx, c0, Cn);
        // grad_joint_W += glogitᵀ · hidden  (contraction over the chunk's nodes; accumulate with beta=1)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, V, Cn, &one, hidden.const_data_ptr<float>(), d,
                                 glogit.const_data_ptr<float>(), V, &one, grad_joint_W.mutable_data_ptr<float>(), d));
        // grad_hidden (Cn, d) = glogit (Cn, V) @ joint_W (V, d)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, Cn, V, &one, joint_W.const_data_ptr<float>(), d,
                                 glogit.const_data_ptr<float>(), V, &zero, grad_hidden.mutable_data_ptr<float>(), d));
        rnnt_scatter_act<<<act_blocks, EW_THREADS, 0, stream>>>(
            grad_hidden.const_data_ptr<float>(), hidden.const_data_ptr<float>(), grad_encoder.mutable_data_ptr<float>(),
            grad_decoder.mutable_data_ptr<float>(), T, U, d, c0, Cn);
    }

    return {std::move(grad_encoder), std::move(grad_decoder), std::move(grad_joint_W)};
}
