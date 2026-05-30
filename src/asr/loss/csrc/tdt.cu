// Generated with Claude Code (https://claude.com/claude-code). Opus 4.8 xhigh.
//
// Fused CUDA TDT (Token-and-Duration Transducer) loss. Mirrors rnnt.cu's memory
// discipline: the joint network runs tiled over lattice nodes so the dense
// (B,T,U,V) tensor is never materialized. TDT adds a second, independently
// normalized duration head; an edge with duration d advances time by d frames,
// blank requires d>=1, and the objective is exact-landing on a (T+1)-frame grid:
//   loss = -alpha[in_lens, tgt_lens].
// The math oracle is tests/loss/tdt_reference.py.
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

// Finite "log zero", matching the pure-PyTorch reference (tests/loss/tdt_reference.py).
#define NEG (-1e30f)
// Per-node lattice kernels stride over V/D with this many threads; power of two for block reductions.
#define NODE_THREADS 128
// Elementwise kernels.
#define EW_THREADS 256
// Joint matmul is tiled over this many lattice nodes at a time (bounds transient memory).
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
static void set_math_mode(cublasHandle_t handle, bool tf32) {
    CUBLAS_CHECK(cublasSetMathMode(handle, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
}

// ---- index helpers -------------------------------------------------------
__forceinline__ __device__ int64_t act_idx(int L, int d, int b, int i, int k) {
    return ((int64_t)b * L + i) * d + k;  // encoder (L=T) / decoder (L=U) layout (B, L, d)
}
__forceinline__ __device__ int64_t node3(int T, int U, int b, int t, int u) {
    return ((int64_t)b * T + t) * U + u;  // (B, T, U)
}
__forceinline__ __device__ int64_t alpha_idx(int T, int U, int b, int t, int u) {
    return ((int64_t)b * (T + 1) + t) * U + u;  // (B, T+1, U)
}
__forceinline__ __device__ int64_t dur_idx(int T, int U, int D, int b, int t, int u, int k) {
    return (((int64_t)b * T + t) * U + u) * D + k;  // (B, T, U, D)
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
__global__ void tdt_build_hidden(const float* __restrict__ encoder,  // (B, T, d)
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

// Token log-softmax over V, keeping only the two slim edge log-probs the lattice needs.
__global__ void tdt_logsoftmax_tok(const float* __restrict__ logits,  // (Cn, V)
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
        log_P_blank[node3(T, U, b, t, u)] = row[blank] - lse;
        log_P_y[node3(T, U, b, t, u)] = (y >= 0) ? row[y] - lse : NEG;
    }
}

// Duration log-softmax over D; writes the full (B,T,U,D) duration log-prob tensor.
__global__ void tdt_logsoftmax_dur(const float* __restrict__ logits,  // (Cn, D)
                                   float* __restrict__ log_P_dur,     // (B, T, U, D)
                                   int T, int U, int D, int c0, int Cn) {
    const int i = blockIdx.x;
    if (i >= Cn) return;
    int b, t, u;
    decode_node(c0 + i, T, U, b, t, u);
    const float* row = logits + (int64_t)i * D;

    extern __shared__ float red[];
    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < D; k += blockDim.x) local_max = fmaxf(local_max, row[k]);
    const float M = block_reduce_max(red, local_max);
    float local_sum = 0.0f;
    for (int k = threadIdx.x; k < D; k += blockDim.x) local_sum += expf(row[k] - M);
    const float lse = M + logf(block_reduce_sum(red, local_sum));

    for (int k = threadIdx.x; k < D; k += blockDim.x) log_P_dur[dur_idx(T, U, D, b, t, u, k)] = row[k] - lse;
}

// ---- forward: alpha (frame-by-frame, variable-stride edges) ---------------
// grid = B, threads stride over u; the block walks t = 0..T. Per frame: phase 1
// gathers the d>=1 edges (parallel over u, reading finished earlier rows from
// global), then a serial d==0 token chain folds along u (thread 0). alpha lives
// in global on a (T+1) time grid; the terminal row is the exact-landing node.
__global__ void tdt_alpha(const float* __restrict__ log_P_blank,  // (B, T, U)
                          const float* __restrict__ log_P_y,      // (B, T, U)
                          const float* __restrict__ log_P_dur,    // (B, T, U, D)
                          const int* __restrict__ durations,      // (D,)
                          const int* __restrict__ in_lens,        // (B,)
                          const int* __restrict__ tgt_lens,       // (B,)
                          float* __restrict__ alpha,              // (B, T+1, U)
                          float* __restrict__ log_prob,           // (B,)
                          float* __restrict__ loss,               // (B,)
                          int T, int U, int D, int has_zero) {
    const int b = blockIdx.x;
    const int T_b = in_lens[b];
    const int U_b = tgt_lens[b] + 1;

    extern __shared__ float ext[];  // U entries: the current frame's alpha row being built
    float* sd = ext + U;            // U entries: scan multipliers for the d==0 token chain

    for (int t = 0; t <= T; ++t) {
        // phase 1: d>=1 edges arriving from strictly earlier frames (+ the start node).
        for (int u = threadIdx.x; u < U; u += blockDim.x) {
            float acc = (t == 0 && u == 0) ? 0.0f : NEG;
            for (int k = 0; k < D; ++k) {
                const int dd = durations[k];
                if (dd < 1) continue;
                const int s = t - dd;
                if (s < 0) continue;
                if (s >= T_b) continue;  // source must be a real frame
                // blank: (s, u) -> (t, u)
                if (u < U_b) {
                    const float a = alpha[alpha_idx(T, U, b, s, u)];
                    acc = logaddexpf(acc, a + log_P_blank[node3(T, U, b, s, u)] + log_P_dur[dur_idx(T, U, D, b, s, u, k)]);
                }
                // token: (s, u-1) -> (t, u), emitting targets[u-1] (validity baked into log_P_y)
                if (u >= 1 && u < U_b) {
                    const float a = alpha[alpha_idx(T, U, b, s, u - 1)];
                    acc = logaddexpf(
                        acc, a + log_P_y[node3(T, U, b, s, u - 1)] + log_P_dur[dur_idx(T, U, D, b, s, u - 1, k)]);
                }
            }
            ext[u] = acc;
        }
        __syncthreads();

        // phase 2: intra-frame d==0 token chain, (t,u-1) -> (t,u). It is a first-order
        // linear recurrence in the (logaddexp, +) semiring -- alpha[u] = f_u(alpha[u-1])
        // with f_u(x) = logaddexp(ext[u], x + w_{u-1}). Representing each f_u as a pair
        // (c=ext[u], d=w_{u-1}), composition is associative:
        //   (cA,dA) o (cB,dB) = (logaddexp(cA, cB + dA), dA + dB),
        // so an inclusive Hillis-Steele scan of those pairs gives alpha[u] = c_u in
        // O(log U) steps. Falls back to a serial fold when U exceeds the block (no 1:1
        // thread-per-u mapping). Only the prefix [0, U_b-1] is active (nodes u >= U_b take
        // no d==0 edge); the scan reads strictly leftward so it never touches u >= U_b.
        if (has_zero && t < T && t < T_b) {
            int zk = 0;
            for (int k = 0; k < D; ++k)
                if (durations[k] == 0) {
                    zk = k;
                    break;
                }
            const int L = U_b;  // active prefix length
            if (U <= blockDim.x) {
                const int u = threadIdx.x;
                // leaf multiplier d_u = w_{u-1} (edge u-1 -> u); leaf 0 has no predecessor.
                sd[u] = (u >= 1) ? (log_P_y[node3(T, U, b, t, u - 1)] + log_P_dur[dur_idx(T, U, D, b, t, u - 1, zk)])
                                 : NEG;
                __syncthreads();
                for (int o = 1; o < L; o <<= 1) {
                    float cc, dd2;
                    const bool dc = (u < L) && (u - o >= 0);
                    if (dc) {
                        const float cb = ext[u - o], db = sd[u - o];  // before-block [.. u-o]
                        const float ca = ext[u], da = sd[u];          // after-block  [u-o+1 .. u]
                        cc = logaddexpf(ca, cb + da);
                        dd2 = db + da;
                    }
                    __syncthreads();
                    if (dc) {
                        ext[u] = cc;
                        sd[u] = dd2;
                    }
                    __syncthreads();
                }
            } else if (threadIdx.x == 0) {
                for (int u = 1; u < L; ++u) {
                    const float w = log_P_y[node3(T, U, b, t, u - 1)] + log_P_dur[dur_idx(T, U, D, b, t, u - 1, zk)];
                    ext[u] = logaddexpf(ext[u], ext[u - 1] + w);
                }
            }
        }
        __syncthreads();

        for (int u = threadIdx.x; u < U; u += blockDim.x) alpha[alpha_idx(T, U, b, t, u)] = ext[u];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float lp = alpha[alpha_idx(T, U, b, T_b, U_b - 1)];  // exact-landing terminal
        log_prob[b] = lp;
        loss[b] = -lp;
    }
}

// ---- backward: beta on the fly -> slim posteriors + duration grad ---------
// grid = B, threads over u, reverse sweep t = T..0. beta is kept in a shared-memory
// ring of (max_duration+1) rows and never stored. As each source node is finalized,
// its edge posteriors are scattered into blank_post / token_post (token head) and
// grad_logp_dur (duration head; aggregates BOTH edge families since P_dur is shared).
__global__ void tdt_betapost(const float* __restrict__ log_P_blank,  // (B, T, U)
                             const float* __restrict__ log_P_y,      // (B, T, U)
                             const float* __restrict__ log_P_dur,    // (B, T, U, D)
                             const float* __restrict__ alpha,        // (B, T+1, U)
                             const float* __restrict__ log_prob,     // (B,)
                             const int* __restrict__ durations,      // (D,)
                             const int* __restrict__ in_lens,        // (B,)
                             const int* __restrict__ tgt_lens,       // (B,)
                             float* __restrict__ blank_post,         // (B, T, U)
                             float* __restrict__ token_post,         // (B, T, U)
                             float* __restrict__ grad_logp_dur,      // (B, T, U, D)
                             int T, int U, int D, int M, int has_zero) {
    const int b = blockIdx.x;
    const int T_b = in_lens[b];
    const int U_b = tgt_lens[b] + 1;
    const float logZ = log_prob[b];

    extern __shared__ float ring[];  // M rows of U entries: ring[m*U + u] = beta[frame with frame%M==m]
    float* sd = ring + (int64_t)M * U;  // U entries: scan multipliers for the d==0 token chain

    for (int idx = threadIdx.x; idx < M * U; idx += blockDim.x) ring[idx] = NEG;
    __syncthreads();

    for (int t = T; t >= 0; --t) {
        float* cur = ring + (t % M) * U;

        // phase 1: d>=1 edges leaving to a later frame (+ terminal seed). Parallel over u.
        for (int u = threadIdx.x; u < U; u += blockDim.x) {
            const bool is_term = (t == T_b) && (u == U_b - 1);
            float acc = NEG;
            if (t < T) {  // log_P_* only defined for source frames 0..T-1
                for (int k = 0; k < D; ++k) {
                    const int dd = durations[k];
                    if (dd < 1) continue;
                    const int nt = t + dd;
                    if (nt > T) continue;
                    if (t >= T_b || nt > T_b) continue;  // src real & land in-bounds
                    const float* nxt = ring + (nt % M) * U;
                    if (u < U_b) {  // blank: (t,u) -> (nt,u)
                        acc = logaddexpf(
                            acc, log_P_blank[node3(T, U, b, t, u)] + log_P_dur[dur_idx(T, U, D, b, t, u, k)] + nxt[u]);
                    }
                    if (u < U_b - 1) {  // token: (t,u) -> (nt,u+1)
                        acc = logaddexpf(
                            acc, log_P_y[node3(T, U, b, t, u)] + log_P_dur[dur_idx(T, U, D, b, t, u, k)] + nxt[u + 1]);
                    }
                }
            }
            cur[u] = is_term ? 0.0f : acc;
        }
        __syncthreads();

        // phase 2: intra-frame d==0 token edge, (t,u) -> (t,u+1). Mirror of the alpha
        // chain, scanned in the +u direction: beta[u] = logaddexp(cur[u], w_u + beta[u+1]).
        // Same associative pair composition, reversed; an inclusive right-to-left
        // Hillis-Steele scan gives beta[u] = c_u in O(log U) steps, serial fallback when
        // U exceeds the block. Active range [0, U_b-1]; the rightmost node has no token edge.
        if (has_zero && t < T && t < T_b) {
            int zk = 0;
            for (int k = 0; k < D; ++k)
                if (durations[k] == 0) {
                    zk = k;
                    break;
                }
            const int Lb = U_b;  // active range length
            if (U <= blockDim.x) {
                const int u = threadIdx.x;
                // leaf multiplier d_u = w_u (edge u -> u+1); leaf U_b-1 has no successor.
                sd[u] = (u < U_b - 1) ? (log_P_y[node3(T, U, b, t, u)] + log_P_dur[dur_idx(T, U, D, b, t, u, zk)]) : NEG;
                __syncthreads();
                for (int o = 1; o < Lb; o <<= 1) {
                    float cc, dd2;
                    const bool dc = (u < Lb) && (u + o < Lb);
                    if (dc) {
                        const float cR = cur[u + o], dR = sd[u + o];  // right-block [u+o ..]
                        const float cL = cur[u], dL = sd[u];          // left-block  [u .. u+o-1]
                        cc = logaddexpf(cL, cR + dL);
                        dd2 = dL + dR;
                    }
                    __syncthreads();
                    if (dc) {
                        cur[u] = cc;
                        sd[u] = dd2;
                    }
                    __syncthreads();
                }
            } else if (threadIdx.x == 0) {
                for (int u = U - 2; u >= 0; --u) {
                    if (u >= U_b - 1) continue;  // token edge needs u < U_b-1
                    const float w = log_P_y[node3(T, U, b, t, u)] + log_P_dur[dur_idx(T, U, D, b, t, u, zk)];
                    cur[u] = logaddexpf(cur[u], w + cur[u + 1]);
                }
            }
        }
        __syncthreads();

        // phase 3: edge posteriors for source nodes at this frame (t < T only).
        if (t < T) {
            for (int u = threadIdx.x; u < U; u += blockDim.x) {
                const float a = alpha[alpha_idx(T, U, b, t, u)];
                float bp = 0.0f, tp = 0.0f;
                for (int k = 0; k < D; ++k) {
                    const int dd = durations[k];
                    const int nt = t + dd;
                    float gblank = 0.0f, gtoken = 0.0f;
                    if (nt <= T && t < T_b && nt <= T_b) {
                        const float* nxt = ring + (nt % M) * U;
                        const float lpdur = log_P_dur[dur_idx(T, U, D, b, t, u, k)];
                        if (dd >= 1 && u < U_b) {
                            gblank = expf(a + log_P_blank[node3(T, U, b, t, u)] + lpdur + nxt[u] - logZ);
                        }
                        if (u < U_b - 1) {
                            gtoken = expf(a + log_P_y[node3(T, U, b, t, u)] + lpdur + nxt[u + 1] - logZ);
                        }
                    }
                    bp += gblank;
                    tp += gtoken;
                    grad_logp_dur[dur_idx(T, U, D, b, t, u, k)] = -(gblank + gtoken);
                }
                blank_post[node3(T, U, b, t, u)] = bp;
                token_post[node3(T, U, b, t, u)] = tp;
            }
        }
        __syncthreads();
    }
}

// d loss / d (token logit), written in place over the token-logits buffer.
//   glogit[v] = grad_loss * (softmax[v]*(bp+tp) - bp*[v==blank] - tp*[v==y]).
__global__ void tdt_build_glogit_tok(float* __restrict__ logits,            // (Cn, V) in -> glogit out
                                     const float* __restrict__ blank_post,  // (B, T, U)
                                     const float* __restrict__ token_post,  // (B, T, U)
                                     const float* __restrict__ grad_loss,   // (B,)
                                     const int* __restrict__ targets,       // (B, S)
                                     const int* __restrict__ tgt_lens,      // (B,)
                                     int T, int U, int V, int S, int blank, int c0, int Cn) {
    const int i = blockIdx.x;
    if (i >= Cn) return;
    int b, t, u;
    decode_node(c0 + i, T, U, b, t, u);
    float* row = logits + (int64_t)i * V;

    const float bp = blank_post[node3(T, U, b, t, u)];
    const float tp = token_post[node3(T, U, b, t, u)];
    if (bp == 0.0f && tp == 0.0f) {  // padded / zero-posterior node contributes nothing
        for (int v = threadIdx.x; v < V; v += blockDim.x) row[v] = 0.0f;
        return;
    }
    const float P = bp + tp;
    const float gl = grad_loss[b];
    const int y = (u < tgt_lens[b]) ? targets[tgt_idx(S, b, u)] : -1;

    extern __shared__ float red[];
    float local_max = -INFINITY;
    for (int v = threadIdx.x; v < V; v += blockDim.x) local_max = fmaxf(local_max, row[v]);
    const float Mx = block_reduce_max(red, local_max);
    float local_sum = 0.0f;
    for (int v = threadIdx.x; v < V; v += blockDim.x) local_sum += expf(row[v] - Mx);
    const float lse = Mx + logf(block_reduce_sum(red, local_sum));

    for (int v = threadIdx.x; v < V; v += blockDim.x) {
        float g = gl * expf(row[v] - lse) * P;
        if (v == blank) g -= gl * bp;
        if (v == y) g -= gl * tp;
        row[v] = g;
    }
}

// d loss / d (duration logit), written in place over the duration-logits buffer.
// grad_logp_dur already holds d loss / d logp_dur (upstream 1); push through the
// duration log-softmax Jacobian and scale by the per-utterance upstream grad.
__global__ void tdt_build_glogit_dur(float* __restrict__ logits,               // (Cn, D) in -> glogit out
                                     const float* __restrict__ grad_logp_dur,  // (B, T, U, D)
                                     const float* __restrict__ grad_loss,      // (B,)
                                     int T, int U, int D, int c0, int Cn) {
    const int i = blockIdx.x;
    if (i >= Cn) return;
    int b, t, u;
    decode_node(c0 + i, T, U, b, t, u);
    float* row = logits + (int64_t)i * D;
    const float* g_in = grad_logp_dur + dur_idx(T, U, D, b, t, u, 0);
    const float gl = grad_loss[b];

    extern __shared__ float red[];
    float local_max = -INFINITY;
    for (int k = threadIdx.x; k < D; k += blockDim.x) local_max = fmaxf(local_max, row[k]);
    const float Mx = block_reduce_max(red, local_max);
    float local_sum = 0.0f, local_gsum = 0.0f;
    for (int k = threadIdx.x; k < D; k += blockDim.x) {
        local_sum += expf(row[k] - Mx);
        local_gsum += g_in[k];
    }
    const float lse = Mx + logf(block_reduce_sum(red, local_sum));
    const float gsum = block_reduce_sum(red, local_gsum);

    for (int k = threadIdx.x; k < D; k += blockDim.x) row[k] = gl * (g_in[k] - expf(row[k] - lse) * gsum);
}

// grad_pre = grad_hidden * (1 - hidden^2), scattered to encoder/decoder (low-contention atomics).
__global__ void tdt_scatter_act(const float* __restrict__ grad_hidden,  // (Cn, d)
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

// ---- host wrappers -------------------------------------------------------
std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor,
           torch::stable::Tensor, torch::stable::Tensor>
tdt_forward_cuda(const torch::stable::Tensor& encoder, const torch::stable::Tensor& decoder,
                 const torch::stable::Tensor& joint_W, const torch::stable::Tensor& joint_W_dur,
                 const torch::stable::Tensor& targets, const torch::stable::Tensor& in_lens,
                 const torch::stable::Tensor& tgt_lens, const torch::stable::Tensor& durations, int64_t max_duration,
                 int64_t has_zero, int64_t blank_idx, int64_t tf32) {
    CHECK_F32_INPUT(encoder);
    CHECK_F32_INPUT(decoder);
    CHECK_F32_INPUT(joint_W);
    CHECK_F32_INPUT(joint_W_dur);
    CHECK_INT_INPUT(targets);
    CHECK_INT_INPUT(in_lens);
    CHECK_INT_INPUT(tgt_lens);
    CHECK_INT_INPUT(durations);

    const int B = encoder.sizes()[0];
    const int T = encoder.sizes()[1];
    const int d = encoder.sizes()[2];
    const int U = decoder.sizes()[1];
    const int V = joint_W.sizes()[0];
    const int Dn = joint_W_dur.sizes()[0];
    const int S = targets.sizes()[1];
    const int N = B * T * U;

    auto log_P_blank = torch::stable::new_empty(encoder, {B, T, U});
    auto log_P_y = torch::stable::new_empty(encoder, {B, T, U});
    auto log_P_dur = torch::stable::new_empty(encoder, {B, T, U, Dn});
    auto alpha = torch::stable::new_empty(encoder, {B, T + 1, U});
    auto log_prob = torch::stable::new_empty(encoder, {B});
    auto loss = torch::stable::new_empty(encoder, {B});

    cudaStream_t stream = get_stream(encoder);
    cublasHandle_t handle = get_cublas();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    set_math_mode(handle, tf32 != 0);

    const int chunk = std::min(N, NODE_CHUNK);
    auto hidden = torch::stable::new_empty(encoder, {chunk, d});
    auto logits_tok = torch::stable::new_empty(encoder, {chunk, V});
    auto logits_dur = torch::stable::new_empty(encoder, {chunk, Dn});
    const float one = 1.0f, zero = 0.0f;
    const size_t red_smem = NODE_THREADS * sizeof(float);

    for (int c0 = 0; c0 < N; c0 += chunk) {
        const int Cn = std::min(chunk, N - c0);
        const int hid_blocks = (int)(((int64_t)Cn * d + EW_THREADS - 1) / EW_THREADS);
        tdt_build_hidden<<<hid_blocks, EW_THREADS, 0, stream>>>(encoder.const_data_ptr<float>(),
                                                                decoder.const_data_ptr<float>(),
                                                                hidden.mutable_data_ptr<float>(), T, U, d, c0, Cn);
        // logits_tok (Cn, V) = hidden (Cn, d) @ joint_W^T (d, V)   [column-major: L_cm = Wᵀ · H]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, V, Cn, d, &one, joint_W.const_data_ptr<float>(), d,
                                 hidden.const_data_ptr<float>(), d, &zero, logits_tok.mutable_data_ptr<float>(), V));
        // logits_dur (Cn, D) = hidden (Cn, d) @ joint_W_dur^T (d, D)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Dn, Cn, d, &one, joint_W_dur.const_data_ptr<float>(),
                                 d, hidden.const_data_ptr<float>(), d, &zero, logits_dur.mutable_data_ptr<float>(), Dn));
        tdt_logsoftmax_tok<<<Cn, NODE_THREADS, red_smem, stream>>>(
            logits_tok.const_data_ptr<float>(), targets.const_data_ptr<int>(), tgt_lens.const_data_ptr<int>(),
            log_P_blank.mutable_data_ptr<float>(), log_P_y.mutable_data_ptr<float>(), T, U, V, S, (int)blank_idx, c0,
            Cn);
        tdt_logsoftmax_dur<<<Cn, NODE_THREADS, red_smem, stream>>>(
            logits_dur.const_data_ptr<float>(), log_P_dur.mutable_data_ptr<float>(), T, U, Dn, c0, Cn);
    }

    // One thread per label index enables the O(log U) d==0 scan (cap at 1024; larger U
    // falls back to the in-kernel serial fold). Shared: alpha row + scan multipliers.
    const int alpha_threads = std::min(U, 1024);
    const size_t alpha_smem = (size_t)2 * U * sizeof(float);
    tdt_alpha<<<B, alpha_threads, alpha_smem, stream>>>(
        log_P_blank.const_data_ptr<float>(), log_P_y.const_data_ptr<float>(), log_P_dur.const_data_ptr<float>(),
        durations.const_data_ptr<int>(), in_lens.const_data_ptr<int>(), tgt_lens.const_data_ptr<int>(),
        alpha.mutable_data_ptr<float>(), log_prob.mutable_data_ptr<float>(), loss.mutable_data_ptr<float>(), T, U, Dn,
        (int)has_zero);

    return {std::move(loss),      std::move(alpha),    std::move(log_P_blank),
            std::move(log_P_y),   std::move(log_P_dur), std::move(log_prob)};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor, torch::stable::Tensor>
tdt_backward_cuda(const torch::stable::Tensor& encoder, const torch::stable::Tensor& decoder,
                  const torch::stable::Tensor& joint_W, const torch::stable::Tensor& joint_W_dur,
                  const torch::stable::Tensor& targets, const torch::stable::Tensor& in_lens,
                  const torch::stable::Tensor& tgt_lens, const torch::stable::Tensor& durations, int64_t max_duration,
                  int64_t has_zero, int64_t blank_idx, const torch::stable::Tensor& alpha,
                  const torch::stable::Tensor& log_P_blank, const torch::stable::Tensor& log_P_y,
                  const torch::stable::Tensor& log_P_dur, const torch::stable::Tensor& log_prob,
                  const torch::stable::Tensor& grad_loss, int64_t tf32) {
    CHECK_F32_INPUT(encoder);
    CHECK_F32_INPUT(decoder);
    CHECK_F32_INPUT(joint_W);
    CHECK_F32_INPUT(joint_W_dur);
    CHECK_INT_INPUT(targets);
    CHECK_INT_INPUT(in_lens);
    CHECK_INT_INPUT(tgt_lens);
    CHECK_INT_INPUT(durations);
    CHECK_F32_INPUT(alpha);
    CHECK_F32_INPUT(log_P_blank);
    CHECK_F32_INPUT(log_P_y);
    CHECK_F32_INPUT(log_P_dur);
    CHECK_F32_INPUT(log_prob);
    CHECK_F32_INPUT(grad_loss);

    const int B = encoder.sizes()[0];
    const int T = encoder.sizes()[1];
    const int d = encoder.sizes()[2];
    const int U = decoder.sizes()[1];
    const int V = joint_W.sizes()[0];
    const int Dn = joint_W_dur.sizes()[0];
    const int S = targets.sizes()[1];
    const int N = B * T * U;

    auto blank_post = torch::stable::new_empty(encoder, {B, T, U});
    auto token_post = torch::stable::new_empty(encoder, {B, T, U});
    auto grad_logp_dur = torch::stable::new_empty(encoder, {B, T, U, Dn});
    auto grad_encoder = torch::stable::new_zeros(encoder, {B, T, d});
    auto grad_decoder = torch::stable::new_zeros(encoder, {B, U, d});
    auto grad_joint_W = torch::stable::new_zeros(joint_W, {V, d});
    auto grad_joint_W_dur = torch::stable::new_zeros(joint_W_dur, {Dn, d});

    cudaStream_t stream = get_stream(encoder);
    cublasHandle_t handle = get_cublas();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    set_math_mode(handle, tf32 != 0);

    const int M = (int)max_duration + 1;  // beta ring rows
    const int beta_threads = std::min(U, 1024);
    const size_t beta_smem = (size_t)(M + 1) * U * sizeof(float);  // ring + scan multipliers
    tdt_betapost<<<B, beta_threads, beta_smem, stream>>>(
        log_P_blank.const_data_ptr<float>(), log_P_y.const_data_ptr<float>(), log_P_dur.const_data_ptr<float>(),
        alpha.const_data_ptr<float>(), log_prob.const_data_ptr<float>(), durations.const_data_ptr<int>(),
        in_lens.const_data_ptr<int>(), tgt_lens.const_data_ptr<int>(), blank_post.mutable_data_ptr<float>(),
        token_post.mutable_data_ptr<float>(), grad_logp_dur.mutable_data_ptr<float>(), T, U, Dn, M, (int)has_zero);

    const int chunk = std::min(N, NODE_CHUNK);
    auto hidden = torch::stable::new_empty(encoder, {chunk, d});
    auto glogit_tok = torch::stable::new_empty(encoder, {chunk, V});
    auto glogit_dur = torch::stable::new_empty(encoder, {chunk, Dn});
    auto grad_hidden = torch::stable::new_empty(encoder, {chunk, d});
    const float one = 1.0f, zero = 0.0f;
    const size_t red_smem = NODE_THREADS * sizeof(float);

    for (int c0 = 0; c0 < N; c0 += chunk) {
        const int Cn = std::min(chunk, N - c0);
        const int act_blocks = (int)(((int64_t)Cn * d + EW_THREADS - 1) / EW_THREADS);
        tdt_build_hidden<<<act_blocks, EW_THREADS, 0, stream>>>(encoder.const_data_ptr<float>(),
                                                                decoder.const_data_ptr<float>(),
                                                                hidden.mutable_data_ptr<float>(), T, U, d, c0, Cn);
        // recompute both logit heads (never stored across chunks)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, V, Cn, d, &one, joint_W.const_data_ptr<float>(), d,
                                 hidden.const_data_ptr<float>(), d, &zero, glogit_tok.mutable_data_ptr<float>(), V));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Dn, Cn, d, &one, joint_W_dur.const_data_ptr<float>(),
                                 d, hidden.const_data_ptr<float>(), d, &zero, glogit_dur.mutable_data_ptr<float>(), Dn));
        tdt_build_glogit_tok<<<Cn, NODE_THREADS, red_smem, stream>>>(
            glogit_tok.mutable_data_ptr<float>(), blank_post.const_data_ptr<float>(), token_post.const_data_ptr<float>(),
            grad_loss.const_data_ptr<float>(), targets.const_data_ptr<int>(), tgt_lens.const_data_ptr<int>(), T, U, V, S,
            (int)blank_idx, c0, Cn);
        tdt_build_glogit_dur<<<Cn, NODE_THREADS, red_smem, stream>>>(
            glogit_dur.mutable_data_ptr<float>(), grad_logp_dur.const_data_ptr<float>(),
            grad_loss.const_data_ptr<float>(), T, U, Dn, c0, Cn);
        // grad_joint_W += glogit_tokᵀ · hidden ; grad_joint_W_dur += glogit_durᵀ · hidden  (accumulate)
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, V, Cn, &one, hidden.const_data_ptr<float>(), d,
                                 glogit_tok.const_data_ptr<float>(), V, &one, grad_joint_W.mutable_data_ptr<float>(),
                                 d));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, Dn, Cn, &one, hidden.const_data_ptr<float>(), d,
                                 glogit_dur.const_data_ptr<float>(), Dn, &one,
                                 grad_joint_W_dur.mutable_data_ptr<float>(), d));
        // grad_hidden (Cn, d) = glogit_tok @ joint_W ; += glogit_dur @ joint_W_dur
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, Cn, V, &one, joint_W.const_data_ptr<float>(), d,
                                 glogit_tok.const_data_ptr<float>(), V, &zero, grad_hidden.mutable_data_ptr<float>(),
                                 d));
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, Cn, Dn, &one, joint_W_dur.const_data_ptr<float>(),
                                 d, glogit_dur.const_data_ptr<float>(), Dn, &one, grad_hidden.mutable_data_ptr<float>(),
                                 d));
        tdt_scatter_act<<<act_blocks, EW_THREADS, 0, stream>>>(
            grad_hidden.const_data_ptr<float>(), hidden.const_data_ptr<float>(), grad_encoder.mutable_data_ptr<float>(),
            grad_decoder.mutable_data_ptr<float>(), T, U, d, c0, Cn);
    }

    return {std::move(grad_encoder), std::move(grad_decoder), std::move(grad_joint_W), std::move(grad_joint_W_dur)};
}
