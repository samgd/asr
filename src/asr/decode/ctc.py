import torch
from jaxtyping import Float, Integer
from torch.utils.cpp_extension import load_inline


def greedy_decode(
    log_probs: Float[torch.Tensor, "batch time vocab"], in_lens: Integer[torch.Tensor, "batch"]
) -> list[list[int]]:
    """Greedy decode each batch entry to a list of token IDs."""
    blank_id = 0

    out = []
    for b in range(log_probs.shape[0]):
        in_len = in_lens[b]
        tokens = log_probs[b].argmax(dim=-1)[:in_len]

        # keep first token and those that differ from previous
        keep = torch.ones_like(tokens, dtype=torch.bool)
        keep[1:] = tokens[1:] != tokens[:-1]
        collapsed = tokens[keep]

        # remove blanks
        collapsed = collapsed[collapsed != blank_id]

        out.append(collapsed.tolist())
    return out


cpp_source = r"""
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

constexpr float ZERO = -std::numeric_limits<float>::infinity();
constexpr float ONE  = 0.0f;

class TrieNode {
public:
    TrieNode* parent;
    int token_id;
    std::unordered_map<int, std::unique_ptr<TrieNode>> children;

    TrieNode(TrieNode* p, int tok) : parent(p), token_id(tok) {}
};

inline float logaddexp(float a, float b) {
    if (a == ZERO) return b;
    if (b == ZERO) return a;
    float hi = std::max(a, b);
    float lo = std::min(a, b);
    return hi + std::log1p(std::exp(lo - hi));
}

class Entry {
public:
    TrieNode* pos;
    float log_p_blank;
    float log_p_nonblank;

    float total_log_p() const {
        return logaddexp(log_p_blank, log_p_nonblank);
    }
};

std::vector<torch::Tensor> beam_decode(
    torch::Tensor log_probs, // (B, T, V)
    torch::Tensor in_lens,   // (B,)
    int beam_width)
{
    int B = log_probs.size(0);
    int V = log_probs.size(2);

    std::vector<torch::Tensor> out(B);

    auto lp = log_probs.accessor<float, 3>();
    auto il = in_lens.accessor<int64_t, 1>();

    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < B; ++b) {
        TrieNode root(nullptr, -1);

        std::vector<Entry> beam;
        std::vector<Entry> next_beam;
        std::unordered_map<TrieNode*, int> next_beam_id;

        beam.push_back(Entry{&root, ONE, ZERO});

        // index-based helpers avoid invalidating references when next_beam grows
        auto get_or_create_next = [&](TrieNode* node) -> int {
            auto it = next_beam_id.find(node);
            if (it != next_beam_id.end()) {
                return it->second;
            }
            int idx = static_cast<int>(next_beam.size());
            next_beam_id[node] = idx;
            next_beam.push_back(Entry{node, ZERO, ZERO});
            return idx;
        };

        auto find_or_create_child = [](TrieNode* parent, int v) -> TrieNode* {
            auto [it, _] = parent->children.try_emplace(v, std::make_unique<TrieNode>(parent, v));
            return it->second.get();
        };

        for (int t = 0; t < static_cast<int>(il[b]); ++t) {
            for (auto& entry : beam) {
                float entry_total = entry.total_log_p();
                for (int v = 0; v < V; ++v) {
                    float log_p = lp[b][t][v];

                    if (v == 0) {
                        // blank: stay at entry.pos, accumulate into log_p_blank from both prior states
                        int idx = get_or_create_next(entry.pos);
                        next_beam[idx].log_p_blank = logaddexp(next_beam[idx].log_p_blank, entry_total + log_p);
                    } else if (v == entry.pos->token_id) {
                        // repeat: stay at entry.pos, from log_p_nonblank only
                        int stay_idx = get_or_create_next(entry.pos);
                        next_beam[stay_idx].log_p_nonblank = logaddexp(next_beam[stay_idx].log_p_nonblank, entry.log_p_nonblank + log_p);

                        // extend to child, from log_p_blank only
                        TrieNode* child = find_or_create_child(entry.pos, v);
                        int ext_idx = get_or_create_next(child);
                        next_beam[ext_idx].log_p_nonblank = logaddexp(next_beam[ext_idx].log_p_nonblank, entry.log_p_blank + log_p);
                    } else {
                        // new token: extend, accumulate from both prior states
                        TrieNode* child = find_or_create_child(entry.pos, v);
                        int idx = get_or_create_next(child);
                        next_beam[idx].log_p_nonblank = logaddexp(next_beam[idx].log_p_nonblank, entry_total + log_p);
                    }
                }
            }

            size_t keep = std::min<size_t>(beam_width, next_beam.size());
            std::nth_element(next_beam.begin(), next_beam.begin() + keep, next_beam.end(),
                [](const Entry& a, const Entry& b) { return a.total_log_p() > b.total_log_p(); });
            next_beam.resize(keep);

            std::swap(beam, next_beam);
            next_beam.clear();
            next_beam_id.clear();
        }

        // argmax across the surviving beam, backtrack through parent pointers
        auto best = std::max_element(beam.begin(), beam.end(), [](const Entry& a, const Entry& b) { return a.total_log_p() < b.total_log_p(); });
        std::vector<int64_t> tokens;
        for (TrieNode* n = best->pos; n != nullptr && n->parent != nullptr; n = n->parent) {
            tokens.push_back(n->token_id);
        }
        std::reverse(tokens.begin(), tokens.end());
        out[b] = torch::tensor(tokens, torch::kInt64);
    }
    return out;
}
"""

ctc_ext = load_inline(
    name="ctc_ext",
    cpp_sources=cpp_source,
    functions=["beam_decode"],
    extra_cflags=["-O3", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=True,
)


def beam_decode(
    log_probs: Float[torch.Tensor, "batch time vocab"],
    in_lens: Integer[torch.Tensor, "batch"],
    beam_width: int = 32,
) -> list[list[int]]:
    log_probs = log_probs.cpu().contiguous()
    in_lens = in_lens.cpu().contiguous()
    return [o.tolist() for o in ctc_ext.beam_decode(log_probs, in_lens, beam_width)]
