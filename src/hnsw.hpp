#pragma once
// HNSW approximate nearest-neighbour index (Malkov & Yashunin, 2018), written
// from scratch.
//
// The index is a stack of proximity graphs. Layer 0 holds every vector; each
// higher layer holds an exponentially smaller random sample. A query starts at
// the top, walks greedily towards the target, drops a layer, and repeats;
// on layer 0 it runs a best-first search that keeps the `ef` best candidates.
//
// Memory layout:
//   vectors      one aligned block, max_elements * dim floats
//   layer-0 links one block, max_elements * (1 + 2M) uint32, [count, n1, n2, ...]
//   upper links  allocated per node, only for the ~1/M of nodes that have any
// so the hot loop (read a node's neighbours, compute distances) touches two
// contiguous arrays and nothing else.
//
// Storage is either float32 or, with scalar quantisation (SQ8), one byte per
// dimension: a quarter of the memory, with distances computed directly on the
// codes. Optionally the original floats are kept alongside to re-rank the
// final candidates exactly.
//
// Threading: add() may be called from many threads at once, and search() may
// be called from many threads at once, but not add() and search() together.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "distance.hpp"

namespace vecdb {

enum class Metric : std::uint32_t { L2 = 0, InnerProduct = 1, Cosine = 2 };

// How a filtered search is answered. Auto picks by estimated cost.
enum class FilterStrategy : std::uint32_t { Auto = 0, Graph = 1, Scan = 2 };

// One byte per node. A std::mutex is 64 bytes on macOS, which for a million
// nodes would cost more memory than the graph's upper layers.
class SpinLock {
public:
    void lock() {
        while (flag_.test_and_set(std::memory_order_acquire))
            while (flag_.test(std::memory_order_relaxed)) {}
    }
    void unlock() { flag_.clear(std::memory_order_release); }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

struct Neighbor {
    float dist;
    std::uint32_t id;
    bool operator<(const Neighbor& o) const { return dist < o.dist; }
    bool operator>(const Neighbor& o) const { return dist > o.dist; }
    bool operator==(const Neighbor& o) const { return dist == o.dist && id == o.id; }
};

class HnswIndex {
public:
    using Label = std::uint64_t;

    // sq8: store one byte per dimension instead of a float.
    // keep_full: with sq8, also keep the floats and use them to re-rank results.
    HnswIndex(std::size_t dim, std::size_t max_elements, Metric metric = Metric::L2, std::size_t M = 16,
              std::size_t ef_construction = 200, std::uint64_t seed = 100, bool sq8 = false, bool keep_full = false)
        : dim_(dim), max_(max_elements), metric_(metric), M_(M), M0_(2 * M), efc_(std::max(ef_construction, M)),
          level_mult_(1.0 / std::log(static_cast<double>(M))), sq8_(sq8), keep_full_(sq8 && keep_full),
          vec_bytes_(sq8 ? dim : dim * sizeof(float)), rng_(seed) {
        if (dim == 0 || max_elements == 0 || M < 2) throw std::invalid_argument("bad index parameters");
        if (sq8 && metric == Metric::InnerProduct)
            throw std::invalid_argument("sq8 supports l2 and cosine (inner product needs unit-length vectors: use cosine)");
        allocate();
    }

    ~HnswIndex() {
        std::free(data_);
        std::free(full_);
        std::free(links0_);
    }
    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

    std::size_t dim() const { return dim_; }
    std::size_t size() const { return count_.load(std::memory_order_acquire); }
    std::size_t capacity() const { return max_; }
    std::size_t deleted() const { return n_deleted_; }
    std::size_t M() const { return M_; }
    Metric metric() const { return metric_; }
    int max_level() const { return max_level_; }
    bool quantized() const { return sq8_; }
    bool trained() const { return !sq8_ || trained_; }

    // Choose the value range the 256 code levels will cover, from a sample of
    // the data. The range runs between the 0.1% and 99.9% quantiles: clipping
    // the rare extreme value buys finer steps for everything else.
    void train(const float* sample, std::size_t n) {
        if (!sq8_ || trained_) return;
        std::vector<float> vals;
        const std::size_t total = n * dim_, want = std::min<std::size_t>(total, 2'000'000);
        vals.reserve(want);
        std::vector<float> tmp(dim_);
        const std::size_t rows = std::max<std::size_t>(1, want / dim_), step = std::max<std::size_t>(1, n / rows);
        for (std::size_t r = 0; r < n; r += step) {
            std::memcpy(tmp.data(), sample + r * dim_, dim_ * sizeof(float));
            if (metric_ == Metric::Cosine) normalize(tmp.data());
            vals.insert(vals.end(), tmp.begin(), tmp.end());
        }
        const std::size_t lo = vals.size() / 1000, hi = vals.size() - 1 - vals.size() / 1000;
        std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(lo), vals.end());
        sq_min_ = vals[lo];
        std::nth_element(vals.begin(), vals.begin() + static_cast<std::ptrdiff_t>(hi), vals.end());
        const float max = vals[hi];
        sq_step_ = max > sq_min_ ? (max - sq_min_) / 255.0f : 1.0f;
        trained_ = true;
    }

    // Bytes held by the index, including the vectors themselves.
    std::size_t memory_bytes() const {
        std::size_t upper = 0;
        const std::size_t n = size();
        for (std::size_t i = 0; i < n; ++i) upper += static_cast<std::size_t>(levels_[i]) * (M_ + 1) * 4;
        return vector_bytes() + n * (M0_ + 1) * 4 + upper + n * (sizeof(Label) + 1 + 1 + sizeof(void*));
    }

    // Bytes spent on the vectors alone.
    std::size_t vector_bytes() const {
        return size() * (vec_bytes_ + (keep_full_ ? dim_ * sizeof(float) : 0));
    }

    // ------------------------------------------------------------- insert

    void add(const float* v, Label label) {
        std::uint32_t id;
        int level;
        {
            std::lock_guard<std::mutex> g(meta_mu_);
            if (label_to_id_.count(label)) throw std::invalid_argument("label already present");
            if (sq8_ && !trained_) throw std::logic_error("call train() before adding to an sq8 index");
            const std::uint32_t n = count_.load(std::memory_order_relaxed);
            if (n >= max_) throw std::length_error("index is full");
            id = n;
            label_to_id_.emplace(label, id);
            std::uniform_real_distribution<double> u(std::numeric_limits<double>::min(), 1.0);
            level = std::min(static_cast<int>(-std::log(u(rng_)) * level_mult_), 255);
            labels_[id] = label;
            levels_[id] = static_cast<std::uint8_t>(level);
            store(id, v);
            if (level > 0) upper_[id].reset(new std::uint32_t[static_cast<std::size_t>(level) * (M_ + 1)]());
            count_.store(n + 1, std::memory_order_release);
        }
        const unsigned char* q = vec(id);

        // Changing the entry point is rare (probability ~1/M per level), so a
        // node that raises the top level simply holds the global lock throughout.
        std::unique_lock<std::mutex> top(top_mu_);
        const int top_level = max_level_;
        const std::uint32_t entry = entry_;
        if (entry == kNone) {
            entry_ = id;
            max_level_ = level;
            return;
        }
        if (level <= top_level) top.unlock();

        std::uint32_t cur = entry;
        float cur_d = dist(q, vec(cur));
        std::uint32_t buf[kMaxLinks];
        for (int l = top_level; l > level; --l) {
            for (bool moved = true; moved;) {
                moved = false;
                const std::uint32_t n = copy_links(cur, l, buf);
                for (std::uint32_t i = 0; i < n; ++i) {
                    const float d = dist(q, vec(buf[i]));
                    if (d < cur_d) {
                        cur_d = d;
                        cur = buf[i];
                        moved = true;
                    }
                }
            }
        }

        std::vector<Neighbor> cand, chosen;
        for (int l = std::min(level, top_level); l >= 0; --l) {
            search_layer(q, cur, l, efc_, /*locked=*/true, nullptr, cand);
            // Another thread may already have linked to this node on a higher
            // layer, which makes it reachable from here. It is not its own neighbour.
            cand.erase(std::remove_if(cand.begin(), cand.end(), [id](const Neighbor& n) { return n.id == id; }),
                       cand.end());
            select_neighbors(cand, M_, chosen);
            if (chosen.empty()) continue;
            cur = chosen.front().id;

            {
                // Other threads may have added back-links to this node before
                // we got here; merge rather than overwrite.
                std::lock_guard<SpinLock> g(locks_[id]);
                std::uint32_t* mine = links(id, l);
                if (mine[0] != 0) {
                    for (std::uint32_t i = 1; i <= mine[0]; ++i) {
                        const std::uint32_t o = mine[i];
                        if (std::none_of(chosen.begin(), chosen.end(), [o](const Neighbor& n) { return n.id == o; }))
                            chosen.push_back({dist(q, vec(o)), o});
                    }
                    std::sort(chosen.begin(), chosen.end());
                    if (chosen.size() > max_links(l)) {
                        cand = chosen;
                        select_neighbors(cand, max_links(l), chosen);
                    }
                }
                mine[0] = static_cast<std::uint32_t>(chosen.size());
                for (std::size_t i = 0; i < chosen.size(); ++i) mine[i + 1] = chosen[i].id;
            }
            for (const Neighbor& nb : chosen) connect(nb.id, id, nb.dist, l);
        }

        if (level > top_level) {
            entry_ = id;
            max_level_ = level;
        }
    }

    // ------------------------------------------------------------- search

    // Up to k nearest live neighbours, closest first. `ef` is the size of the
    // candidate list on layer 0: larger is more accurate and slower.
    // `allow`, if given, is one byte per internal id; only non-zero ids may be returned.
    // `allow_ids`, if given, lists the same allowed ids; it lets a very
    // selective filter be answered by scanning just those vectors, which is
    // both exact and cheaper than wandering the graph looking for them.
    std::vector<std::pair<float, Label>> search(const float* query, std::size_t k, std::size_t ef,
                                                const std::uint8_t* allow = nullptr,
                                                const std::uint32_t* allow_ids = nullptr, std::size_t n_allow = 0,
                                                FilterStrategy strategy = FilterStrategy::Auto) const {
        std::vector<std::pair<float, Label>> out;
        const std::uint32_t entry = entry_;
        if (entry == kNone || k == 0) return out;
        ef = std::max(ef, k);

        Query q(*this, query);
        std::vector<Neighbor> found;
        if (allow_ids && use_scan(strategy, n_allow, ef)) {
            // Scan: one distance per allowed vector, and no approximation from the graph.
            std::priority_queue<Neighbor> heap;
            const std::size_t keep = sq8_ && keep_full_ ? ef : k;
            for (std::size_t i = 0; i < n_allow; ++i) {
                const std::uint32_t id = allow_ids[i];
                if (deleted_[id]) continue;
                const float d = dist(q.stored, vec(id));
                if (heap.size() < keep) heap.push({d, id});
                else if (d < heap.top().dist) {
                    heap.pop();
                    heap.push({d, id});
                }
            }
            found.resize(heap.size());
            for (std::size_t i = heap.size(); i-- > 0;) {
                found[i] = heap.top();
                heap.pop();
            }
        } else {
            std::uint32_t cur = entry;
            float cur_d = dist(q.stored, vec(cur));
            for (int l = max_level_; l > 0; --l) {
                for (bool moved = true; moved;) {
                    moved = false;
                    const std::uint32_t* ls = links(cur, l);
                    for (std::uint32_t i = 1; i <= ls[0]; ++i) {
                        const float d = dist(q.stored, vec(ls[i]));
                        if (d < cur_d) {
                            cur_d = d;
                            cur = ls[i];
                            moved = true;
                        }
                    }
                }
            }
            search_layer(q.stored, cur, 0, ef, /*locked=*/false, allow, found);
        }

        if (sq8_ && keep_full_) {
            // Re-rank every candidate with the original floats.
            for (Neighbor& n : found) n.dist = exact(q.floats, full(n.id));
            std::sort(found.begin(), found.end());
        }
        const std::size_t n = std::min(k, found.size());
        out.reserve(n);
        const bool approximate = sq8_ && !keep_full_;
        for (std::size_t i = 0; i < n; ++i)
            out.emplace_back(approximate ? code_to_metric(found[i].dist) : found[i].dist, labels_[found[i].id]);
        return out;
    }

    // Exact search over every live vector. The ground truth for recall.
    // Uses the original floats when the index has them; a codes-only sq8
    // index can only scan its codes, which is not ground truth.
    std::vector<std::pair<float, Label>> brute_force(const float* query, std::size_t k) const {
        Query q(*this, query);
        const bool floats = !sq8_ || keep_full_;
        std::priority_queue<Neighbor> heap;
        const std::uint32_t n = static_cast<std::uint32_t>(size());
        for (std::uint32_t i = 0; i < n; ++i) {
            if (deleted_[i]) continue;
            const float d = !sq8_ ? dist(q.stored, vec(i)) : floats ? exact(q.floats, full(i)) : dist(q.stored, vec(i));
            if (heap.size() < k) heap.push({d, i});
            else if (d < heap.top().dist) {
                heap.pop();
                heap.push({d, i});
            }
        }
        std::vector<std::pair<float, Label>> out(heap.size());
        for (std::size_t i = heap.size(); i-- > 0;) {
            out[i] = {floats ? heap.top().dist : code_to_metric(heap.top().dist), labels_[heap.top().id]};
            heap.pop();
        }
        return out;
    }

    // Tombstone: the node stays in the graph as a stepping stone but is never
    // returned. (Unlinking it would leave holes that hurt recall.)
    bool remove(Label label) {
        std::lock_guard<std::mutex> g(meta_mu_);
        auto it = label_to_id_.find(label);
        if (it == label_to_id_.end() || deleted_[it->second]) return false;
        deleted_[it->second] = 1;
        ++n_deleted_;
        return true;
    }

    // Internal id for a label, or -1. Used to build `allow` masks.
    std::int64_t id_of(Label label) const {
        auto it = label_to_id_.find(label);
        return it == label_to_id_.end() ? -1 : static_cast<std::int64_t>(it->second);
    }

    // -------------------------------------------------------- persistence

    void save(const std::string& path) const {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot open " + path + " for writing");
        const std::uint64_t n = size();
        std::uint64_t header[14] = {kMagic, dim_, max_, n, M_, efc_, static_cast<std::uint64_t>(metric_),
                                    static_cast<std::uint64_t>(max_level_), entry_, n_deleted_,
                                    sq8_, keep_full_, 0, 0};
        std::memcpy(&header[12], &sq_min_, sizeof sq_min_);
        std::memcpy(&header[13], &sq_step_, sizeof sq_step_);
        bool ok = std::fwrite(header, sizeof header, 1, f) == 1;
        ok = ok && put(f, data_, n * vec_bytes_);
        if (keep_full_) ok = ok && put(f, full_, n * dim_ * sizeof(float));
        ok = ok && put(f, links0_, n * (M0_ + 1) * 4);
        ok = ok && put(f, levels_.data(), n);
        ok = ok && put(f, labels_.data(), n * sizeof(Label));
        ok = ok && put(f, deleted_.data(), n);
        for (std::uint64_t i = 0; ok && i < n; ++i)
            if (levels_[i]) ok = put(f, upper_[i].get(), static_cast<std::size_t>(levels_[i]) * (M_ + 1) * 4);
        ok = (std::fclose(f) == 0) && ok;
        if (!ok) throw std::runtime_error("write to " + path + " failed");
    }

    static std::unique_ptr<HnswIndex> load(const std::string& path) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);
        std::uint64_t h[14];
        if (std::fread(h, sizeof h, 1, f) != 1 || h[0] != kMagic || h[3] > h[2] || h[6] > 2 || h[10] > 1 || h[11] > 1) {
            std::fclose(f);
            throw std::runtime_error(path + " is not a vecdb index");
        }
        std::unique_ptr<HnswIndex> ix;
        try {
            ix.reset(new HnswIndex(h[1], h[2], static_cast<Metric>(h[6]), h[4], h[5], 100, h[10] != 0, h[11] != 0));
        } catch (const std::invalid_argument&) {
            std::fclose(f);
            throw std::runtime_error(path + " is corrupt");
        }
        std::memcpy(&ix->sq_min_, &h[12], sizeof ix->sq_min_);
        std::memcpy(&ix->sq_step_, &h[13], sizeof ix->sq_step_);
        ix->trained_ = ix->sq8_;
        const std::uint64_t n = h[3];
        bool ok = get(f, ix->data_, n * ix->vec_bytes_);
        if (ix->keep_full_) ok = ok && get(f, ix->full_, n * ix->dim_ * sizeof(float));
        ok = ok && get(f, ix->links0_, n * (ix->M0_ + 1) * 4);
        ok = ok && get(f, ix->levels_.data(), n);
        ok = ok && get(f, ix->labels_.data(), n * sizeof(Label));
        ok = ok && get(f, ix->deleted_.data(), n);
        for (std::uint64_t i = 0; ok && i < n; ++i) {
            if (!ix->levels_[i]) continue;
            const std::size_t words = static_cast<std::size_t>(ix->levels_[i]) * (ix->M_ + 1);
            ix->upper_[i].reset(new std::uint32_t[words]);
            ok = get(f, ix->upper_[i].get(), words * 4);
        }
        std::fclose(f);
        if (!ok) throw std::runtime_error(path + " is truncated");
        // A file is untrusted input: every link must point inside the index.
        for (std::uint64_t i = 0; i < n; ++i) {
            for (int l = 0; l <= ix->levels_[i]; ++l) {
                const std::uint32_t* ls = ix->links(static_cast<std::uint32_t>(i), l);
                if (ls[0] > ix->max_links(l)) throw std::runtime_error(path + " is corrupt");
                for (std::uint32_t j = 1; j <= ls[0]; ++j)
                    if (ls[j] >= n || ix->levels_[ls[j]] < l) throw std::runtime_error(path + " is corrupt");
            }
            ix->label_to_id_.emplace(ix->labels_[i], static_cast<std::uint32_t>(i));
        }
        if (n && (h[8] >= n || static_cast<int>(h[7]) != ix->levels_[h[8]]))
            throw std::runtime_error(path + " is corrupt");
        ix->count_.store(static_cast<std::uint32_t>(n));
        ix->max_level_ = n ? static_cast<int>(h[7]) : -1;
        ix->entry_ = n ? static_cast<std::uint32_t>(h[8]) : kNone;
        ix->n_deleted_ = h[9];
        return ix;
    }

    // Would Auto answer a filter of this size by scanning? Exposed for benchmarks.
    bool use_scan(FilterStrategy strategy, std::size_t n_allow, std::size_t ef) const {
        if (strategy != FilterStrategy::Auto) return strategy == FilterStrategy::Scan;
        // A scan costs one distance per allowed vector. A filtered graph search
        // has to visit about ef / selectivity nodes, with a few distance
        // computations wasted per useful one. Setting the two equal gives
        // n_allow^2 = c * ef * N; below that, scanning is cheaper.
        const double n = static_cast<double>(n_allow);
        return n * n < kFilterCost * static_cast<double>(ef) * static_cast<double>(size());
    }

private:
    static constexpr std::uint32_t kNone = 0xFFFFFFFFu;
    static constexpr std::uint64_t kMagic = 0x3242444345567632ull;  // file format v2 (adds sq8)
    static constexpr double kFilterCost = 8.0;  // measured: see bench/bench_filter.py
    static constexpr std::size_t kMaxLinks = 256;

    void allocate() {
        if (M0_ > kMaxLinks) throw std::invalid_argument("M is too large");
        const std::size_t data_bytes = ((max_ * vec_bytes_) + 63) / 64 * 64;
        const std::size_t link_bytes = ((max_ * (M0_ + 1) * 4) + 63) / 64 * 64;
        data_ = static_cast<unsigned char*>(std::aligned_alloc(64, data_bytes));
        links0_ = static_cast<std::uint32_t*>(std::aligned_alloc(64, link_bytes));
        if (keep_full_) {
            full_ = static_cast<float*>(std::aligned_alloc(64, ((max_ * dim_ * sizeof(float)) + 63) / 64 * 64));
            if (!full_) throw std::bad_alloc();
        }
        if (!data_ || !links0_) throw std::bad_alloc();
        std::memset(links0_, 0, link_bytes);
        levels_.assign(max_, 0);
        labels_.assign(max_, 0);
        deleted_.assign(max_, 0);
        upper_.reset(new std::unique_ptr<std::uint32_t[]>[max_]);
        locks_.reset(new SpinLock[max_]);
    }

    const unsigned char* vec(std::uint32_t id) const { return data_ + static_cast<std::size_t>(id) * vec_bytes_; }
    const float* full(std::uint32_t id) const { return full_ + static_cast<std::size_t>(id) * dim_; }
    std::size_t max_links(int level) const { return level == 0 ? M0_ : M_; }

    std::uint32_t* links(std::uint32_t id, int level) const {
        if (level == 0) return links0_ + static_cast<std::size_t>(id) * (M0_ + 1);
        return upper_[id].get() + static_cast<std::size_t>(level - 1) * (M_ + 1);
    }

    std::uint32_t copy_links(std::uint32_t id, int level, std::uint32_t* out) const {
        std::lock_guard<SpinLock> g(locks_[id]);
        const std::uint32_t* ls = links(id, level);
        std::memcpy(out, ls + 1, ls[0] * sizeof(std::uint32_t));
        return ls[0];
    }

    // Distance between two STORED vectors (floats, or sq8 codes).
    float dist(const unsigned char* a, const unsigned char* b) const {
        if (sq8_) return static_cast<float>(l2_sq_u8(a, b, dim_));
        return exact(reinterpret_cast<const float*>(a), reinterpret_cast<const float*>(b));
    }

    float exact(const float* a, const float* b) const {
        return metric_ == Metric::L2 ? l2_sq(a, b, dim_) : 1.0f - dot(a, b, dim_);
    }

    // A code-space distance expressed in the index's metric. Codes are L2 in
    // units of one quantisation step; for unit vectors, 1 - cos = L2^2 / 2.
    float code_to_metric(float d) const {
        const float l2 = d * sq_step_ * sq_step_;
        return metric_ == Metric::L2 ? l2 : l2 / 2;
    }

    void normalize(float* v) const {
        const float n = std::sqrt(dot(v, v, dim_));
        if (n > 0)
            for (std::size_t i = 0; i < dim_; ++i) v[i] /= n;
    }

    void encode(const float* v, unsigned char* out) const {
        const float inv = 1.0f / sq_step_;
        for (std::size_t i = 0; i < dim_; ++i) {
            const float c = (v[i] - sq_min_) * inv + 0.5f;
            out[i] = static_cast<unsigned char>(c < 0 ? 0 : c > 255 ? 255 : c);
        }
    }

    void store(std::uint32_t id, const float* v) {
        unsigned char* dst = data_ + static_cast<std::size_t>(id) * vec_bytes_;
        if (!sq8_) {
            std::memcpy(dst, v, dim_ * sizeof(float));
            if (metric_ == Metric::Cosine) normalize(reinterpret_cast<float*>(dst));
            return;
        }
        std::vector<float> tmp(v, v + dim_);
        if (metric_ == Metric::Cosine) normalize(tmp.data());
        encode(tmp.data(), dst);
        if (keep_full_) std::memcpy(full_ + static_cast<std::size_t>(id) * dim_, tmp.data(), dim_ * sizeof(float));
    }

    // A query prepared in both forms the index may need: floats in the
    // index's metric space, and whatever representation vectors are stored in.
    struct Query {
        std::vector<float> normed;
        std::vector<unsigned char> codes;
        const float* floats;
        const unsigned char* stored;
        Query(const HnswIndex& ix, const float* q) : floats(q) {
            if (ix.metric_ == Metric::Cosine) {
                normed.assign(q, q + ix.dim_);
                ix.normalize(normed.data());
                floats = normed.data();
            }
            if (ix.sq8_) {
                codes.resize(ix.dim_);
                ix.encode(floats, codes.data());
                stored = codes.data();
            } else {
                stored = reinterpret_cast<const unsigned char*>(floats);
            }
        }
    };

    // Per-thread "have I seen this node in the current search" marks. Bumping
    // an epoch number resets all of them at once without touching the array.
    struct Visited {
        std::vector<std::uint16_t> tag;
        std::uint16_t epoch = 0;
        void begin(std::size_t n) {
            if (tag.size() < n) tag.assign(n, 0);
            if (++epoch == 0) {
                std::fill(tag.begin(), tag.end(), 0);
                epoch = 1;
            }
        }
        bool test_and_set(std::uint32_t id) {
            if (tag[id] == epoch) return true;
            tag[id] = epoch;
            return false;
        }
    };

    // Best-first search on one layer. Leaves up to `ef` results in `out`, closest first.
    void search_layer(const unsigned char* q, std::uint32_t entry, int level, std::size_t ef, bool locked,
                      const std::uint8_t* allow, std::vector<Neighbor>& out) const {
        static thread_local Visited visited;
        visited.begin(max_);

        auto usable = [&](std::uint32_t id) { return !deleted_[id] && (!allow || allow[id]); };
        std::priority_queue<Neighbor> best;                                                    // worst on top
        std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>> frontier;  // closest on top

        const float d0 = dist(q, vec(entry));
        visited.test_and_set(entry);
        frontier.push({d0, entry});
        float bound = std::numeric_limits<float>::max();
        if (usable(entry)) {
            best.push({d0, entry});
            bound = d0;
        }

        std::uint32_t buf[kMaxLinks];
        while (!frontier.empty()) {
            const Neighbor c = frontier.top();
            if (c.dist > bound && best.size() >= ef) break;
            frontier.pop();

            std::uint32_t n;
            const std::uint32_t* ls;
            if (locked) {
                n = copy_links(c.id, level, buf);
                ls = buf;
            } else {
                const std::uint32_t* raw = links(c.id, level);
                n = raw[0];
                ls = raw + 1;
            }
            // Pick out the neighbours not seen before, then compute their
            // distances four at a time (see dot4 in distance.hpp).
            std::uint32_t fresh[kMaxLinks];
            float dists[kMaxLinks];
            std::uint32_t m = 0;
            for (std::uint32_t i = 0; i < n; ++i)
                if (!visited.test_and_set(ls[i])) fresh[m++] = ls[i];
            std::uint32_t j = 0;
            if (!sq8_) {
                const float* qf = reinterpret_cast<const float*>(q);
                for (; j + 4 <= m; j += 4) {
                    const float* a = reinterpret_cast<const float*>(vec(fresh[j]));
                    const float* b = reinterpret_cast<const float*>(vec(fresh[j + 1]));
                    const float* c = reinterpret_cast<const float*>(vec(fresh[j + 2]));
                    const float* d = reinterpret_cast<const float*>(vec(fresh[j + 3]));
                    if (metric_ == Metric::L2) {
                        l2_sq4(qf, a, b, c, d, dim_, dists + j);
                    } else {
                        dot4(qf, a, b, c, d, dim_, dists + j);
                        for (int t = 0; t < 4; ++t) dists[j + t] = 1.0f - dists[j + t];
                    }
                }
            }
            for (; j < m; ++j) {
                if (j + 1 < m) __builtin_prefetch(vec(fresh[j + 1]));
                dists[j] = dist(q, vec(fresh[j]));
            }
            for (std::uint32_t i = 0; i < m; ++i) {
                const std::uint32_t nb = fresh[i];
                const float d = dists[i];
                if (best.size() < ef || d < bound) {
                    frontier.push({d, nb});
                    if (usable(nb)) {
                        best.push({d, nb});
                        if (best.size() > ef) best.pop();
                    }
                    if (best.size() >= ef) bound = best.top().dist;
                }
            }
        }
        out.resize(best.size());
        for (std::size_t i = best.size(); i-- > 0;) {
            out[i] = best.top();
            best.pop();
        }
    }

    // The paper's Algorithm 4. Take candidates closest-first, but skip one
    // that is closer to an already chosen neighbour than to the base node:
    // that direction is already covered. This keeps links spread out, which is
    // what lets a greedy walk cross between clusters.
    void select_neighbors(const std::vector<Neighbor>& cand, std::size_t m, std::vector<Neighbor>& out) const {
        out.clear();
        for (const Neighbor& c : cand) {
            if (out.size() >= m) break;
            bool keep = true;
            for (const Neighbor& s : out) {
                if (dist(vec(c.id), vec(s.id)) < c.dist) {
                    keep = false;
                    break;
                }
            }
            if (keep) out.push_back(c);
        }
    }

    // Add `id` to `node`'s list on `level`; if the list is full, re-select.
    void connect(std::uint32_t node, std::uint32_t id, float d, int level) {
        std::lock_guard<SpinLock> g(locks_[node]);
        std::uint32_t* ls = links(node, level);
        const std::uint32_t n = ls[0];
        for (std::uint32_t i = 1; i <= n; ++i)
            if (ls[i] == id) return;
        const std::size_t cap = max_links(level);
        if (n < cap) {
            ls[n + 1] = id;
            ls[0] = n + 1;
            return;
        }
        std::vector<Neighbor> cand;
        cand.reserve(n + 1);
        cand.push_back({d, id});
        const unsigned char* base = vec(node);
        for (std::uint32_t i = 1; i <= n; ++i) cand.push_back({dist(base, vec(ls[i])), ls[i]});
        std::sort(cand.begin(), cand.end());
        std::vector<Neighbor> chosen;
        select_neighbors(cand, cap, chosen);
        ls[0] = static_cast<std::uint32_t>(chosen.size());
        for (std::size_t i = 0; i < chosen.size(); ++i) ls[i + 1] = chosen[i].id;
    }

    static bool put(std::FILE* f, const void* p, std::size_t n) { return n == 0 || std::fwrite(p, 1, n, f) == n; }
    static bool get(std::FILE* f, void* p, std::size_t n) { return n == 0 || std::fread(p, 1, n, f) == n; }

    const std::size_t dim_, max_;
    const Metric metric_;
    const std::size_t M_, M0_, efc_;
    const double level_mult_;
    const bool sq8_, keep_full_;
    const std::size_t vec_bytes_;
    float sq_min_ = 0, sq_step_ = 1;  // code c stands for sq_min_ + c * sq_step_
    bool trained_ = false;

    unsigned char* data_ = nullptr;  // max_ * vec_bytes_: floats, or sq8 codes
    float* full_ = nullptr;          // original floats, only with sq8 + keep_full
    std::uint32_t* links0_ = nullptr;
    std::unique_ptr<std::unique_ptr<std::uint32_t[]>[]> upper_;
    std::unique_ptr<SpinLock[]> locks_;
    std::vector<std::uint8_t> levels_;
    std::vector<Label> labels_;
    std::vector<std::uint8_t> deleted_;
    std::unordered_map<Label, std::uint32_t> label_to_id_;

    std::atomic<std::uint32_t> count_{0};
    std::size_t n_deleted_ = 0;
    std::mutex meta_mu_;  // label map, level generator, slot allocation
    std::mutex top_mu_;   // entry point and top level
    int max_level_ = -1;
    std::uint32_t entry_ = kNone;
    std::mt19937_64 rng_;
};

}  // namespace vecdb
