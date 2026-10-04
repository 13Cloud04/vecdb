// C++ tests. `make test` runs them; `make asan` and `make tsan` run them
// under the sanitizers (the parallel-build test is the one TSan cares about).
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <thread>
#include <vector>

#include "../src/hnsw.hpp"

using namespace vecdb;

#define CHECK(c)                                                                  \
    do {                                                                          \
        if (!(c)) {                                                               \
            std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__); \
            std::exit(1);                                                         \
        }                                                                         \
    } while (0)

static std::vector<float> random_vectors(std::size_t n, std::size_t d, std::uint64_t seed) {
    // Clustered, not uniform: uniform random high-dimensional data is a
    // misleadingly easy (or hard) case for graph indexes.
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0, 1);
    const std::size_t centers = 32;
    std::vector<float> c(centers * d), v(n * d);
    for (float& x : c) x = g(rng) * 4;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t k = rng() % centers;
        for (std::size_t j = 0; j < d; ++j) v[i * d + j] = c[k * d + j] + g(rng);
    }
    return v;
}

static double recall(const HnswIndex& ix, const std::vector<float>& q, std::size_t nq, std::size_t k, std::size_t ef,
                     const std::uint8_t* allow = nullptr) {
    std::size_t hit = 0, total = 0;
    for (std::size_t i = 0; i < nq; ++i) {
        const float* v = q.data() + i * ix.dim();
        auto truth = ix.brute_force(v, allow ? ix.size() : k);
        std::set<HnswIndex::Label> want;
        for (auto& t : truth) {
            if (allow && !allow[ix.id_of(t.second)]) continue;
            if (want.size() < k) want.insert(t.second);
        }
        for (auto& r : ix.search(v, k, ef, allow)) hit += want.count(r.second);
        total += want.size();
    }
    return static_cast<double>(hit) / total;
}

static void build(HnswIndex& ix, const std::vector<float>& v, std::size_t n, unsigned threads) {
    std::atomic<std::size_t> next{0};
    auto work = [&] {
        for (std::size_t i; (i = next.fetch_add(1)) < n;) ix.add(v.data() + i * ix.dim(), 1000 + i);
    };
    std::vector<std::thread> ts;
    for (unsigned t = 1; t < threads; ++t) ts.emplace_back(work);
    work();
    for (auto& t : ts) t.join();
}

static void kernels_match_scalar() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1, 1);
    for (std::size_t d : {1, 3, 4, 7, 16, 17, 31, 64, 100, 128, 129, 960}) {
        std::vector<float> a(d), b(d);
        for (std::size_t i = 0; i < d; ++i) a[i] = u(rng), b[i] = u(rng);
        const float l = l2_sq_scalar(a.data(), b.data(), d), p = dot_scalar(a.data(), b.data(), d);
        CHECK(std::fabs(l2_sq(a.data(), b.data(), d) - l) <= 1e-4f * (1 + l));
        CHECK(std::fabs(dot(a.data(), b.data(), d) - p) <= 1e-4f * (1 + std::fabs(p)));
        // The four-at-a-time kernels must agree with four separate calls.
        std::vector<float> c(d), e(d);
        for (std::size_t i = 0; i < d; ++i) c[i] = u(rng), e[i] = u(rng);
        float l4[4], p4[4];
        l2_sq4(a.data(), b.data(), c.data(), e.data(), a.data(), d, l4);
        dot4(a.data(), b.data(), c.data(), e.data(), a.data(), d, p4);
        const float* others[4] = {b.data(), c.data(), e.data(), a.data()};
        for (int k = 0; k < 4; ++k) {
            const float ls = l2_sq_scalar(a.data(), others[k], d), ps = dot_scalar(a.data(), others[k], d);
            CHECK(std::fabs(l4[k] - ls) <= 1e-4f * (1 + ls));
            CHECK(std::fabs(p4[k] - ps) <= 1e-4f * (1 + std::fabs(ps)));
        }
    }
}

static void recall_all_metrics(std::size_t n) {
    const std::size_t d = 48, nq = 200;
    auto v = random_vectors(n + nq, d, 7);  // queries come from the same clusters as the data
    std::vector<float> q(v.end() - nq * d, v.end());
    for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
        HnswIndex ix(d, n, m, 16, 200);
        if (m == Metric::InnerProduct) {
            // Inner product is only a sensible "distance" on normalised data.
            for (std::size_t i = 0; i < n + nq; ++i) {
                float s = std::sqrt(dot(v.data() + i * d, v.data() + i * d, d));
                for (std::size_t j = 0; j < d; ++j) v[i * d + j] /= s;
            }
            q.assign(v.end() - nq * d, v.end());
        }
        build(ix, v, n, 1);
        CHECK(ix.size() == n);
        const double r = recall(ix, q, nq, 10, 100);
        std::printf("  metric %u: recall@10 = %.4f (ef=100, n=%zu)\n", static_cast<unsigned>(m), r, n);
        CHECK(r > 0.97);
        CHECK(recall(ix, q, nq, 10, 400) >= r - 0.005);  // more effort never hurts much
    }
}

static void results_are_sorted_and_exact_for_members(std::size_t n) {
    const std::size_t d = 32;
    auto v = random_vectors(n, d, 3);
    HnswIndex ix(d, n);
    build(ix, v, n, 1);
    for (std::size_t i = 0; i < 100; ++i) {
        auto r = ix.search(v.data() + i * d, 5, 50);
        CHECK(r.size() == 5 && r[0].second == 1000 + i && r[0].first < 1e-6f);  // a stored vector finds itself
        for (std::size_t j = 1; j < r.size(); ++j) CHECK(r[j - 1].first <= r[j].first);
    }
}

static void parallel_build_matches_serial_quality(std::size_t n, unsigned threads) {
    const std::size_t d = 48, nq = 200;
    auto v = random_vectors(n + nq, d, 11);
    std::vector<float> q(v.end() - nq * d, v.end());
    HnswIndex serial(d, n), par(d, n);
    build(serial, v, n, 1);
    build(par, v, n, threads);
    CHECK(par.size() == n);
    const double rs = recall(serial, q, nq, 10, 100), rp = recall(par, q, nq, 10, 100);
    std::printf("  recall@10 serial %.4f, %u threads %.4f\n", rs, threads, rp);
    CHECK(rp > rs - 0.01);
}

static void deletes_and_filters(std::size_t n) {
    const std::size_t d = 32, nq = 100;
    auto v = random_vectors(n + nq, d, 21);
    std::vector<float> q(v.end() - nq * d, v.end());
    HnswIndex ix(d, n);
    build(ix, v, n, 1);
    for (std::size_t i = 0; i < n; i += 3) CHECK(ix.remove(1000 + i));
    CHECK(!ix.remove(1000) && !ix.remove(5));
    CHECK(ix.deleted() == (n + 2) / 3);
    for (std::size_t i = 0; i < nq; ++i)
        for (auto& r : ix.search(q.data() + i * d, 10, 100)) CHECK((r.second - 1000) % 3 != 0);
    CHECK(recall(ix, q, nq, 10, 200) > 0.95);

    std::vector<std::uint8_t> allow(n, 0);  // keep one node in ten
    for (std::size_t i = 0; i < n; i += 10) allow[i] = 1;
    for (std::size_t i = 0; i < nq; ++i)
        for (auto& r : ix.search(q.data() + i * d, 5, 400, allow.data())) CHECK(allow[ix.id_of(r.second)]);
    const double rf = recall(ix, q, nq, 5, 400, allow.data());
    std::printf("  recall@5 with a 10%% filter and 33%% deleted: %.4f (ef=400)\n", rf);
    CHECK(rf > 0.85);
}

static void save_load_round_trip(std::size_t n) {
    const std::size_t d = 24, nq = 50;
    auto v = random_vectors(n + nq, d, 31);
    std::vector<float> q(v.end() - nq * d, v.end());
    HnswIndex ix(d, n + 10, Metric::Cosine, 12, 100);
    build(ix, v, n, 1);
    ix.remove(1003);
    const std::string path = "build/roundtrip.vecdb";
    ix.save(path);
    auto back = HnswIndex::load(path);
    CHECK(back->size() == n && back->deleted() == 1 && back->M() == 12 && back->metric() == Metric::Cosine);
    for (std::size_t i = 0; i < nq; ++i) {
        auto a = ix.search(q.data() + i * d, 10, 80), b = back->search(q.data() + i * d, 10, 80);
        CHECK(a == b);
    }
    back->add(v.data(), 999999);  // a loaded index accepts new vectors
    CHECK(back->size() == n + 1);

    // Corrupt files are rejected instead of crashing.
    std::FILE* f = std::fopen(path.c_str(), "r+b");
    std::fseek(f, 14 * 8 + static_cast<long>(n * d * 4) + 4, SEEK_SET);  // header, vectors, then node 0's first neighbour
    const std::uint32_t bad = 0x7FFFFFFF;
    std::fwrite(&bad, 4, 1, f);
    std::fclose(f);
    bool threw = false;
    try {
        HnswIndex::load(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

static void quantized_index(std::size_t n) {
    const std::size_t d = 64, nq = 200;
    auto v = random_vectors(n + nq, d, 41);
    std::vector<float> q(v.end() - nq * d, v.end());
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> byte(0, 255);
    for (std::size_t dim : {1, 15, 16, 17, 100, 128, 1536}) {  // the u8 kernel against the plain loop
        std::vector<std::uint8_t> a(dim), b(dim);
        for (std::size_t i = 0; i < dim; ++i) a[i] = static_cast<std::uint8_t>(byte(rng)), b[i] = static_cast<std::uint8_t>(byte(rng));
        CHECK(l2_sq_u8(a.data(), b.data(), dim) == l2_sq_u8_scalar(a.data(), b.data(), dim));
    }
    for (Metric m : {Metric::L2, Metric::Cosine}) {
        HnswIndex exact(d, n, m), codes(d, n, m, 16, 200, 100, true, false), reranked(d, n, m, 16, 200, 100, true, true);
        bool untrained = false;
        try { codes.add(v.data(), 1); } catch (const std::logic_error&) { untrained = true; }
        CHECK(untrained);
        codes.train(v.data(), n);
        reranked.train(v.data(), n);
        build(exact, v, n, 4);
        build(codes, v, n, 4);
        build(reranked, v, n, 4);
        CHECK(codes.vector_bytes() * 4 == exact.vector_bytes());

        std::size_t hit_codes = 0, hit_rr = 0, total = 0;
        double err = 0;
        for (std::size_t i = 0; i < nq; ++i) {
            const float* qv = q.data() + i * d;
            std::set<HnswIndex::Label> want;
            auto truth = exact.brute_force(qv, 10);
            for (auto& t : truth) want.insert(t.second);
            auto c = codes.search(qv, 10, 100);
            for (auto& r : c) hit_codes += want.count(r.second);
            for (auto& r : reranked.search(qv, 10, 100)) hit_rr += want.count(r.second);
            total += want.size();
            err += std::fabs(c[0].first - truth[0].first) / (truth[0].first + 1e-6f);  // reported distance is in the metric's units
        }
        const double rc = double(hit_codes) / total, rr = double(hit_rr) / total;
        std::printf("  sq8 metric %u: recall@10 codes only %.4f, with re-ranking %.4f, distance error %.1f%%\n",
                    static_cast<unsigned>(m), rc, rr, 100 * err / nq);
        CHECK(rc > 0.90 && rr > 0.97 && rr >= rc && err / nq < 0.10);

        codes.save("build/sq8.vecdb");
        auto back = HnswIndex::load("build/sq8.vecdb");
        CHECK(back->quantized() && back->size() == n);
        for (std::size_t i = 0; i < 20; ++i)
            CHECK(back->search(q.data() + i * d, 10, 100) == codes.search(q.data() + i * d, 10, 100));
    }
    bool refused = false;
    try { HnswIndex bad(8, 10, Metric::InnerProduct, 16, 200, 100, true); } catch (const std::invalid_argument&) { refused = true; }
    CHECK(refused);
}

static void filter_strategies(std::size_t n) {
    const std::size_t d = 32, nq = 100;
    auto v = random_vectors(n + nq, d, 51);
    std::vector<float> q(v.end() - nq * d, v.end());
    HnswIndex ix(d, n);
    build(ix, v, n, 4);
    for (std::size_t every : {2, 20, 500}) {
        std::vector<std::uint8_t> allow(n, 0);
        std::vector<std::uint32_t> ids;
        for (std::size_t i = 0; i < n; i += every) allow[i] = 1, ids.push_back(static_cast<std::uint32_t>(i));
        std::size_t hit = 0, total = 0;
        for (std::size_t i = 0; i < nq; ++i) {
            const float* qv = q.data() + i * d;
            auto scan = ix.search(qv, 5, 50, allow.data(), ids.data(), ids.size(), FilterStrategy::Scan);
            auto autos = ix.search(qv, 5, 50, allow.data(), ids.data(), ids.size(), FilterStrategy::Auto);
            CHECK(scan.size() == std::min<std::size_t>(5, ids.size()));
            for (auto& r : scan) CHECK(allow[ix.id_of(r.second)]);
            for (std::size_t j = 1; j < scan.size(); ++j) CHECK(scan[j - 1].first <= scan[j].first);
            if (ix.use_scan(FilterStrategy::Auto, ids.size(), 50)) CHECK(autos == scan);  // Auto chose the exact path
            std::set<HnswIndex::Label> want;
            for (auto& r : scan) want.insert(r.second);  // the scan is exact, so it is the truth
            for (auto& r : autos) hit += want.count(r.second);
            total += want.size();
        }
        std::printf("  filter keeping 1 in %zu: auto uses %s, recall@5 %.4f\n", every,
                    ix.use_scan(FilterStrategy::Auto, ids.size(), 50) ? "scan" : "graph", double(hit) / total);
        CHECK(double(hit) / total > 0.95);
    }
}

static void rejects_misuse() {
    HnswIndex ix(4, 2);
    const float v[4] = {1, 2, 3, 4};
    CHECK(ix.search(v, 3, 10).empty());
    ix.add(v, 1);
    bool dup = false, full = false;
    try { ix.add(v, 1); } catch (const std::invalid_argument&) { dup = true; }
    ix.add(v, 2);
    try { ix.add(v, 3); } catch (const std::length_error&) { full = true; }
    CHECK(dup && full && ix.size() == 2);
    CHECK(ix.search(v, 10, 10).size() == 2);
}

int main(int argc, char** argv) {
    const std::size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000;
    const unsigned threads = argc > 2 ? std::atoi(argv[2]) : 8;
    kernels_match_scalar();
    rejects_misuse();
    recall_all_metrics(n);
    results_are_sorted_and_exact_for_members(n / 4);
    parallel_build_matches_serial_quality(n, threads);
    deletes_and_filters(n / 2);
    save_load_round_trip(n / 4);
    quantized_index(n / 2);
    filter_strategies(n);
    std::puts("test_hnsw: all passed");
}
