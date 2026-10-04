// How much does the hand-written SIMD kernel buy over the plain loop?
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "../src/distance.hpp"

using namespace vecdb;

template <class Fn>
static double ns_per_call(Fn fn, const std::vector<float>& data, std::size_t d, std::size_t n) {
    const std::size_t iters = 400'000'000 / d;  // ~0.1-0.5 s per measurement
    volatile float sink = 0;
    double best = 1e18;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        float acc = 0;
        for (std::size_t i = 0; i < iters; ++i)
            acc += fn(data.data() + (i % n) * d, data.data() + ((i * 7 + 1) % n) * d, d);
        sink = acc;
        best = std::min(best, std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count() / iters);
    }
    (void)sink;
    return best;
}

int main() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(-1, 1);
    std::printf("%6s %14s %14s %9s\n", "dim", "scalar ns", "SIMD ns", "speedup");
    for (std::size_t d : {64, 128, 384, 768, 1536}) {
        const std::size_t n = 512;  // 512 vectors: fits in cache, so this measures arithmetic, not memory
        std::vector<float> data(n * d);
        for (float& x : data) x = u(rng);
        const double s = ns_per_call(l2_sq_scalar, data, d, n), v = ns_per_call(l2_sq, data, d, n);
        std::printf("%6zu %14.1f %14.1f %8.1fx\n", d, s, v, s / v);
    }
}
