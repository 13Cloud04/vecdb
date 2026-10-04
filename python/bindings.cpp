// Python bindings. Everything heavy runs with the GIL released, on a small
// pool of std::threads that pull row indices from a shared atomic counter.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>
#include <exception>
#include <thread>

#include "../src/hnsw.hpp"

namespace py = pybind11;
using vecdb::FilterStrategy;
using vecdb::HnswIndex;
using vecdb::Metric;
using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using LabelArray = py::array_t<std::uint64_t, py::array::c_style | py::array::forcecast>;

template <class Fn>
static void parallel_for(std::size_t n, int threads, Fn fn) {
    if (threads <= 0) threads = static_cast<int>(std::thread::hardware_concurrency());
    threads = std::max(1, std::min<int>(threads, static_cast<int>(n)));
    std::atomic<std::size_t> next{0};
    std::exception_ptr error;
    std::mutex error_mu;
    auto work = [&] {
        try {
            for (std::size_t i; (i = next.fetch_add(1, std::memory_order_relaxed)) < n;) fn(i);
        } catch (...) {
            std::lock_guard<std::mutex> g(error_mu);
            if (!error) error = std::current_exception();
            next.store(n);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    if (error) std::rethrow_exception(error);
}

static Metric parse_metric(const std::string& m) {
    if (m == "l2") return Metric::L2;
    if (m == "ip") return Metric::InnerProduct;
    if (m == "cosine") return Metric::Cosine;
    throw py::value_error("metric must be 'l2', 'ip' or 'cosine'");
}

static FilterStrategy parse_strategy(const std::string& s) {
    if (s == "auto") return FilterStrategy::Auto;
    if (s == "graph") return FilterStrategy::Graph;
    if (s == "scan") return FilterStrategy::Scan;
    throw py::value_error("filter_strategy must be 'auto', 'graph' or 'scan'");
}

static void check_dim(const FloatArray& a, std::size_t dim) {
    if (a.ndim() != 2 || static_cast<std::size_t>(a.shape(1)) != dim)
        throw py::value_error("expected a 2-D float32 array with " + std::to_string(dim) + " columns");
}

PYBIND11_MODULE(vecdb, m) {
    m.doc() = "HNSW vector index written from scratch in C++";

    py::class_<HnswIndex>(m, "Index")
        .def(py::init([](std::size_t dim, std::size_t max_elements, const std::string& metric, std::size_t M,
                         std::size_t ef_construction, std::uint64_t seed, py::object quantize, bool rerank) {
                 bool sq8 = false;
                 if (!quantize.is_none()) {
                     if (quantize.cast<std::string>() != "sq8") throw py::value_error("quantize must be None or 'sq8'");
                     sq8 = true;
                 }
                 return new HnswIndex(dim, max_elements, parse_metric(metric), M, ef_construction, seed, sq8, rerank);
             }),
             py::arg("dim"), py::arg("max_elements"), py::arg("metric") = "l2", py::arg("M") = 16,
             py::arg("ef_construction") = 200, py::arg("seed") = 100, py::arg("quantize") = py::none(),
             py::arg("rerank") = false,
             "quantize='sq8' stores one byte per dimension instead of a float32.\n"
             "rerank=True (with sq8) also keeps the floats and re-ranks the final candidates exactly.")
        .def("add",
             [](HnswIndex& ix, FloatArray vectors, py::object labels, int threads) {
                 check_dim(vectors, ix.dim());
                 const std::size_t n = vectors.shape(0);
                 std::vector<std::uint64_t> ids(n);
                 if (labels.is_none()) {
                     for (std::size_t i = 0; i < n; ++i) ids[i] = ix.size() + i;
                 } else {
                     auto l = labels.cast<LabelArray>();
                     if (l.ndim() != 1 || static_cast<std::size_t>(l.shape(0)) != n)
                         throw py::value_error("labels must be a 1-D array with one entry per vector");
                     std::copy(l.data(), l.data() + n, ids.begin());
                 }
                 const float* data = vectors.data();
                 const std::size_t dim = ix.dim();
                 py::gil_scoped_release release;
                 if (!ix.trained()) ix.train(data, n);  // sq8: the first batch fixes the code range
                 parallel_for(n, threads, [&](std::size_t i) { ix.add(data + i * dim, ids[i]); });
             },
             py::arg("vectors"), py::arg("labels") = py::none(), py::arg("threads") = 0,
             "Insert vectors. Labels default to consecutive integers.")
        .def("search",
             [](const HnswIndex& ix, FloatArray queries, std::size_t k, std::size_t ef, int threads,
                py::object allowed, const std::string& filter_strategy) {
                 check_dim(queries, ix.dim());
                 const std::size_t nq = queries.shape(0), dim = ix.dim();
                 const FilterStrategy strategy = parse_strategy(filter_strategy);
                 std::vector<std::uint8_t> mask;
                 std::vector<std::uint32_t> ids;
                 if (!allowed.is_none()) {
                     auto a = allowed.cast<LabelArray>();
                     mask.assign(ix.capacity(), 0);
                     ids.reserve(a.size());
                     for (py::ssize_t i = 0; i < a.size(); ++i) {
                         const auto id = ix.id_of(a.data()[i]);
                         if (id >= 0 && !mask[id]) {
                             mask[id] = 1;
                             ids.push_back(static_cast<std::uint32_t>(id));
                         }
                     }
                 }
                 py::array_t<std::int64_t> labels({nq, k});
                 py::array_t<float> dists({nq, k});
                 std::int64_t* lp = labels.mutable_data();
                 float* dp = dists.mutable_data();
                 const float* q = queries.data();
                 {
                     py::gil_scoped_release release;
                     parallel_for(nq, threads, [&](std::size_t i) {
                         auto r = mask.empty() ? ix.search(q + i * dim, k, ef)
                                               : ix.search(q + i * dim, k, ef, mask.data(), ids.data(), ids.size(), strategy);
                         for (std::size_t j = 0; j < k; ++j) {
                             lp[i * k + j] = j < r.size() ? static_cast<std::int64_t>(r[j].second) : -1;
                             dp[i * k + j] = j < r.size() ? r[j].first : std::numeric_limits<float>::infinity();
                         }
                     });
                 }
                 return py::make_tuple(labels, dists);
             },
             py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 100, py::arg("threads") = 1,
             py::arg("allowed") = py::none(), py::arg("filter_strategy") = "auto",
             "Returns (labels, distances), each of shape (n_queries, k). Missing results are -1 / inf.\n"
             "`allowed`, if given, is an array of labels; only those may be returned. A filter that\n"
             "keeps few vectors is answered by scanning them (exact); 'graph' or 'scan' force one way.")
        .def("brute_force",
             [](const HnswIndex& ix, FloatArray queries, std::size_t k, int threads) {
                 check_dim(queries, ix.dim());
                 const std::size_t nq = queries.shape(0), dim = ix.dim();
                 py::array_t<std::int64_t> labels({nq, k});
                 std::int64_t* lp = labels.mutable_data();
                 const float* q = queries.data();
                 {
                     py::gil_scoped_release release;
                     parallel_for(nq, threads, [&](std::size_t i) {
                         auto r = ix.brute_force(q + i * dim, k);
                         for (std::size_t j = 0; j < k; ++j)
                             lp[i * k + j] = j < r.size() ? static_cast<std::int64_t>(r[j].second) : -1;
                     });
                 }
                 return labels;
             },
             py::arg("queries"), py::arg("k") = 10, py::arg("threads") = 0, "Exact search (ground truth).")
        .def("remove", &HnswIndex::remove, "Tombstone a label. Returns False if it is absent or already removed.")
        .def("save", &HnswIndex::save)
        .def_static("load", [](const std::string& path) { return HnswIndex::load(path).release(); },
                    py::return_value_policy::take_ownership)
        .def("__len__", &HnswIndex::size)
        .def_property_readonly("dim", &HnswIndex::dim)
        .def_property_readonly("deleted", &HnswIndex::deleted)
        .def_property_readonly("max_level", &HnswIndex::max_level)
        .def_property_readonly("memory_bytes", &HnswIndex::memory_bytes)
        .def_property_readonly("vector_bytes", &HnswIndex::vector_bytes)
        .def_property_readonly("quantized", &HnswIndex::quantized);
}
