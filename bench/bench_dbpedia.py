"""Text embeddings: float32 vs 8-bit scalar quantisation.

114,386 DBpedia entities embedded with OpenAI text-embedding-ada-002 (1536-d),
cosine similarity, 1,000 held-out queries. Ground truth is exact brute force
over the float vectors.

For each configuration: bytes spent on vectors, recall@10, and single-thread
throughput, at several ef. hnswlib and FAISS (both float32) are included as
references, interleaved with ours within each measurement round.

    python bench/fetch_dbpedia.py && make py && python bench/bench_dbpedia.py
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import vecdb  # noqa: E402

M, EFC, K = 16, 200, 10
EFS = [20, 40, 80, 160, 320]
ROUNDS = int(os.environ.get("ROUNDS", 3))
THREADS = os.cpu_count()


def recall(found, truth):
    return float(np.mean([len(set(f) & set(t)) for f, t in zip(found, truth)]) / K)


def main():
    data = ROOT / "data" / "dbpedia"
    base = np.load(data / "base.npy")
    queries = np.load(data / "queries.npy")
    n, d = base.shape
    print(f"{n:,} x {d} vectors, {len(queries):,} queries, cosine, M={M}, ef_construction={EFC}")

    search, info = {}, {}

    def ours(name, **kw):
        t = time.perf_counter()
        ix = vecdb.Index(d, n, metric="cosine", M=M, ef_construction=EFC, **kw)
        ix.add(base, threads=THREADS)
        info[name] = {"build_s": round(time.perf_counter() - t, 1), "vector_mb": round(ix.vector_bytes / 2**20, 1),
                      "index_mb": round(ix.memory_bytes / 2**20, 1)}
        search[name] = lambda ef, ix=ix: ix.search(queries, K, ef, threads=1)[0]
        return ix

    exact = ours("vecdb float32")
    truth = exact.brute_force(queries, K)
    ours("vecdb sq8", quantize="sq8")
    ours("vecdb sq8 + rerank", quantize="sq8", rerank=True)

    import hnswlib
    t = time.perf_counter()
    h = hnswlib.Index(space="cosine", dim=d)
    h.init_index(max_elements=n, M=M, ef_construction=EFC, random_seed=100)
    h.set_num_threads(THREADS)
    h.add_items(base)
    info["hnswlib float32"] = {"build_s": round(time.perf_counter() - t, 1), "vector_mb": round(n * d * 4 / 2**20, 1)}

    def hnswlib_search(ef):
        h.set_ef(ef)
        return h.knn_query(queries, K, num_threads=1)[0]
    search["hnswlib float32"] = hnswlib_search

    try:
        import faiss
        unit = base / np.linalg.norm(base, axis=1, keepdims=True)
        unit_q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        faiss.omp_set_num_threads(THREADS)
        t = time.perf_counter()
        f = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
        f.hnsw.efConstruction = EFC
        f.add(unit)
        info["faiss float32"] = {"build_s": round(time.perf_counter() - t, 1), "vector_mb": round(n * d * 4 / 2**20, 1)}
        faiss.omp_set_num_threads(1)

        def faiss_search(ef):
            f.hnsw.efSearch = ef
            return f.search(unit_q, K)[1]
        search["faiss float32"] = faiss_search
    except ImportError:
        pass

    best = {name: {ef: float("inf") for ef in EFS} for name in search}
    found = {name: {} for name in search}
    for _ in range(ROUNDS):
        for ef in EFS:
            for name, fn in search.items():
                t = time.perf_counter()
                out = fn(ef)
                best[name][ef] = min(best[name][ef], time.perf_counter() - t)
                found[name][ef] = np.asarray(out)

    t = time.perf_counter()
    exact.brute_force(queries[:100], K, threads=1)
    brute_qps = 100 / (time.perf_counter() - t)

    print(f"exact scan: {brute_qps:.0f} queries/sec on one thread\n")
    for name in search:
        info[name]["curve"] = [{"ef": ef, "recall": round(recall(found[name][ef], truth), 4),
                                "qps": round(len(queries) / best[name][ef])} for ef in EFS]
        i = info[name]
        print(f"{name}: vectors {i['vector_mb']} MB, build {i['build_s']} s")
        print(f"  {'ef':>5} {'recall@10':>10} {'queries/sec':>12}")
        for p in i["curve"]:
            print(f"  {p['ef']:>5} {p['recall']:>10.4f} {p['qps']:>12,}")
    info["brute_force_qps"] = round(brute_qps, 1)
    (ROOT / "bench" / "results_dbpedia.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
