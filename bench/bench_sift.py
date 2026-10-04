"""Recall vs throughput on SIFT1M (1,000,000 base vectors, 128-d, 10,000 queries).

Compares this index with hnswlib and FAISS's HNSW at identical parameters
(M=16, ef_construction=200). Build uses every core. Queries run on ONE thread:
each library is handed all 10,000 queries in a single call with its thread
count set to 1, so no per-call Python overhead is included for any of them.
All indexes are built before any timing starts, and the libraries then take
turns inside each measurement round.

    python bench/fetch_sift.py && make py && python bench/bench_sift.py
"""
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import vecdb  # noqa: E402

M, EFC, K = 16, 200, 10
EFS = [10, 20, 40, 80, 160, 320]
THREADS = os.cpu_count()
ROUNDS = int(os.environ.get("ROUNDS", 5))


def fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)


def ivecs(path):
    a = np.fromfile(path, dtype=np.int32)
    return a.reshape(-1, a[0] + 1)[:, 1:]


def recall(found, truth):
    return float(np.mean([len(set(f) & set(t)) for f, t in zip(found, truth)]) / K)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    sift = ROOT / "data" / "sift"
    cache = ROOT / "data"
    base = fvecs(sift / "sift_base.fvecs")[:n]
    queries = fvecs(sift / "sift_query.fvecs")
    print(f"{n:,} x {base.shape[1]} base vectors, {len(queries):,} queries, M={M}, ef_construction={EFC}, "
          f"build threads={THREADS}, query threads=1")
    results = {"machine": platform.processor() or platform.machine(), "n": n, "M": M, "ef_construction": EFC,
               "build_threads": THREADS, "libraries": {}}
    search = {}

    # ---- build all three first (indexes are cached on disk between runs)
    path = cache / f"sift{n}.vecdb"
    t = time.perf_counter()
    if path.exists():
        mine = vecdb.Index.load(str(path))
        build = None
    else:
        mine = vecdb.Index(128, n, M=M, ef_construction=EFC)
        mine.add(base, threads=THREADS)
        build = time.perf_counter() - t
        mine.save(str(path))
    results["libraries"]["vecdb"] = {"build_s": build and round(build, 1), "index_mb": round(mine.memory_bytes / 2**20)}
    search["vecdb"] = lambda ef: mine.search(queries, K, ef, threads=1)[0]

    import hnswlib
    path = cache / f"sift{n}.hnswlib"
    h = hnswlib.Index(space="l2", dim=128)
    t = time.perf_counter()
    if path.exists():
        h.load_index(str(path), max_elements=n)
        build = None
    else:
        h.init_index(max_elements=n, M=M, ef_construction=EFC, random_seed=100)
        h.set_num_threads(THREADS)
        h.add_items(base)
        build = time.perf_counter() - t
        h.save_index(str(path))
    results["libraries"]["hnswlib"] = {"build_s": build and round(build, 1)}

    def hnswlib_search(ef):
        h.set_ef(ef)
        return h.knn_query(queries, K, num_threads=1)[0]
    search["hnswlib"] = hnswlib_search

    try:
        import faiss
        path = cache / f"sift{n}.faiss"
        faiss.omp_set_num_threads(THREADS)
        t = time.perf_counter()
        if path.exists():
            f = faiss.read_index(str(path))
            build = None
        else:
            f = faiss.IndexHNSWFlat(128, M)
            f.hnsw.efConstruction = EFC
            f.add(base)
            build = time.perf_counter() - t
            faiss.write_index(f, str(path))
        faiss.omp_set_num_threads(1)
        results["libraries"]["faiss"] = {"build_s": build and round(build, 1)}

        def faiss_search(ef):
            f.hnsw.efSearch = ef
            return f.search(queries, K)[1]
        search["faiss"] = faiss_search
    except ImportError:
        pass

    if n == 1_000_000:
        truth = ivecs(sift / "sift_groundtruth.ivecs")[:, :K]
    else:
        truth = mine.brute_force(queries, K)

    # ---- measure. Libraries take turns within each round, so heat, core
    # scheduling and cache state affect all of them alike; the best of ROUNDS
    # rounds is reported for each.
    best = {name: {ef: float("inf") for ef in EFS} for name in search}
    found = {name: {} for name in search}
    for _ in range(ROUNDS):
        for ef in EFS:
            for name, fn in search.items():
                t = time.perf_counter()
                out = fn(ef)
                best[name][ef] = min(best[name][ef], time.perf_counter() - t)
                found[name][ef] = np.asarray(out)
    for name in search:
        results["libraries"][name]["curve"] = [
            {"ef": ef, "recall": round(recall(found[name][ef], truth), 4), "qps": round(len(queries) / best[name][ef])}
            for ef in EFS]

    sample = queries[:200]
    t = time.perf_counter()
    mine.brute_force(sample, K, threads=1)
    results["brute_force_qps"] = round(len(sample) / (time.perf_counter() - t), 1)

    print(f"\nbrute force (exact, 1 thread): {results['brute_force_qps']} queries/sec")
    for name, lib in results["libraries"].items():
        mem = f", index {lib['index_mb']} MB" if "index_mb" in lib else ""
        b = f"build {lib['build_s']} s" if lib["build_s"] else "loaded from cache"
        print(f"\n{name}: {b}{mem}")
        print(f"  {'ef':>5} {'recall@10':>10} {'queries/sec':>12}")
        for p in lib["curve"]:
            print(f"  {p['ef']:>5} {p['recall']:>10.4f} {p['qps']:>12,}")
    out = ROOT / "bench" / ("results_sift1m.json" if n == 1_000_000 else f"results_sift{n}.json")
    out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
