"""Filtered search on SIFT1M: graph traversal vs scanning the allowed vectors.

For each selectivity (fraction of the million vectors a query may return) this
measures recall@10 against the exact answer among the allowed vectors, and
single-thread throughput, for three strategies:

    graph  traverse the HNSW graph, skipping disallowed nodes when collecting results
    scan   compute the distance to every allowed vector (exact)
    auto   choose by estimated cost (what search() does by default)

    python bench/bench_filter.py
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
import vecdb  # noqa: E402

K, EF, NQ = 10, 100, 500


def fvecs(path):
    a = np.fromfile(path, dtype=np.int32)
    return a.reshape(-1, a[0] + 1)[:, 1:].copy().view(np.float32)


def main():
    base = fvecs(ROOT / "data" / "sift" / "sift_base.fvecs")
    queries = fvecs(ROOT / "data" / "sift" / "sift_query.fvecs")[:NQ]
    n = len(base)
    path = ROOT / "data" / f"sift{n}.vecdb"
    if path.exists():
        ix = vecdb.Index.load(str(path))
    else:
        ix = vecdb.Index(128, n)
        ix.add(base)
        ix.save(str(path))

    rng = np.random.default_rng(0)
    rows = []
    print(f"SIFT1M, k={K}, ef={EF}, {NQ} queries, 1 thread")
    print(f"{'allowed':>9} {'vectors':>9} | {'graph recall':>12} {'graph q/s':>10} | {'scan q/s':>9} | {'auto picks':>10} {'auto recall':>11} {'auto q/s':>9}")
    for frac in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.001, 0.0001):
        allowed = np.sort(rng.choice(n, size=max(K, int(n * frac)), replace=False)).astype(np.uint64)
        res, qps = {}, {}
        for strategy in ("scan", "graph", "auto"):
            best = float("inf")
            for _ in range(2):
                t = time.perf_counter()
                res[strategy] = ix.search(queries, K, EF, threads=1, allowed=allowed, filter_strategy=strategy)[0]
                best = min(best, time.perf_counter() - t)
            qps[strategy] = NQ / best
        truth = res["scan"]  # exact among the allowed vectors
        rec = {s: float(np.mean([len(set(a) & set(b)) for a, b in zip(res[s], truth)]) / K) for s in res}
        picks = "scan" if np.array_equal(res["auto"], res["scan"]) and qps["auto"] < qps["graph"] * 3 and rec["graph"] < 1 \
            else ("scan" if abs(qps["auto"] - qps["scan"]) < abs(qps["auto"] - qps["graph"]) else "graph")
        rows.append({"selectivity": frac, "allowed": len(allowed), "graph_recall": round(rec["graph"], 4),
                     "graph_qps": round(qps["graph"]), "scan_qps": round(qps["scan"]), "auto": picks,
                     "auto_recall": round(rec["auto"], 4), "auto_qps": round(qps["auto"])})
        print(f"{frac:>9.2%} {len(allowed):>9,} | {rec['graph']:>12.4f} {qps['graph']:>10,.0f} | {qps['scan']:>9,.0f} | "
              f"{picks:>10} {rec['auto']:>11.4f} {qps['auto']:>9,.0f}")
    (ROOT / "bench" / "results_filter.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
