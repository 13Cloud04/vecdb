import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import vecdb  # noqa: E402


def clustered(n, d, seed):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 4, (32, d))
    return (centers[rng.integers(0, 32, n)] + rng.normal(0, 1, (n, d))).astype(np.float32)


@pytest.fixture(scope="module")
def data():
    v = clustered(21_000, 64, 0)
    return v[:20_000], v[20_000:]


def recall(found, truth):
    return np.mean([len(set(f) & set(t)) / len(t) for f, t in zip(found, truth)])


def test_recall_and_shapes(data):
    base, queries = data
    ix = vecdb.Index(64, len(base))
    ix.add(base)
    assert len(ix) == len(base) and ix.dim == 64
    labels, dists = ix.search(queries, k=10, ef=100)
    assert labels.shape == dists.shape == (len(queries), 10)
    assert np.all(np.diff(dists, axis=1) >= 0)
    assert recall(labels, ix.brute_force(queries, 10)) > 0.97


def test_matches_numpy_ground_truth(data):
    base, queries = data
    ix = vecdb.Index(64, len(base))
    ix.add(base, threads=1)
    q = queries[:50]
    d2 = ((q[:, None, :] - base[None, :, :]) ** 2).sum(-1)
    truth = np.argsort(d2, axis=1)[:, :10]
    assert np.array_equal(np.sort(ix.brute_force(q, 10), axis=1), np.sort(truth, axis=1))


def test_custom_labels_remove_and_filter(data):
    base, queries = data
    labels = np.arange(len(base), dtype=np.uint64) * 7 + 3
    ix = vecdb.Index(64, len(base), metric="cosine")
    ix.add(base, labels)
    found, _ = ix.search(base[:5], k=1, ef=50)
    assert found[:, 0].tolist() == labels[:5].tolist()

    assert ix.remove(int(labels[0])) and not ix.remove(int(labels[0])) and not ix.remove(1)
    assert ix.deleted == 1
    assert ix.search(base[:1], k=1, ef=50)[0][0, 0] != labels[0]

    allowed = labels[::50]
    found, _ = ix.search(queries, k=5, ef=400, allowed=allowed)
    assert set(found.ravel().tolist()) <= set(allowed.tolist())


def test_save_and_load(tmp_path, data):
    base, queries = data
    ix = vecdb.Index(64, len(base), M=12, ef_construction=100)
    ix.add(base[:5000])
    path = str(tmp_path / "index.vecdb")
    ix.save(path)
    back = vecdb.Index.load(path)
    assert len(back) == 5000
    a, b = ix.search(queries, 10, 80), back.search(queries, 10, 80)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    with open(path, "r+b") as f:
        f.truncate(os.path.getsize(path) // 2)
    with pytest.raises(RuntimeError):
        vecdb.Index.load(path)


def test_sq8_quantisation(data, tmp_path):
    base, queries = data
    exact = vecdb.Index(64, len(base), metric="cosine")
    exact.add(base)
    truth = exact.brute_force(queries, 10)

    codes = vecdb.Index(64, len(base), metric="cosine", quantize="sq8")
    codes.add(base)
    both = vecdb.Index(64, len(base), metric="cosine", quantize="sq8", rerank=True)
    both.add(base)
    assert codes.quantized and not exact.quantized
    assert codes.vector_bytes * 4 == exact.vector_bytes       # one byte per dimension instead of four
    assert both.vector_bytes == codes.vector_bytes + exact.vector_bytes

    r_codes = recall(codes.search(queries, 10, ef=100)[0], truth)
    r_both = recall(both.search(queries, 10, ef=100)[0], truth)
    assert r_codes > 0.85 and r_both > 0.97 and r_both >= r_codes

    # Distances come back in the metric's own units, exact when re-ranked.
    d_exact = exact.search(queries[:20], 1, ef=200)[1][:, 0]
    assert np.allclose(both.search(queries[:20], 1, ef=200)[1][:, 0], d_exact, atol=1e-5)
    assert np.allclose(codes.search(queries[:20], 1, ef=200)[1][:, 0], d_exact, rtol=0.25, atol=0.01)

    path = str(tmp_path / "sq8.vecdb")
    codes.save(path)
    back = vecdb.Index.load(path)
    assert back.quantized and np.array_equal(back.search(queries, 10, 80)[0], codes.search(queries, 10, 80)[0])

    with pytest.raises(ValueError):
        vecdb.Index(64, 10, metric="ip", quantize="sq8")
    with pytest.raises(ValueError):
        vecdb.Index(64, 10, quantize="pq")


def test_filter_strategies_agree(data):
    base, queries = data
    ix = vecdb.Index(64, len(base))
    ix.add(base)
    for every in (3, 40, 2000):
        allowed = np.arange(0, len(base), every, dtype=np.uint64)
        scan, scan_d = ix.search(queries, 5, ef=200, allowed=allowed, filter_strategy="scan")
        graph, _ = ix.search(queries, 5, ef=200, allowed=allowed, filter_strategy="graph")
        auto, _ = ix.search(queries, 5, ef=200, allowed=allowed)
        assert set(scan.ravel().tolist()) <= set(allowed.tolist())
        assert np.all(np.diff(scan_d, axis=1) >= 0)
        # The scan is exact among the allowed vectors; compare it with numpy.
        sub = base[::every]
        d2 = ((queries[:20, None, :] - sub[None, :, :]) ** 2).sum(-1)
        want = np.sort(np.argsort(d2, axis=1)[:, :5] * every, axis=1)
        assert np.array_equal(np.sort(scan[:20], axis=1), want)
        assert recall(graph, scan) > 0.95 and recall(auto, scan) > 0.95
    with pytest.raises(ValueError):
        ix.search(queries, 5, allowed=np.arange(10, dtype=np.uint64), filter_strategy="magic")


def test_errors(data):
    base, _ = data
    ix = vecdb.Index(64, 10)
    with pytest.raises(ValueError):
        ix.add(base[:5, :32])
    with pytest.raises(ValueError):
        vecdb.Index(64, 10, metric="manhattan")
    ix.add(base[:10])
    with pytest.raises(ValueError, match="full"):
        ix.add(base[10:11])
    with pytest.raises(ValueError):
        ix.add(base[:1], np.array([3], dtype=np.uint64))  # label 3 already used
    assert ix.search(base[:2], k=20, ef=50)[0][0].tolist().count(-1) == 10  # fewer than k available
