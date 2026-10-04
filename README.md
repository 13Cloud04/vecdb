# vecdb

An HNSW approximate-nearest-neighbour index written from scratch in C++20,
with hand-written SIMD distance kernels, a parallel build, 8-bit scalar
quantisation, filter-aware search, and Python bindings.

Every retrieval-augmented system sits on an index like this one. I had used
them (Azure AI Search, FAISS) without knowing what happens inside, so I built
one and measured it against the two standard open-source implementations on
two public datasets.

```python
import numpy as np, vecdb

index = vecdb.Index(dim=1536, max_elements=200_000, metric="cosine", quantize="sq8")
index.add(vectors)                                        # float32 [n, 1536]; uses all cores
labels, distances = index.search(queries, k=10, ef=80)
labels, _ = index.search(queries, k=10, allowed=my_ids)   # only these labels may be returned
index.save("docs.vecdb")
```

## Results

Apple M5. All libraries use M=16, ef_construction=200. Queries run on **one
thread**; each library receives every query in a single call, so none pays
Python overhead per query. All indexes are built before timing starts, the
libraries take turns inside each measurement round, and the best round is
reported.

### 1. SIFT1M: 1,000,000 image descriptors, 128-d, L2

`python bench/fetch_sift.py && make py && python bench/bench_sift.py`

| ef | vecdb recall@10 | vecdb queries/sec | FAISS recall@10 | FAISS queries/sec | hnswlib recall@10 | hnswlib queries/sec |
|---|---|---|---|---|---|---|
| 10 | 0.7091 | 24,435 | 0.7157 | 27,670 | 0.7094 | 11,243 |
| 20 | 0.8397 | 17,871 | 0.8460 | 13,533 | 0.8388 | 7,398 |
| 40 | 0.9272 | 10,124 | 0.9336 | 7,946 | 0.9270 | 4,602 |
| 80 | **0.9746** | **5,317** | 0.9775 | 4,310 | 0.9746 | 2,563 |
| 160 | 0.9933 | 3,035 | 0.9937 | 2,164 | 0.9930 | 1,441 |
| 320 | 0.9981 | 1,613 | 0.9982 | 1,156 | 0.9983 | 806 |

### 2. DBpedia / OpenAI embeddings: 114,386 text embeddings, 1536-d, cosine

`python bench/fetch_dbpedia.py && python bench/bench_dbpedia.py`. Entities
embedded with text-embedding-ada-002; 1,000 held-out queries; ground truth by
exact scan.

| Index | Vector memory | ef=80 recall@10 | ef=80 queries/sec | ef=320 recall@10 | ef=320 queries/sec |
|---|---|---|---|---|---|
| vecdb float32 | 670 MB | 0.9792 | 1,795 | 0.9961 | 543 |
| **vecdb sq8** (1 byte per dimension) | **168 MB** | 0.9736 | 2,114 | 0.9870 | 656 |
| vecdb sq8 + exact re-rank | 838 MB | 0.9811 | 2,183 | 0.9962 | 571 |
| FAISS float32 | 670 MB | 0.9842 | 1,604 | 0.9960 | 522 |
| hnswlib float32 | 670 MB | 0.9774 | 499 | 0.9967 | 175 |

### How to read these

- **Correctness.** On both datasets the recall at every ef is within about
  0.01 of both libraries. A mistake in graph construction or search would show
  up as lower recall at the same ef, so this is the strongest evidence the
  algorithm is implemented correctly.
- **Against FAISS** (which has NEON kernels): on SIFT, vecdb is 12% slower at
  ef=10 and 23-40% faster from ef=20 up; on the 1536-d embeddings it is 3-26%
  faster. FAISS's recall is slightly higher at each ef (0.003-0.005 at
  ef=80), so at equal recall the two are close. Call it level.
- **Against hnswlib: 2-3.6x, and that needs a caveat.** hnswlib ships SSE and
  AVX kernels but no NEON one, so on Apple Silicon it runs a scalar loop (a
  profile of the benchmark shows it in `hnswlib::L2Sqr`). The gap is my SIMD
  kernel against their scalar fallback on this CPU. On x86 I would expect the
  two to be about level.
- **Against exact search:** a brute-force scan with the same SIMD kernel does
  72 queries/sec on SIFT1M and 48 on the embeddings. At ef=80 the index is
  74x and 37x faster.
- **Quantisation.** Storing one byte per dimension cuts vector memory 4x (the
  whole index from 687 MB to 184 MB) and is 18-21% *faster*, because a
  quantised vector is a quarter of the memory traffic. It costs 0.6 points of
  recall at ef=80 and 0.9 at ef=320. Keeping the floats to re-rank the final
  candidates recovers the recall but, in this implementation, also keeps the
  floats in RAM; the point of the mode is to show the loss is in the last
  ranking step, not in the graph traversal.

### 3. Filtered search: `python bench/bench_filter.py`

"Nearest neighbours, but only among these ids." SIFT1M, k=10, ef=100. The
graph strategy traverses the index and skips disallowed nodes when collecting
results; the scan strategy computes the distance to every allowed vector.

| Allowed | Graph recall | Graph queries/sec | Scan queries/sec (exact) | Auto picks |
|---|---|---|---|---|
| 50% (500,000) | 0.9924 | 2,071 | 50 | graph |
| 10% (100,000) | 0.9990 | 564 | 82 | graph |
| 5% (50,000) | 0.9992 | 327 | 227 | graph |
| 2% (20,000) | 1.0000 | 166 | 2,129 | scan |
| 1% (10,000) | 1.0000 | 97 | 4,137 | scan |
| 0.1% (1,000) | 1.0000 | 15 | 48,648 | scan |
| 0.01% (100) | 1.0000 | **2** | **322,823** | scan |

A graph index is the wrong tool for a selective filter: with 100 allowed
vectors it visits most of the million to find them. `search()` estimates both
costs (a scan is one distance per allowed vector; a filtered traversal visits
about ef / selectivity nodes) and switches at roughly 3% on this dataset. It
picks the faster strategy at every row above.

### 4. Other measurements

| | |
|---|---|
| SIFT1M search on 10 threads (ef=80) | 47,357 queries/sec |
| SIFT1M index size | 636 MB (488 MB of that is the vectors) |
| Save / load 1 M vectors | 0.5 s / 0.2-1.5 s (every link is validated on load) |
| Build, 10 threads | SIFT1M about 2 minutes; embeddings 30 s (FAISS 39 s, hnswlib 57 s) |

**SIMD kernels** (`make build/bench_kernels && ./build/bench_kernels`), squared
L2 distance, NEON with four accumulators against the plain loop:

| Dimensions | Scalar | NEON | Speedup |
|---|---|---|---|
| 64 | 20.1 ns | 5.5 ns | 3.6x |
| 128 | 56.0 ns | 14.3 ns | 3.9x |
| 384 | 235 ns | 50.0 ns | 4.7x |
| 768 | 603 ns | 100 ns | 6.0x |
| 1536 | 1363 ns | 208 ns | 6.5x |

## What measuring changed

1. **Four distances at a time.** On the 1536-d embeddings the first version
   was 20-40% slower than FAISS even though the single-distance kernel was
   as fast as it could be. A 6 KB vector makes the search memory-bound. The
   graph search now gathers a node's unvisited neighbours and computes their
   distances four at a time in one loop: four memory streams in flight, and
   each piece of the query loaded once instead of four times. Throughput at
   ef=80 went from 1,343 to 1,795 queries/sec and the build from 43 s to 30 s.
2. **Fair benchmarking is its own problem.** Measuring one library to
   completion and then the next gave numbers that moved by almost 2x between
   runs, depending on machine temperature and which cores the thread landed
   on. Building everything first and interleaving the libraries within each
   round fixed it.
3. **A filter can make an index slower than no index.** See the table above;
   the selectivity-aware switch exists because I measured 2 queries/sec.

## How it works

HNSW (Malkov & Yashunin, 2018) is a stack of proximity graphs. The bottom
layer contains every vector; each layer above contains a random ~1/M of the
layer below. A search starts at the top, walks greedily towards the query,
drops down a layer and repeats, then runs a best-first search on the bottom
layer that keeps the `ef` best candidates. It is a skip list generalised from
one dimension to many.

What is in `src/hnsw.hpp`:

- **Layout.** Vectors in one aligned block; bottom-layer links in one block of
  fixed-size slots `[count, n1, ..., n2M]`; upper-layer links allocated only
  for the nodes that have them. The inner loop reads two contiguous arrays.
- **Neighbour selection.** The paper's heuristic (Algorithm 4): a candidate is
  skipped if it is closer to an already-chosen neighbour than to the node
  itself. This preserves links between clusters; keeping simply the M closest
  does not.
- **Parallel build.** One spinlock byte per node guards its link list. No
  thread ever holds two node locks, so there is no lock-order to get wrong and
  no deadlock. Because another thread can add a back-link to a node that is
  still being inserted, a node merges its own links with whatever is already
  there instead of overwriting. Checked under ThreadSanitizer.
- **Scalar quantisation (SQ8).** The first batch fixes a value range (the
  0.1%-99.9% quantiles, so rare outliers do not waste code levels); each
  dimension is stored as one of 256 steps across it. Distances are computed
  directly on the bytes with a NEON kernel (absolute difference, widening
  multiply, pairwise accumulate) and converted back to the metric's units
  when returned. Cosine is handled by normalising first, after which L2 on
  the codes orders vectors the same way.
- **Filtered search.** A bitmask of allowed ids for graph traversal, the id
  list for scanning, and a cost estimate to choose between them.
- **Visited set.** A per-thread array of 16-bit epoch tags; starting a new
  search increments the epoch instead of clearing the array.
- **Deletes** are tombstones: the node still routes searches but is never
  returned.
- **Persistence.** A single binary file. `load()` treats it as untrusted
  input and checks every link is in range before accepting it; a corrupt or
  truncated file raises an error rather than causing an out-of-bounds read.
- **Metrics.** L2, inner product, cosine (vectors normalised on the way in).

## Tests

```
make test    # kernels vs scalar, recall on 3 metrics vs brute force, parallel build, deletes,
             # filter strategies, sq8 with and without re-ranking, save/load, corrupt file
make asan    # the same under AddressSanitizer + UBSan
make tsan    # the same under ThreadSanitizer
make py && pytest -q tests/test_python.py
```

## Limits

- SQ8 uses one value range for all dimensions. Per-dimension ranges or
  product quantisation would compress further or lose less.
- The re-rank mode keeps the floats in RAM, so it saves no memory; a real
  system would memory-map them from disk.
- `add()` and `search()` are each thread-safe but must not run at the same time.
- Capacity is fixed at construction; there is no resize.
- Deleted space is not reclaimed without a rebuild.
- The four-at-a-time kernels are NEON only; on x86 they fall back to four
  single calls.
- Two datasets, one machine. The filter crossover (about 3%) was measured on
  SIFT1M and will sit elsewhere for other dimensions and sizes.
