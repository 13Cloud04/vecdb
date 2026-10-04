# Defending vecdb in an interview

Read `src/hnsw.hpp` top to bottom (about 750 lines); `add()` and
`search_layer()` are the two functions that matter. Then read
`src/distance.hpp`.

## The 30-second pitch

"Every RAG system sits on a vector index, and I had only ever called one. So I
wrote HNSW from scratch in C++ with NEON SIMD, a lock-per-node parallel build,
8-bit quantisation and Python bindings, and benchmarked it against hnswlib and
FAISS at the same parameters on SIFT1M and on 114,000 OpenAI text embeddings.
The recall curves match theirs, which is how I know the algorithm is right;
single-thread speed is level with FAISS; and quantisation cuts memory 4x for
about half a point of recall."

It ties directly to the KPMG internship: there you used Azure AI Search for
retrieval; here you built the thing underneath it.

## Questions you will get

**What problem does HNSW solve?**
Finding the nearest vectors to a query without comparing against all of them.
Exact search over a million 128-d vectors takes about 14 ms per query on one
core here; HNSW answers in a fraction of a millisecond by visiting a few
thousand vectors, at the cost of occasionally missing a true neighbour.

**How does it work?**
A graph where each vector links to some near neighbours. To search, start
somewhere and keep moving to whichever neighbour is closer to the query. To
make that fast, stack several graphs: the top one has few nodes and long
jumps, each lower one is denser, the bottom one has everything. It is a skip
list generalised to many dimensions.

**What are M, ef_construction and ef?**
M: links per node (2M on the bottom layer). More links, better recall, more
memory. ef_construction: how many candidates are considered when inserting;
higher builds a better graph more slowly. ef: candidates kept during a query;
the runtime dial between speed and recall.

**How is a node's level chosen?**
`floor(-ln(uniform) * 1/ln(M))`. Geometric: each level has about 1/M as many
nodes as the one below.

**Explain the neighbour-selection heuristic.**
Do not simply keep the M closest. Go through candidates closest first and
skip any that is closer to an already-selected neighbour than to the node
itself, because that direction is already covered. This keeps some long links
between clusters; with "M closest" a tight cluster links only to itself and
the search cannot get out.

**How does the parallel build avoid deadlock?**
Each node has a one-byte spinlock guarding its link list. No thread ever holds
two node locks at once: it locks a node, copies or updates its list, unlocks,
then moves on. Since nobody waits while holding, there is no cycle. The
consequence is that another thread may add a back-link to my new node before I
write its list, so I merge with what is there instead of overwriting.

**Why a spinlock rather than std::mutex?**
Size. `std::mutex` is 64 bytes on macOS; a million of them is 64 MB. The
critical section is a few dozen nanoseconds, so spinning is cheaper than
sleeping.

**Why four accumulators in the SIMD loop?**
A single running sum makes each addition wait for the previous one (a
dependency chain of 3-4 cycles per step). Four independent sums let the CPU
overlap them. Without `-ffast-math` the compiler is not allowed to do this
reordering itself, because float addition is not associative.

**Why was hnswlib slower in your benchmark?**
hnswlib ships SSE and AVX kernels but no NEON one, so on Apple Silicon it runs
the scalar loop. The fair comparison on this machine is FAISS, which has NEON,
and there the two are close. On an x86 machine I would expect hnswlib to be
about level. Say this yourself before the interviewer works it out.

**What is scalar quantisation and what did it cost?**
Store each dimension as one byte instead of a four-byte float: pick a value
range from the data, divide it into 256 steps, keep the step number. Vector
memory drops 4x. On the OpenAI embeddings recall fell from 0.979 to 0.974 at
ef=80, and search got about 18% faster because each vector is a quarter of
the memory traffic. Distances are computed directly on the bytes.

**Why clip the range at the 0.1% and 99.9% quantiles?**
A few extreme values would stretch the range and make every step coarser for
all the ordinary values. Clipping them costs a little error on rare values and
buys precision everywhere else.

**How does cosine work on quantised vectors?**
I normalise vectors to unit length first. For unit vectors, squared L2
distance equals 2 minus 2 times the cosine, so ordering by L2 on the codes is
ordering by cosine. That is also why SQ8 refuses raw inner product: without
normalisation the equivalence does not hold.

**Why compute four distances at once?**
A 1536-d vector is 6 KB. Computing one distance is mostly waiting for memory,
and the single-vector kernel cannot go faster than the memory arrives. Doing
four neighbours in one loop keeps four streams of memory requests in flight
and loads each piece of the query once. It took the embedding benchmark from
1,343 to 1,795 queries a second, which closed the gap to FAISS. FAISS does
the same thing; I found out why I was slower by measuring, then reading.

**How do you know the implementation is correct?**
Recall against exact brute-force search, on three metrics; and on SIFT1M the
recall at every ef is within about 0.01 of hnswlib's and FAISS's at the same
parameters, on two datasets. A bug in the graph would show up as lower recall at equal ef.

**How did you make the benchmark fair?**
Same M and ef_construction; one thread for queries; all queries passed in one
call so no library pays Python overhead per query; all indexes built before
timing; libraries take turns within each round and I report the best round,
because I saw the numbers shift with machine temperature when I ran them one
after another.

**How do deletes work?**
Tombstones. The node stays in the graph as a stepping stone but is never
returned. Actually unlinking nodes leaves holes and hurts recall. The cost is
that space is not reclaimed until a rebuild.

**How does filtered search work, and when does it break down?**
A bitmask of allowed ids; disallowed nodes are still traversed but not
returned. If the filter keeps few nodes the search wanders through most of
the graph to find ten allowed ones: with 100 allowed out of a million I
measured 2 queries a second. Scanning just those 100 vectors is exact and did
322,000 a second. So search() estimates both costs, one distance per allowed
vector against roughly ef divided by selectivity node visits, and switches.
On SIFT1M the crossover is near 3% and it picks the faster one at every
selectivity I tested.

**Why validate the file on load?**
An index file is input. A link that points past the end of the array would be
an out-of-bounds read on the first search. `load()` checks every link.

**What would you add next?**
Product quantisation for 16-32x compression, memory-mapped floats so re-ranking
costs no RAM, concurrent search during insert, AVX2 versions of the batched
kernels, and filtering during traversal (as in ACORN) for the middle range of
selectivities where neither strategy is good.

## Things to try yourself

1. Replace `select_neighbors` with "take the M closest" and rerun the recall
   test on clustered data.
2. Remove the four-accumulator trick (use one) and rerun `build/bench_kernels`.
3. Build with `threads=1` and `threads=10` and compare recall.
