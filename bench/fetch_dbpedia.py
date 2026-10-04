"""Download 3 shards (~115,000 vectors) of DBpedia entities embedded with
OpenAI text-embedding-ada-002 (1536-d). Dataset: KShivendu/dbpedia-entities-openai-1M (MIT).
Writes data/dbpedia/base.npy and queries.npy as float32.
"""
import pathlib
import urllib.request

import numpy as np
import pyarrow.parquet as pq

SHARDS = ["train-00000-of-00026-3c7b99d1c7eda36e.parquet", "train-00001-of-00026-2b24035a6390fdcb.parquet",
          "train-00002-of-00026-b05ce48965853dad.parquet"]
URL = "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/resolve/main/data/"
out = pathlib.Path(__file__).resolve().parent.parent / "data" / "dbpedia"
out.mkdir(parents=True, exist_ok=True)

if not (out / "base.npy").exists():
    parts = []
    for name in SHARDS:
        path = out / name
        if not path.exists():
            print("downloading", name)
            urllib.request.urlretrieve(URL + name, path)
        col = pq.read_table(path, columns=["openai"]).column("openai")
        parts.append(np.stack(col.to_numpy(zero_copy_only=False)).astype(np.float32))
    vectors = np.concatenate(parts)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(vectors))
    np.save(out / "queries.npy", vectors[order[:1000]])   # held out: never inserted into the index
    np.save(out / "base.npy", vectors[order[1000:]])
    for name in SHARDS:
        (out / name).unlink()
print("ready:", np.load(out / "base.npy", mmap_mode="r").shape, "base,",
      np.load(out / "queries.npy", mmap_mode="r").shape, "queries")
