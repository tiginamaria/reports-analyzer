"""Embedding throughput benchmark for the ingestion bottleneck.

`pipeline._embed` calls `encode` once per 32 chunks, and sentence-transformers sorts by length
only within one call, so every batch is padded to its longest chunk (bge-m3 accepts up to 8192
tokens, and table chunks can be long). This compares the current scheme with one sorted call,
fp16 weights and a capped sequence length, and checks the vectors stay the same (cosine to
the current vectors).

    uv run python -m eval.embed_bench --chunks 512
"""

import argparse
import random
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from app.db import SessionLocal
from app.ingestion.chunker import contextual_header
from app.models import Chunk, Report
from app.retrieval.embedder import get_embedder
from eval.common import write_json

BATCH = 32


def texts_sample(n: int) -> list[str]:
    with SessionLocal() as db:
        rows = db.query(Chunk, Report).join(Report, Report.id == Chunk.report_id).all()
    random.Random(0).shuffle(rows)
    return [
        f"{contextual_header(r.company, r.fiscal_year, c.page)}\n{c.content}" for c, r in rows[:n]
    ]


def run(model: SentenceTransformer, texts: list[str], per_batch: bool) -> tuple[float, np.ndarray]:
    start = time.perf_counter()
    if per_batch:
        parts = [
            model.encode(texts[i : i + BATCH], batch_size=BATCH, normalize_embeddings=True)
            for i in range(0, len(texts), BATCH)
        ]
        vectors = np.concatenate(parts)
    else:
        vectors = model.encode(texts, batch_size=BATCH, normalize_embeddings=True)
    return time.perf_counter() - start, vectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=int, default=512)
    args = parser.parse_args()
    texts = texts_sample(args.chunks)
    model = get_embedder().model
    tokens = [len(model.tokenizer(t)["input_ids"]) for t in texts]
    print(f"{len(texts)} chunks, tokens p50={int(np.median(tokens))} max={max(tokens)}")
    model.encode(texts[:BATCH])  # warm-up

    results = {}
    seconds, reference = run(model, texts, per_batch=True)
    results["current (32 per call)"] = seconds
    seconds, vectors = run(model, texts, per_batch=False)
    results["one call, length-sorted"] = seconds
    similarity = {"one call, length-sorted": float(np.min(np.sum(reference * vectors, axis=1)))}

    original_max = model.max_seq_length
    model.max_seq_length = 1024
    seconds, vectors = run(model, texts, per_batch=False)
    results["one call, max 1024 tokens"] = seconds
    similarity["one call, max 1024 tokens"] = float(np.min(np.sum(reference * vectors, axis=1)))
    model.max_seq_length = original_max

    model.half()
    seconds, vectors = run(model, texts, per_batch=False)
    results["one call, fp16"] = seconds
    similarity["one call, fp16"] = float(np.min(np.sum(reference * vectors, axis=1)))

    out = {
        "chunks": len(texts),
        "tokens_p50": int(np.median(tokens)),
        "tokens_max": max(tokens),
        "over_1024_tokens": sum(t > 1024 for t in tokens),
        "chunks_per_second": {k: round(len(texts) / v, 1) for k, v in results.items()},
        "min_cosine_to_current": {k: round(v, 5) for k, v in similarity.items()},
    }
    print(out)
    print(write_json("embed_bench.json", out))


if __name__ == "__main__":
    main()
