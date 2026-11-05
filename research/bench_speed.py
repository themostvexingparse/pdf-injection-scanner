"""Speed + fidelity benchmark for bge-small-en-v1.5 on CPU.

Measures: ONNX-vs-PyTorch output agreement, load time, single-sentence latency (short/medium/max
length), batch throughput and thread scaling, dynamic int8 quantisation (speed and embedding drift),
and the cost of embedding every sentence of real documents.
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time

import fitz
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedder import MODEL_DIR, ROOT, Embedder  # noqa: E402

FP32 = os.path.join(MODEL_DIR, "onnx", "model.onnx")
INT8 = os.path.join(MODEL_DIR, "onnx", "model_int8.onnx")
import local_docs as L  # noqa: E402  private documents, configured in research/local_docs.json

DOCS = L.docs("train") or L.AUDIT_FIXTURES


def sentences_from(path: str) -> list[str]:
    text = " ".join(p.get_text() for p in fitz.open(path))
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if len(p.split()) >= 3]


def timeit(fn, n: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    res: dict = {}
    sents = []
    for d in DOCS:
        s = sentences_from(d)
        res[f"sentences_{os.path.basename(d)}"] = len(s)
        sents += s

    # ---------------------------------------------------------------- quantised model
    if not os.path.exists(INT8):
        from onnxruntime.quantization import QuantType, quantize_dynamic
        t = time.perf_counter()
        quantize_dynamic(FP32, INT8, weight_type=QuantType.QInt8)
        res["int8_quantize_s"] = round(time.perf_counter() - t, 1)
    res["size_mb"] = {"fp32": round(os.path.getsize(FP32) / 2**20, 1),
                      "int8": round(os.path.getsize(INT8) / 2**20, 1)}

    e32, e8 = Embedder(FP32), Embedder(INT8)
    res["load_s"] = {"onnx_fp32": round(e32.load_s, 3), "onnx_int8": round(e8.load_s, 3)}
    lens = [len(e32.tok.encode(s).ids) for s in sents]
    res["token_length"] = {"median": int(np.median(lens)), "p90": int(np.percentile(lens, 90)),
                           "max": int(max(lens)), "truncated_at_128": int(sum(l >= 128 for l in lens))}

    # ---------------------------------------------------------------- fidelity
    import torch
    from transformers import AutoModel, AutoTokenizer
    t = time.perf_counter()
    tm = AutoModel.from_pretrained(MODEL_DIR).eval()
    tt = AutoTokenizer.from_pretrained(MODEL_DIR)
    res["load_s"]["torch_fp32"] = round(time.perf_counter() - t, 3)

    def torch_encode(texts):
        with torch.no_grad():
            b = tt(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = tm(**b).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(h, dim=-1).numpy()

    sample = sents[:200]
    a, b, c = e32.encode(sample), e8.encode(sample), torch_encode(sample)
    cos = lambda x, y: np.sum(x * y, axis=1)  # noqa: E731
    res["fidelity_cosine"] = {
        "onnx_fp32_vs_torch": {"mean": float(cos(a, c).mean()), "min": float(cos(a, c).min())},
        "onnx_int8_vs_fp32": {"mean": float(cos(a, b).mean()), "min": float(cos(a, b).min())},
    }
    # does int8 preserve neighbourhoods? (rank agreement of nearest neighbours)
    sa, sb = a @ a.T, b @ b.T
    np.fill_diagonal(sa, -1); np.fill_diagonal(sb, -1)
    res["int8_top1_neighbour_agreement"] = float(np.mean(sa.argmax(1) == sb.argmax(1)))

    # ---------------------------------------------------------------- latency (batch of 1)
    by_len = sorted(sents, key=lambda s: len(e32.tok.encode(s).ids))
    short = by_len[len(by_len) // 10]
    medium = by_len[len(by_len) // 2]
    longest = " ".join(by_len[-6:])  # hits the 128-token cap
    lat = {}
    for name, emb in (("onnx_fp32", e32), ("onnx_int8", e8)):
        lat[name] = {k: round(1000 * timeit(lambda s=s: emb.encode([s]), 50), 2)
                     for k, s in (("short", short), ("medium", medium), ("max128", longest))}
    lat["torch_fp32"] = {k: round(1000 * timeit(lambda s=s: torch_encode([s]), 30), 2)
                         for k, s in (("short", short), ("medium", medium), ("max128", longest))}
    lat["tokens"] = {"short": len(e32.tok.encode(short).ids), "medium": len(e32.tok.encode(medium).ids),
                     "max128": min(128, len(e32.tok.encode(longest).ids))}
    res["latency_ms_batch1"] = lat

    # ---------------------------------------------------------------- throughput + threads
    corpus = by_len  # length-sorted batches minimise padding (what a real pipeline would do)
    thr = {}
    for threads in (1, 2, 4, 8):
        for name, path in (("fp32", FP32), ("int8", INT8)):
            emb = Embedder(path, threads=threads)
            emb.encode(corpus[:64])
            t = time.perf_counter()
            emb.encode(corpus, batch=32)
            dt = time.perf_counter() - t
            thr[f"{name}_threads{threads}"] = {"sentences_per_s": round(len(corpus) / dt, 1),
                                               "total_s": round(dt, 3)}
    t = time.perf_counter()
    for i in range(0, len(corpus), 32):
        torch_encode(corpus[i:i + 32])
    thr["torch_fp32_default_threads"] = {"sentences_per_s": round(len(corpus) / (time.perf_counter() - t), 1)}
    res["throughput_batch32"] = thr
    res["corpus_sentences"] = len(corpus)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_speed.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
