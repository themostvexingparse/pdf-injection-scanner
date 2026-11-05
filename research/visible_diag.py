"""Diagnose embedding flags on visible text of real documents.

For each PDF: every visible sentence the scanner would score, its probe score/tier, its similarity to
the nearest training injection, and its similarity to the rest of the document (an injection is usually
off-topic for the document it is planted in). Also simulates planted injections (deepset test) to see
how document-relative measures separate them from genuine document sentences.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from pdfscan import embedding  # noqa: E402
from pdfscan.__main__ import looks_like_code, visible_sentences  # noqa: E402
from pdfscan.differential import meaningful  # noqa: E402

sys.path.insert(0, HERE)
import local_docs as L  # noqa: E402  private documents, configured in research/local_docs.json

FILES = L.all_docs() + L.AUDIT_FIXTURES[:1]


def doc_sentences(pdf):
    r = subprocess.run([sys.executable, "-m", "pdfscan", pdf, "--quiet", "--include-raw", "--semantic", "regex"],
                       cwd=ROOT, capture_output=True)
    js = json.loads(r.stdout.decode("utf-8"))
    sents = []
    for p in js["pages"]:
        sents += [s for s in visible_sentences(p.get("safe_text", "")) if meaningful(s) and not looks_like_code(s)]
    return list(dict.fromkeys(sents))


def main():
    sig = embedding.get()
    thr = sig.thr
    inj = pd.read_parquet(glob.glob(os.path.join(HERE, "data", "deepset", "data", "test-*.parquet"))[0])
    inj = [t for t, l in zip(inj.text, inj.label) if l == 1]
    Vi = sig._embed(inj)
    si = Vi @ sig.w + sig.b
    out = {}
    for pdf in FILES:
        S = doc_sentences(pdf)
        if not S:
            continue
        V = sig._embed(S)
        s = V @ sig.w + sig.b
        # document-relative: mean cosine to the k most similar OTHER sentences of the same document
        sim = V @ V.T
        np.fill_diagonal(sim, -1)
        k = min(5, len(S) - 1)
        ctx = np.sort(sim, axis=1)[:, -k:].mean(1)
        # the same measure for planted injections (each compared against this document's sentences)
        ctx_inj = np.sort(Vi @ V.T, axis=1)[:, -k:].mean(1)
        near = sig.ex_vec @ V.T
        name = os.path.basename(pdf)
        flagged = np.where(s > thr["moderate"])[0]
        out[name] = {
            "sentences": len(S),
            "moderate": int(np.sum(s > thr["moderate"])), "strong": int(np.sum(s > thr["strong"])),
            "expected_false_flags_at_1pct": round(0.01 * len(S), 1),
            "doc_context_sim_median": round(float(np.median(ctx)), 3),
            "planted_injection_context_sim_median": round(float(np.median(ctx_inj)), 3),
            "flagged": [{"score": round(float(s[i]), 2), "context_sim": round(float(ctx[i]), 3),
                         "nearest_injection_cos": round(float(near[:, i].max()), 3),
                         "sentence": S[i][:120]} for i in flagged[np.argsort(-s[flagged])]],
        }
        # how well does context similarity separate planted injections from this document's sentences?
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.zeros(len(S)), np.ones(len(inj))]
        out[name]["auc_probe_only"] = round(float(roc_auc_score(y, np.r_[s, si])), 3)
        out[name]["auc_offtopic_only"] = round(float(roc_auc_score(y, -np.r_[ctx, ctx_inj])), 3)
        out[name]["auc_probe_minus_context"] = round(float(roc_auc_score(y, np.r_[s - 4 * ctx, si - 4 * ctx_inj])), 3)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
