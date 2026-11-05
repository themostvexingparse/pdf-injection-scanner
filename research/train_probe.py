"""Train the shipped embedding probe -> pdfscan/models/probe.json (+ exemplars.npz).

Probe: logistic regression on frozen bge-small-en-v1.5 (fp32 ONNX, CLS, L2-normalised) embeddings.
Training data (deepset TEST split is held out for regression evaluation):
  injections + chat benign : deepset/prompt-injections train (Apache-2.0)
  document benign          : sentences of the training PDFs, WikiText-103 (CC BY-SA 3.0),
                             generated instructions-to-humans and layout fragments (benign_gen.py)
Thresholds from 5-fold out-of-fold scores on DOCUMENT-domain benign text:
  moderate = 1% false positives, strong = 0.1% false positives.
Exemplars (for explanations only, never for decisions): embeddings of the training injections,
stored with dataset row ids, not text.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_embed as E  # noqa: E402
from embedder import MODEL_DIR, ROOT, Embedder  # noqa: E402

OUT_DIR = os.path.join(ROOT, "pdfscan", "models")
ONNX = os.path.join(MODEL_DIR, "onnx", "model.onnx")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def threshold(scores, fpr):
    s = np.sort(scores)
    return float(s[int(np.ceil((1 - fpr) * len(s))) - 1])


def main():
    emb = Embedder(ONNX, threads=8)
    emb.enable_cache(os.path.join(HERE, "data", "embcache_fp32.pkl"))
    load = lambda s: pd.read_parquet(glob.glob(os.path.join(HERE, "data", "deepset", "data", f"{s}-*.parquet"))[0])  # noqa: E731
    dtr, dte = load("train"), load("test")
    doc_tr = [s for d in E.TRAIN_DOCS for s in E.doc_sentences(os.path.join(ROOT, d))]
    wiki = [E.normalize(x) for x in E.wiki_sentences("train", 6000, 0)]
    gen = [E.normalize(x) for x in E.benign_generate()]

    texts = list(dtr.text) + doc_tr + wiki + gen
    y = np.array(list(dtr.label) + [0] * (len(doc_tr) + len(wiki) + len(gen)))
    docdom = np.array([False] * len(dtr) + [True] * (len(doc_tr) + len(wiki) + len(gen)))
    X = emb.encode(texts)

    make = lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)  # noqa: E731
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        oof[va] = make().fit(X[tr], y[tr]).decision_function(X[va])
    neg = oof[(y == 0) & docdom]
    thr_mod, thr_strong = threshold(neg, 0.01), threshold(neg, 0.001)
    clf = make().fit(X, y)

    # held-out check on the deepset test split
    ste = clf.decision_function(emb.encode(list(dte.text)))
    yte = dte.label.values
    held_out = {"deepset_test_tpr_moderate": round(float(np.mean(ste[yte == 1] > thr_mod)), 3),
                "deepset_test_tpr_strong": round(float(np.mean(ste[yte == 1] > thr_strong)), 3),
                "oof_injection_tpr_moderate": round(float(np.mean(oof[y == 1] > thr_mod)), 3),
                "oof_injection_tpr_strong": round(float(np.mean(oof[y == 1] > thr_strong)), 3),
                "oof_doc_fpr_moderate": round(float(np.mean(neg > thr_mod)), 4),
                "oof_doc_fpr_strong": round(float(np.mean(neg > thr_strong)), 4)}
    emb.save_cache()

    os.makedirs(OUT_DIR, exist_ok=True)
    probe = {
        "format": "pdfscan-embedding-probe/1",
        "created": datetime.date.today().isoformat(),
        "embedding_model": {"name": "BAAI/bge-small-en-v1.5", "onnx_sha256": sha256(ONNX), "dim": 384,
                            "pooling": "cls", "normalize": True, "max_tokens": 128, "precision": "fp32"},
        "weights": [round(float(w), 7) for w in clf.coef_[0]],
        "bias": round(float(clf.intercept_[0]), 7),
        "thresholds": {"moderate": round(thr_mod, 5), "strong": round(thr_strong, 5)},
        "calibration": "5-fold out-of-fold decision scores on document-domain benign text; "
                       "moderate = 1% FPR, strong = 0.1% FPR",
        "training_data": [
            {"name": "deepset/prompt-injections (train split)", "license": "Apache-2.0",
             "injection": int(dtr.label.sum()), "benign": int((dtr.label == 0).sum())},
            {"name": "sentences of local training PDFs", "benign": len(doc_tr)},
            {"name": "Salesforce/wikitext wikitext-103-raw-v1 (train, sampled)", "license": "CC BY-SA 3.0",
             "benign": len(wiki)},
            {"name": "generated instructions-to-humans + layout fragments (research/benign_gen.py)",
             "benign": len(gen)},
        ],
        "held_out_metrics": held_out,
    }
    with open(os.path.join(OUT_DIR, "probe.json"), "w") as f:
        json.dump(probe, f, indent=1)
    pos = np.where(y == 1)[0]
    np.savez_compressed(os.path.join(OUT_DIR, "exemplars.npz"),
                        vectors=X[pos].astype(np.float16),
                        ids=np.array([f"deepset/prompt-injections:train:{i}" for i in pos]))
    print(json.dumps({"thresholds": probe["thresholds"], "held_out": held_out, "exemplars": len(pos)}, indent=1))


if __name__ == "__main__":
    main()
