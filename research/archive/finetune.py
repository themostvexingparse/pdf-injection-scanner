"""Light fine-tuning of bge-small-en-v1.5 (encoder + linear head) vs. frozen embeddings + logistic probe.

Same data and the same threshold procedure for both:
  train = deepset train + training-PDF sentences + Wikipedia (6000) + generated reader instructions
  85/15 stratified split; threshold = 1% FPR on the validation split's document-domain benign text
  test  = deepset test, held-out PDFs, hand-written hard negatives, Wikipedia test articles, probes
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
# PyArrow must be loaded before torch on Windows: importing pyarrow.dataset after torch crashes
# with an access violation (native DLL conflict).
import pyarrow.dataset  # noqa: F401
import pyarrow.parquet  # noqa: F401
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_embed as E  # noqa: E402
from embedder import MODEL_DIR, ROOT, Embedder  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(8)


def build_data():
    import glob
    import pandas as pd
    load = lambda s: pd.read_parquet(glob.glob(os.path.join(HERE, "data", "deepset", "data", f"{s}-*.parquet"))[0])  # noqa: E731
    dtr, dte = load("train"), load("test")
    doc_tr = [s for d in E.TRAIN_DOCS for s in E.doc_sentences(os.path.join(ROOT, d))]
    wtr = [E.normalize(x) for x in E.wiki_sentences("train", 6000, 0)]
    gen = [E.normalize(x) for x in E.benign_generate()]
    texts = list(dtr.text) + doc_tr + wtr + gen
    labels = list(dtr.label) + [0] * (len(doc_tr) + len(wtr) + len(gen))
    docdom = [False] * len(dtr) + [True] * (len(doc_tr) + len(wtr) + len(gen))
    tests = {
        "deepset": (list(dte.text), list(dte.label)),
        "doc": ([s for d in E.TEST_DOCS for s in E.doc_sentences(d)], None),
        "hard": (E.HARD_NEGATIVES, None),
        "wiki": ([E.normalize(x) for x in E.wiki_sentences("test", 2000, 1)], None),
    }
    return texts, np.array(labels), np.array(docdom), tests


class Clf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_DIR)
        self.drop = torch.nn.Dropout(0.1)
        self.head = torch.nn.Linear(384, 1)

    def forward(self, **b):
        h = self.enc(**b).last_hidden_state[:, 0]
        return self.head(self.drop(h)).squeeze(-1)


def predict(model, tok, texts, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt")
            out.append(model(**b).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def thr_at(scores, fpr=0.01):
    neg = np.sort(scores)
    return float(neg[int(np.ceil((1 - fpr) * len(neg))) - 1]) + 1e-9


def report(name, score_fn, thr, tests, probe_texts):
    r = {}
    s = score_fn(tests["deepset"][0]); y = np.array(tests["deepset"][1])
    r["deepset_tpr"] = round(float(np.mean(s[y == 1] > thr)), 3)
    r["deepset_benign_fpr"] = round(float(np.mean(s[y == 0] > thr)), 3)
    all_s, all_y = [s], [y]
    for k in ("doc", "hard", "wiki"):
        sk = score_fn(tests[k][0])
        r[f"{k}_fpr"] = round(float(np.mean(sk > thr)), 3)
        all_s.append(sk); all_y.append(np.zeros(len(sk), int))
    r["auc_all"] = round(float(roc_auc_score(np.concatenate(all_y), np.concatenate(all_s))), 4)
    r["probes"] = "".join("Y" if v > thr else "n" for v in score_fn(probe_texts)) if probe_texts else ""
    print(f"{name:10} AUC {r['auc_all']:.4f} | TPR {r['deepset_tpr']:.2f} | FPR doc {r['doc_fpr']:.3f} "
          f"hard {r['hard_fpr']:.3f} wiki {r['wiki_fpr']:.3f} chat {r['deepset_benign_fpr']:.3f} | probes {r['probes']}")
    return r


def main(epochs=2, bs=32, pos_weight=8.0):
    texts, y, docdom, tests = build_data()
    idx = np.arange(len(texts))
    tr, va = train_test_split(idx, test_size=0.15, stratify=y, random_state=0)
    probe_texts = [s for _, s in E.probes()]
    R = {"sizes": {"train": len(tr), "val": len(va), "pos_train": int(y[tr].sum())}}

    # ---------------- baseline: frozen embeddings + logistic probe, same split, same threshold rule
    emb = Embedder(threads=8)
    emb.enable_cache(os.path.join(HERE, "data", "embcache_fp32.pkl"))
    X = emb.encode(texts)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(X[tr], y[tr])
    vneg = va[(y[va] == 0) & docdom[va]]
    thr_lr = thr_at(lr.decision_function(X[vneg]))
    R["frozen_probe"] = report("frozen+LR", lambda t: lr.decision_function(emb.encode(t)), thr_lr, tests, probe_texts)
    emb.save_cache()

    # ---------------- fine-tune encoder + head
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = Clf()
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": 2e-5},
                             {"params": model.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    tr_texts, tr_y = [texts[i] for i in tr], y[tr]
    t0 = time.perf_counter()
    steps = 0
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(ep).permutation(len(tr_texts))
        tot = 0.0
        for i in range(0, len(order), bs):
            bi = order[i:i + bs]
            b = tok([tr_texts[j] for j in bi], padding=True, truncation=True, max_length=128, return_tensors="pt")
            loss = lossf(model(**b), torch.tensor(tr_y[bi], dtype=torch.float32))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); steps += 1
            if steps % 50 == 0:
                print(f"  epoch {ep + 1} step {steps} loss {tot / ((i // bs) + 1):.4f} "
                      f"elapsed {time.perf_counter() - t0:.0f}s", flush=True)
    R["train_seconds"] = round(time.perf_counter() - t0, 1)
    vs = predict(model, tok, [texts[i] for i in vneg])
    thr_ft = thr_at(vs)
    R["finetuned"] = report("fine-tuned", lambda t: predict(model, tok, t), thr_ft, tests, probe_texts)
    torch.save(model.state_dict(), os.path.join(HERE, "data", "finetuned_bge_small.pt"))
    with open(os.path.join(HERE, "results_finetune.json"), "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps({k: v for k, v in R.items() if k in ("sizes", "train_seconds")}))


if __name__ == "__main__":
    main()
