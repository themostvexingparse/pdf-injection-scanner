"""Accuracy study: can bge-small-en-v1.5 embeddings separate prompt injections from document text?

Data
  injections + chatbot-style benign: deepset/prompt-injections (Apache-2.0), official train/test split
  benign document sentences: real PDFs; TRAIN docs and TEST docs are disjoint files
  hard negatives: instruction-looking benign sentences (research/hard_negatives.py), test only
  probes: novel spans the scanner extracts from the audit fixtures + configured private documents

Methods (all thresholds frozen from 5-fold CV on train at 1% benign FPR)
  knn_max      max cosine to the injection database (the "vector DB + threshold" idea)
  knn_margin   max cos(injection DB) - max cos(benign DB)
  knn5_margin  mean top-5 cos(injection) - mean top-5 cos(benign)
  centroid     projection on mean(injection) - mean(benign)  (a single "direction")
  logreg       logistic-regression probe (learned direction)
  best_dim     the single most discriminative embedding dimension
  regex        the scanner's current pattern classifier (no training)
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import sys
import time

import fitz
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from embedder import MODEL_DIR, ROOT, Embedder  # noqa: E402
from hard_negatives import HARD_NEGATIVES  # noqa: E402
from benign_gen import generate as benign_generate  # noqa: E402
from pdfscan import semantic  # noqa: E402

GER = re.compile(r"\b(und|der|die|das|ist|nicht|ich|sie|du|ein|eine|mit|auf|für|wie|vergiss|alle|bitte)\b", re.I)
import local_docs as L  # noqa: E402  private documents, configured in research/local_docs.json

TRAIN_DOCS = L.docs("train")
TEST_DOCS = L.docs("test")
PROBE_FILES = L.AUDIT_FIXTURES + L.docs("probe")


def normalize(t: str) -> str:
    """What the scanner would do before embedding: collapse dot leaders / rules / bullets, whitespace."""
    t = re.sub(r"([.…·_\-=*]){3,}", " ", t)
    t = re.sub(r"[•▪●·]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def doc_sentences(path: str) -> list[str]:
    try:
        text = " ".join(p.get_text() for p in fitz.open(path))
    except Exception:
        return []
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?:])\s+(?=[A-Z0-9])", text)
    out = []
    for p in parts:
        p = p.strip()
        p = normalize(p)
        if len(p.split()) >= 4 and "museum room monitor" not in p:  # that one is a probe
            out.append(p[:600])
    return list(dict.fromkeys(out))


def probes() -> list[tuple[str, str]]:
    """Novel hidden spans as the scanner extracts them (not retyped here)."""
    out = []
    for f in PROBE_FILES:
        r = subprocess.run([sys.executable, "-m", "pdfscan", os.path.join(ROOT, f), "--quiet", "--ocr", "off"],
                           cwd=ROOT, capture_output=True)
        js = json.loads(r.stdout.decode("utf-8"))
        for fd in js["findings"]:
            for sp in (fd.get("details") or {}).get("novel_spans", []):
                out.append((f, sp))
    return out


# ------------------------------------------------------------------ methods
class Knn:
    def __init__(self, mode, k=1):
        self.mode, self.k = mode, k

    def fit(self, X, y):
        self.P, self.N = X[y == 1], X[y == 0]
        return self

    def score(self, X):
        sp = X @ self.P.T
        top = lambda s: np.sort(s, axis=1)[:, -self.k:].mean(1)  # noqa: E731
        if self.mode == "max":
            return top(sp)
        return top(sp) - top(X @ self.N.T)


class Centroid:
    def fit(self, X, y):
        d = X[y == 1].mean(0) - X[y == 0].mean(0)
        self.d = d / np.linalg.norm(d)
        return self

    def score(self, X):
        return X @ self.d


class Probe:
    def __init__(self, C=1.0, dims=None):
        self.C, self.dims = C, dims

    def fit(self, X, y):
        Xd = X if self.dims is None else X[:, self.dims]
        self.m = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000).fit(Xd, y)
        return self

    def score(self, X):
        return self.m.decision_function(X if self.dims is None else X[:, self.dims])


class BestDim:
    def fit(self, X, y):
        aucs = np.array([roc_auc_score(y, X[:, j]) for j in range(X.shape[1])])
        self.j = int(np.argmax(np.abs(aucs - 0.5)))
        self.sign = 1 if aucs[self.j] >= 0.5 else -1
        self.auc = float(aucs[self.j])
        return self

    def score(self, X):
        return self.sign * X[:, self.j]


METHODS = {
    "knn_max": lambda: Knn("max"),
    "knn_margin": lambda: Knn("margin"),
    "knn5_margin": lambda: Knn("margin", 5),
    "centroid": lambda: Centroid(),
    "logreg": lambda: Probe(1.0),
    "best_dim": lambda: BestDim(),
}


def cv_threshold(make, X, y, fpr=0.01, folds=5, seed=0, calib=None):
    """Threshold giving <= fpr on benign, from out-of-fold scores (no test data involved).
    calib: boolean mask of the benign examples to calibrate on (document-domain text)."""
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        oof[va] = make().fit(X[tr], y[tr]).score(X[va])
    neg_mask = (y == 0) if calib is None else ((y == 0) & calib)
    neg = np.sort(oof[neg_mask])
    thr = neg[int(np.ceil((1 - fpr) * len(neg))) - 1] if len(neg) else 0.0
    return float(thr) + 1e-9, float(roc_auc_score(y, oof))


def rates(scores, y, thr):
    pred = scores > thr
    tp = int(np.sum(pred & (y == 1))); fn = int(np.sum(~pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0))); tn = int(np.sum(~pred & (y == 0)))
    return {"tpr": round(tp / max(1, tp + fn), 3), "fpr": round(fp / max(1, fp + tn), 3),
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def wiki_sentences(split: str, n: int, seed: int) -> list[str]:
    """Wikipedia sentences (WikiText-103, CC BY-SA). Train/test come from different articles."""
    f = glob.glob(os.path.join(HERE, "data", "wikitext", "wikitext-103-raw-v1", f"{split}-*.parquet"))[0]
    lines = [l.strip() for l in pd.read_parquet(f).text if len(l.strip()) > 40 and not l.strip().startswith("=")]
    sents = []
    for l in lines:
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", l):
            s = re.sub(r"\s+@(.)@\s+", r"", s)       # WikiText "@-@" tokenisation artefacts
            if 5 <= len(s.split()) <= 60:
                sents.append(s)
    rng = np.random.default_rng(seed)
    return [sents[i] for i in rng.choice(len(sents), size=min(n, len(sents)), replace=False)]


def load_extra(exclude: set[str], max_pos: int = 3000, seed: int = 0) -> tuple[list[str], list[int]]:
    """Extra MIT-licensed training data: Lakera gandalf (all injections) + SPML user prompts
    (deduplicated; injections subsampled). deepset test texts are excluded to keep the test clean."""
    texts, labels = [], []
    for f in glob.glob(os.path.join(HERE, "data", "gandalf", "data", "*.parquet")):
        for t in pd.read_parquet(f).text:
            texts.append(t); labels.append(1)
    sp = pd.read_csv(os.path.join(HERE, "data", "spml", "spml_prompt_injection.csv"))
    sp = sp.dropna(subset=["User Prompt"]).drop_duplicates(subset=["User Prompt"])
    pos = sp[sp["Prompt injection"] == 1].sample(frac=1, random_state=seed).head(max_pos)
    neg = sp[sp["Prompt injection"] == 0]
    for df, lab in ((pos, 1), (neg, 0)):
        for t in df["User Prompt"]:
            texts.append(str(t)); labels.append(lab)
    seen, T, Y = set(exclude), [], []
    for t, l in zip(texts, labels):
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k); T.append(t); Y.append(l)
    return T, Y


def main(model="fp32", extra=False, wiki=False, gen=False):
    path = os.path.join(MODEL_DIR, "onnx", "model.onnx" if model == "fp32" else "model_int8.onnx")
    emb = Embedder(path, threads=4)
    emb.enable_cache(os.path.join(HERE, "data", f"embcache_{model}.pkl"))
    R = {"model": model, "extra_training_data": extra, "wiki_benign": wiki, "generated_benign": gen}

    load = lambda s: pd.read_parquet(glob.glob(os.path.join(HERE, "data", "deepset", "data", f"{s}-*.parquet"))[0])  # noqa: E731
    dtr, dte = load("train"), load("test")
    for d in (dtr, dte):
        d["lang"] = d.text.map(lambda t: "de" if len(GER.findall(t)) >= 2 else "en")
    doc_tr = [s for d in TRAIN_DOCS for s in doc_sentences(os.path.join(ROOT, d))]
    doc_te = [s for d in TEST_DOCS for s in doc_sentences(d)]
    R["sizes"] = {"deepset_train": len(dtr), "deepset_test": len(dte), "doc_train_sentences": len(doc_tr),
                  "doc_test_sentences": len(doc_te), "hard_negatives": len(HARD_NEGATIVES)}

    t = time.perf_counter()
    Xtr = np.vstack([emb.encode(list(dtr.text)), emb.encode(doc_tr)])
    ytr = np.concatenate([dtr.label.values, np.zeros(len(doc_tr), int)])
    calib = np.concatenate([np.zeros(len(dtr), bool), np.ones(len(doc_tr), bool)])
    wiki_te = []
    if wiki:
        wtr, wiki_te = wiki_sentences("train", 6000, 0), wiki_sentences("test", 2000, 1)
        R["sizes"]["wiki_train"], R["sizes"]["wiki_test"] = len(wtr), len(wiki_te)
        Xtr = np.vstack([Xtr, emb.encode(wtr)])
        ytr = np.concatenate([ytr, np.zeros(len(wtr), int)])
        calib = np.concatenate([calib, np.ones(len(wtr), bool)])
    if gen:
        g = [normalize(x) for x in benign_generate()]
        R["sizes"]["generated_benign"] = len(g)
        Xtr = np.vstack([Xtr, emb.encode(g)])
        ytr = np.concatenate([ytr, np.zeros(len(g), int)])
        calib = np.concatenate([calib, np.ones(len(g), bool)])
    if extra:
        et, ey = load_extra({s.strip().lower() for s in dte.text})
        R["sizes"]["extra_train"] = {"total": len(et), "injection": int(sum(ey)), "benign": len(ey) - int(sum(ey))}
        Xtr = np.vstack([Xtr, emb.encode(et)])
        ytr = np.concatenate([ytr, np.array(ey, int)])
        calib = np.concatenate([calib, np.zeros(len(ey), bool)])

    # thresholds are calibrated on document-domain benign text when we have enough of it
    calib = calib if (wiki or extra or gen) else None
    Xwiki = emb.encode(wiki_te) if wiki_te else np.zeros((0, 384))
    Xds, Xdoc, Xhn = emb.encode(list(dte.text)), emb.encode(doc_te), emb.encode(HARD_NEGATIVES)
    R["embed_seconds"] = round(time.perf_counter() - t, 2)
    yds = dte.label.values
    en = (dte.lang == "en").values

    pr = probes()
    Xpr = emb.encode([s for _, s in pr]) if pr else np.zeros((0, 384))

    res = {}
    for name, make in METHODS.items():
        thr, cv_auc = cv_threshold(make, Xtr, ytr, calib=calib)
        m = make().fit(Xtr, ytr)
        s_ds, s_doc, s_hn, s_wk = m.score(Xds), m.score(Xdoc), m.score(Xhn), m.score(Xwiki)
        # threshold-free: deepset test injections vs ALL benign test sets
        y_all = np.concatenate([yds, np.zeros(len(s_doc) + len(s_hn) + len(s_wk), int)])
        s_all = np.concatenate([s_ds, s_doc, s_hn, s_wk])
        res[name] = {
            "cv_auc_train": round(cv_auc, 4),
            "auc_test_all_benign": round(roc_auc_score(y_all, s_all), 4),
            "deepset_test": rates(s_ds, yds, thr),
            "deepset_test_en": rates(s_ds[en], yds[en], thr),
            "deepset_test_de": rates(s_ds[~en], yds[~en], thr),
            "doc_test_fpr": rates(s_doc, np.zeros(len(s_doc), int), thr)["fpr"],
            "hard_neg_fpr": rates(s_hn, np.zeros(len(s_hn), int), thr)["fpr"],
            "wiki_test_fpr": rates(s_wk, np.zeros(len(s_wk), int), thr)["fpr"] if len(s_wk) else None,
            "hard_neg_flagged": [HARD_NEGATIVES[i] for i in np.where(s_hn > thr)[0]][:8],
            "doc_flagged": [doc_te[i][:140] for i in np.where(s_doc > thr)[0]][:40],
            "probes": [{"file": f, "flagged": bool(sc > thr), "margin": round(float(sc - thr), 3)}
                       for (f, _), sc in zip(pr, m.score(Xpr))] if pr else [],
        }
        if name == "best_dim":
            res[name]["dimension"] = m.j
            res[name]["dimension_train_auc"] = round(m.auc, 4)
    # regex baseline (the scanner's current check), threshold = its hidden-text threshold 0.3
    rx = lambda texts: np.array([semantic.classify(t)["score"] for t in texts])  # noqa: E731
    s_ds, s_doc, s_hn = rx(list(dte.text)), rx(doc_te), rx(HARD_NEGATIVES)
    res["regex"] = {"deepset_test": rates(s_ds, yds, 0.3 - 1e-9),
                    "deepset_test_en": rates(s_ds[en], yds[en], 0.3 - 1e-9),
                    "doc_test_fpr": rates(s_doc, np.zeros(len(s_doc), int), 0.3 - 1e-9)["fpr"],
                    "hard_neg_fpr": rates(s_hn, np.zeros(len(s_hn), int), 0.3 - 1e-9)["fpr"],
                    "probes": [{"file": f, "flagged": bool(semantic.classify(s)["score"] >= 0.3)} for f, s in pr]}
    R["methods"] = res

    # ---------------------------------------------- direction analysis: is it ONE dimension?
    aucs = np.array([roc_auc_score(ytr, Xtr[:, j]) for j in range(Xtr.shape[1])])
    strength = np.abs(aucs - 0.5)
    order = np.argsort(-strength)
    lr = Probe(1.0).fit(Xtr, ytr)
    w = lr.m.coef_[0] / np.linalg.norm(lr.m.coef_[0])
    c = Centroid().fit(Xtr, ytr).d
    dim_curve = {}
    y_eval = np.concatenate([yds, np.zeros(len(Xdoc) + len(Xhn), int)])
    X_eval = np.vstack([Xds, Xdoc, Xhn])
    for k in (1, 2, 5, 10, 25, 50, 100, 200, 384):
        p = Probe(1.0, dims=order[:k]).fit(Xtr, ytr)
        dim_curve[k] = round(roc_auc_score(y_eval, p.score(X_eval)), 4)
    # how much of the probe's weight sits in its top dimensions
    wsq = np.sort(w ** 2)[::-1]
    R["direction"] = {
        "top_dimensions_by_auc": [{"dim": int(j), "auc": round(float(aucs[j]), 3)} for j in order[:10]],
        "dims_with_auc_dev_over_0.15": int(np.sum(strength > 0.15)),
        "cosine_probe_vs_centroid": round(float(w @ c), 3),
        "probe_weight_mass_in_top10_dims": round(float(wsq[:10].sum()), 3),
        "probe_weight_mass_in_top50_dims": round(float(wsq[:50].sum()), 3),
        "test_auc_vs_num_dims_used": dim_curve,
    }

    # ---------------------------------------------- dilution: injection buried in a paragraph
    rng = np.random.default_rng(0)
    inj_en = list(dte.text[(dte.label == 1) & en])
    dil = {}
    lr_thr, _ = cv_threshold(METHODS["logreg"], Xtr, ytr, calib=calib)
    kn_thr, _ = cv_threshold(METHODS["knn_max"], Xtr, ytr, calib=calib)
    kn = Knn("max").fit(Xtr, ytr)
    for k in (0, 1, 3, 6, 10):
        paras = []
        for s in inj_en:
            ctx = list(rng.choice(doc_te, size=k, replace=False)) if k else []
            pos = rng.integers(0, k + 1)
            paras.append(" ".join(ctx[:pos] + [s] + ctx[pos:]))
        Xp = emb.encode(paras)
        dil[f"{k}_benign_sentences_around"] = {
            "logreg_tpr": round(float(np.mean(lr.score(Xp) > lr_thr)), 3),
            "knn_max_tpr": round(float(np.mean(kn.score(Xp) > kn_thr)), 3)}
    R["dilution"] = dil

    emb.save_cache()
    out = os.path.join(HERE, f"results_accuracy_{model}{'_extra' if extra else ''}{'_wiki' if wiki else ''}{'_gen' if gen else ''}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    return R


if __name__ == "__main__":
    for arg in sys.argv[1:] or ["fp32"]:
        model, *flags = arg.split("+")
        R = main(model, extra="extra" in flags, wiki="wiki" in flags, gen="gen" in flags)
        print(f"===== {arg}  sizes={R['sizes']}  embed={R['embed_seconds']}s")
        hdr = f"{'method':12} {'cvAUC':>6} {'AUC':>6} | {'TPR':>5} {'TPRen':>5} {'TPRde':>5} | {'FPRdoc':>6} {'FPRhard':>7} {'FPRwiki':>7} | probes"
        print(hdr)
        for n, m in R["methods"].items():
            pr = "".join("Y" if p["flagged"] else "n" for p in m["probes"])
            print(f"{n:12} {m.get('cv_auc_train', float('nan')):6.3f} {m.get('auc_test_all_benign', float('nan')):6.3f} | "
                  f"{m['deepset_test']['tpr']:5.2f} {m['deepset_test_en']['tpr']:5.2f} "
                  f"{m.get('deepset_test_de', {}).get('tpr', float('nan')):5.2f} | "
                  f"{m['doc_test_fpr']:6.3f} {m['hard_neg_fpr']:7.3f} {m.get('wiki_test_fpr') if m.get('wiki_test_fpr') is not None else float('nan'):7.3f} | {pr}")
        print("direction:", json.dumps(R["direction"]))
        print("dilution:", json.dumps(R["dilution"]))
