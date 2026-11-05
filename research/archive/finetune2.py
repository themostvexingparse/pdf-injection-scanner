"""Fine-tune bge-small-en-v1.5 on the synthetic + real set (research/ft_dataset.parquet) and compare
with the frozen-embedding probe trained on the same data.

Model = encoder -> CLS -> L2-normalise -> linear head: the same shape the scanner runs today, so a
fine-tuned model is a drop-in replacement (new ONNX + probe weights).
Selection: every EVAL_EVERY steps, validation score = mean TPR at 0.5% and 1% FPR on validation
DOCUMENT benign text; the best checkpoint is kept.
Report: per-source TPR / FPR on the test split at thresholds calibrated on validation document text
(moderate = 1% FPR, strong = 0.1% FPR), plus the audit/CS223 probes.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.dataset  # noqa: F401  (must load before torch on Windows)
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embedder import MODEL_DIR, Embedder  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(8)
EPOCHS, BS, LR_ENC, LR_HEAD, EVAL_EVERY, MAXLEN = 2, 32, 2e-5, 1e-3, 400, 128
SRC_WEIGHT = {"spml": 0.5, "deepset": 1.0}       # chat-style benign: down-weighted (not our domain)


class Clf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_DIR)
        self.head = torch.nn.Linear(384, 1)
        torch.nn.init.normal_(self.head.weight, std=0.02)

    def embed(self, **b):
        h = self.enc(**b).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(h, dim=-1)

    def forward(self, **b):
        return self.head(self.embed(**b)).squeeze(-1)


def batches_by_length(lengths, bs, rng):
    """Shuffled batches of similar length (little padding -> much faster on CPU)."""
    idx = np.argsort(np.array(lengths) + rng.random(len(lengths)) * 8)
    bl = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    rng.shuffle(bl)
    return bl


def predict(model, tok, texts, bs=128):
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros(len(texts), np.float32)
    with torch.no_grad():
        for i in range(0, len(order), bs):
            ids = order[i:i + bs]
            b = tok([texts[j] for j in ids], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            out[ids] = model(**b).numpy()
    return out


def thr_at(neg_scores, fpr):
    s = np.sort(neg_scores)
    return float(s[min(len(s) - 1, int(np.ceil((1 - fpr) * len(s))) - 1)])


def val_metric(scores, df_val):
    neg = scores[(df_val.label.values == 0) & (df_val.domain.values == "doc")]
    pos = scores[df_val.label.values == 1]
    return float(np.mean([np.mean(pos > thr_at(neg, f)) for f in (0.005, 0.01)]))


def report(name, score_fn, df_val, df_test, probe_texts):
    sv = score_fn(df_val.text.tolist())
    neg = sv[(df_val.label.values == 0) & (df_val.domain.values == "doc")]
    thr = {"moderate": thr_at(neg, 0.01), "strong": thr_at(neg, 0.001)}
    st = score_fn(df_test.text.tolist())
    r = {"thresholds": thr, "auc_test": round(float(roc_auc_score(df_test.label, st)), 4), "by_source": {}}
    for src, g in df_test.groupby("source"):
        s = st[g.index.map(df_test.index.get_loc)]
        lab = g.label.iloc[0]
        key = "tpr" if lab == 1 else "fpr"
        r["by_source"][src] = {"n": len(g), "label": int(lab),
                               f"{key}_moderate": round(float(np.mean(s > thr["moderate"])), 3),
                               f"{key}_strong": round(float(np.mean(s > thr["strong"])), 3)}
    ps = score_fn(probe_texts) if probe_texts else []
    r["probes"] = ["S" if v > thr["strong"] else "M" if v > thr["moderate"] else "-" for v in ps]
    print(f"\n== {name}: test AUC {r['auc_test']}  probes {''.join(r['probes'])}")
    print(f"   {'source':34}{'n':>5}  {'moderate':>9} {'strong':>8}")
    for src, v in sorted(r["by_source"].items(), key=lambda x: (-x[1]["label"], x[0])):
        k = "tpr" if v["label"] else "fpr"
        print(f"   {src:34}{v['n']:>5}  {k} {v[f'{k}_moderate']:>5.3f} {v[f'{k}_strong']:>8.3f}")
    return r


def main():
    df = pd.read_parquet(os.path.join(HERE, "data", "ft_dataset.parquet"))
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    import eval_embed as E
    probe_texts = [s for _, s in E.probes()]
    R = {"sizes": {"train": len(tr), "val": len(va), "test": len(te)}}

    # ---------------- baseline: frozen embeddings + logistic probe on the same training data
    emb = Embedder(threads=8)
    emb.enable_cache(os.path.join(HERE, "data", "embcache_fp32.pkl"))
    t0 = time.perf_counter()
    Xtr = emb.encode(tr.text.tolist())
    w = tr.source.map(lambda s: SRC_WEIGHT.get(s, 1.0)).values
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(Xtr, tr.label.values, sample_weight=w)
    emb.save_cache()
    R["frozen"] = report("frozen + logistic probe", lambda t: lr.decision_function(emb.encode(t)), va, te, probe_texts)
    R["frozen"]["fit_seconds"] = round(time.perf_counter() - t0, 1)
    emb.save_cache()

    # ---------------- fine-tune
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = Clf()
    with torch.no_grad():   # start the head from the frozen probe (same input space at step 0)
        model.head.weight.copy_(torch.tensor(lr.coef_, dtype=torch.float32))
        model.head.bias.copy_(torch.tensor(lr.intercept_, dtype=torch.float32))
    opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": LR_ENC},
                             {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=0.01)
    texts, y = tr.text.tolist(), tr.label.values.astype(np.float32)
    sw = torch.tensor(w, dtype=torch.float32)
    lens = [len(t) for t in texts]
    rng = np.random.default_rng(0)
    steps_total = EPOCHS * int(np.ceil(len(texts) / BS))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / (0.05 * steps_total)) * max(0.0, (steps_total - s) / (0.95 * steps_total)))
    pos_weight = torch.tensor((y == 0).sum() / max(1, (y == 1).sum()))
    best, best_state, step, t0, hist = -1.0, None, 0, time.perf_counter(), []
    for ep in range(EPOCHS):
        for bi in batches_by_length(lens, BS, rng):
            model.train()
            b = tok([texts[i] for i in bi], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
            logits = model(**b)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, torch.tensor(y[bi]), weight=sw[bi], pos_weight=pos_weight)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 100 == 0:
                print(f"  epoch {ep + 1} step {step}/{steps_total} loss {loss.item():.4f} "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
            if step % EVAL_EVERY == 0 or step == steps_total:
                m = val_metric(predict(model, tok, va.text.tolist()), va)
                hist.append((step, round(m, 4)))
                print(f"  -> val TPR@0.5-1%FPR(doc) = {m:.4f}{'  (best)' if m > best else ''}", flush=True)
                if m > best:
                    best, best_state = m, {k: v.detach().clone() for k, v in model.state_dict().items()}
    R["train_seconds"] = round(time.perf_counter() - t0, 1)
    R["val_history"] = hist
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(HERE, "data", "ft2_best.pt"))
    R["finetuned"] = report("fine-tuned (best val checkpoint)", lambda t: predict(model, tok, t), va, te, probe_texts)
    R["frozen_val_metric"] = round(val_metric(lr.decision_function(emb.encode(va.text.tolist())), va), 4)
    R["finetuned_val_metric"] = round(best, 4)
    with open(os.path.join(HERE, "results_finetune2.json"), "w") as f:
        json.dump(R, f, indent=1)
    print(json.dumps({k: R[k] for k in ("sizes", "train_seconds", "val_history", "frozen_val_metric",
                                        "finetuned_val_metric")}, indent=1))


if __name__ == "__main__":
    main()
