"""Install the GPU notebook's output (pdfscan_ft_artifacts.zip) and calibrate it on local documents.

  python research/install_ft_artifacts.py path/to/pdfscan_ft_artifacts.zip
  python research/install_ft_artifacts.py --revert          # go back to the base-model probe

Steps
 1. unpack the fine-tuned ONNX encoder + tokenizer into models/<model_subdir>/
 2. back up the current pdfscan/models/probe.json + exemplars.npz (*.base.*), install the new ones
 3. load it through pdfscan.embedding (verifies the ONNX hash against probe.json)
 4. calibrate the visible-sentence gate (nearest-injection cosine) on LOCAL PDFs, which never leave
    this machine: the smallest cut-off with no flags on their genuine sentences, reported together
    with how many planted held-out injections (deepset test) would still be flagged.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG_MODELS = os.path.join(ROOT, "pdfscan", "models")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def revert():
    for n in ("probe", "exemplars"):
        ext = "json" if n == "probe" else "npz"
        b = os.path.join(PKG_MODELS, f"{n}.base.{ext}")
        if os.path.exists(b):
            shutil.copy2(b, os.path.join(PKG_MODELS, f"{n}.{ext}"))
    print("reverted to the base-model probe")


def install(zip_path: str):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        probe = json.loads(z.read("probe.json"))
        sub = probe["embedding_model"]["model_subdir"]
        dest = os.path.join(ROOT, "models", sub)
        for n in names:
            if n.startswith(sub + "/"):
                z.extract(n, os.path.join(ROOT, "models"))
        for n, ext in (("probe", "json"), ("exemplars", "npz")):
            cur = os.path.join(PKG_MODELS, f"{n}.{ext}")
            base = os.path.join(PKG_MODELS, f"{n}.base.{ext}")
            if os.path.exists(cur) and not os.path.exists(base):
                shutil.copy2(cur, base)
            with open(cur, "wb") as f:
                f.write(z.read(f"{n}.{ext}"))
        if "metrics.json" in names:
            with open(os.path.join(HERE, "results_finetune_gpu.json"), "wb") as f:
                f.write(z.read("metrics.json"))
    print(f"installed model into {dest}")
    return probe


def calibrate_visible():
    from pdfscan import embedding
    import visible_diag as D
    sig = embedding.get()
    if sig is None:
        raise SystemExit(f"embedding unavailable: {embedding.last_error()}")
    print("loaded:", sig.model_dir)
    thr = sig.thr
    inj = pd.read_parquet(glob.glob(os.path.join(HERE, "data", "deepset", "data", "test-*.parquet"))[0])
    inj = [t for t, l in zip(inj.text, inj.label) if l == 1]
    Vi = sig._embed(inj); si = Vi @ sig.w + sig.b; near_i = (sig.ex_vec @ Vi.T).max(0)
    docs = []
    for pdf in D.FILES + [os.path.join(ROOT, "cs211_hw2.pdf")]:
        S = D.doc_sentences(pdf)
        if len(S) >= 5:
            V = sig._embed(S)
            docs.append((os.path.basename(pdf), S, V @ sig.w + sig.b, (sig.ex_vec @ V.T).max(0)))
    print("\nstrong-tier sentences per local document (no cosine gate):",
          {n: int((s > thr["strong"]).sum()) for n, S, s, _ in docs})
    print(f"{'cosine >=':>10}{'local false flags':>19}{'planted flagged':>17}")
    chosen = None
    for c in np.round(np.arange(0.0, 0.96, 0.02), 2):
        fp = sum(int(((s > thr["strong"]) & (near >= c)).sum()) for _, _, s, near in docs)
        tp = float(np.mean((si > thr["strong"]) & (near_i >= c)))
        if c in (0.0,) or c >= 0.5:
            print(f"{c:>10.2f}{fp:>19}{tp:>16.0%}")
        if chosen is None and fp == 0:
            chosen = (float(c), tp)
    if chosen is None:
        # no cut-off removes all strong-tier flags on genuine local sentences: the model does not
        # transfer to these documents. Leave the gate unset and say so loudly.
        print("\nWARNING: no nearest-injection cut-off gives zero false flags on local documents. "
              "This model does not transfer to them; consider `--revert`.")
        return
    p = os.path.join(PKG_MODELS, "probe.json")
    probe = json.load(open(p))
    probe["visible_gate"] = {"min_exemplar_cosine": chosen[0],
                             "calibration": "smallest cut-off with zero strong-tier flags on local genuine PDF "
                                            "sentences; planted deepset-test injections still flagged: "
                                            f"{chosen[1]:.0%}"}
    json.dump(probe, open(p, "w"), indent=1)
    print(f"\nvisible gate set to cosine >= {chosen[0]} (planted injections still flagged: {chosen[1]:.0%})")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--revert"]:
        revert()
    else:
        install(sys.argv[1])
        calibrate_visible()
