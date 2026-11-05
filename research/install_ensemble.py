"""Add a fine-tuned candidate as an ENSEMBLE MEMBER next to the installed base probe.

  python research/install_ensemble.py ft_e2_lowlr        # from research/data/candidates/candidates/<name>
  python research/install_ensemble.py --remove           # back to the base probe alone

Why an ensemble: at equal false-positive rates on real documents the fine-tuned model catches about
twice as many document-embedded injections as the base probe but misses short chat-style ones the
base probe catches (research/threshold_analysis.py). "Flag if either flags" keeps both strengths at
practically the base model's false-positive cost.

The member's thresholds are re-set on LOCAL calibration documents (they never leave the machine):
strong = at most 0.3% of their sentences, moderate = at most 1%. Its held-out evaluation used other
files (threshold_analysis.TEST_FILES).
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG_MODELS = os.path.join(ROOT, "pdfscan", "models")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def remove():
    p = os.path.join(PKG_MODELS, "probe.json")
    probe = json.load(open(p))
    probe.pop("ensemble", None)
    json.dump(probe, open(p, "w"), indent=1)
    print("ensemble removed; base probe alone")


def install(name: str):
    src = os.path.join(HERE, "data", "candidates", "candidates", name)
    cprobe = json.load(open(os.path.join(src, "probe.json")))
    sub = cprobe["embedding_model"]["model_subdir"]
    dest_model = os.path.join(ROOT, "models", sub)
    if os.path.isdir(dest_model):
        shutil.rmtree(dest_model)
    shutil.copytree(os.path.join(src, sub), dest_model)
    shutil.copy2(os.path.join(src, "exemplars.npz"), os.path.join(PKG_MODELS, f"exemplars_{name}.npz"))
    member_path = os.path.join(PKG_MODELS, f"probe_{name}.json")
    cprobe.update(name=name, exemplars=f"exemplars_{name}.npz")
    json.dump(cprobe, open(member_path, "w"), indent=1)

    # re-set thresholds on local calibration documents
    from pdfscan.embedding import EmbeddingSignal
    import threshold_analysis as T
    sets_path = os.path.join(HERE, "data", "threshold_sets.json")
    if not os.path.exists(sets_path):
        T.build(sets_path)
    calib = json.load(open(sets_path, encoding="utf-8"))["calib_pdf"]
    sig = EmbeddingSignal(probe_path=member_path, load_members=False)
    s = np.sort([r["score"] for r in sig.score_units(calib) if r["score"] is not None])
    at = lambda f: float(s[int(np.ceil((1 - f) * len(s))) - 1])  # noqa: E731
    cprobe["thresholds_notebook"] = cprobe["thresholds"]
    cprobe["thresholds"] = {"moderate": round(at(0.01), 5), "strong": round(at(0.003), 5)}
    cprobe["calibration"] = (f"local calibration PDFs {', '.join(T.CALIB_FILES)} ({len(s)} sentences): "
                             "moderate = 1% flagged, strong = 0.3% flagged; held-out check on "
                             f"{', '.join(T.TEST_FILES)} (research/threshold_analysis.py)")
    # strong threshold is already conservative on real documents: no extra nearest-injection gate
    cprobe["visible_gate"] = {"min_exemplar_cosine": 0.0,
                              "calibration": "not needed: strong threshold set on local documents"}
    json.dump(cprobe, open(member_path, "w"), indent=1)

    p = os.path.join(PKG_MODELS, "probe.json")
    base = json.load(open(p))
    base["name"] = base.get("name", "base")
    base["ensemble"] = [os.path.basename(member_path)]
    json.dump(base, open(p, "w"), indent=1)
    print(f"installed ensemble member '{name}' ({dest_model}); thresholds {cprobe['thresholds']} "
          f"(notebook: {cprobe['thresholds_notebook']})")


if __name__ == "__main__":
    if sys.argv[1] == "--remove":
        remove()
    else:
        install(sys.argv[1])
