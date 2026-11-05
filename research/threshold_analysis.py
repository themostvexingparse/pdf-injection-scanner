"""Is a candidate better at the SAME false-positive rate on real documents? (ranking vs. calibration)

Each candidate's strong threshold is re-set on HALF of the local PDFs (by file: at most 1% of their
sentences flagged) and then measured on the OTHER half and on every other acceptance set. If a
fine-tuned model then still catches more held-out injections than the base model at the same local
false-positive level, its problem is calibration, not quality.

  python research/threshold_analysis.py      (after accept_models.py extracted the candidates)
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import local_docs as L  # noqa: E402  private documents, configured in research/local_docs.json

CALIB_PATHS, TEST_PATHS = L.docs("calib"), L.docs("heldout")
CALIB_FILES = [os.path.basename(p) for p in CALIB_PATHS]
TEST_FILES = [os.path.basename(p) for p in TEST_PATHS]


def build(path):
    import visible_diag as D
    sets = json.load(open(os.path.join(HERE, "data", "acceptance_sets.json"), encoding="utf-8"))
    sets.pop("local_pdf", None)
    sets["calib_pdf"] = [s for p in CALIB_PATHS for s in D.doc_sentences(p)]
    sets["test_pdf"] = [s for p in TEST_PATHS for s in D.doc_sentences(p)]
    json.dump(sets, open(path, "w", encoding="utf-8"))


def worker(sets_path, out_path):
    from pdfscan import embedding
    sig = embedding.get()
    sets = json.load(open(sets_path, encoding="utf-8"))
    out = {k: [r["score"] if r["score"] is not None else -1e9 for r in sig.score_units(v)] for k, v in sets.items()}
    out["_thr"] = sig.thr
    json.dump(out, open(out_path, "w"))


def scores_for(probe, sets_path):
    out = tempfile.mktemp(suffix=".json")
    env = dict(os.environ)
    env.pop("PDFSCAN_PROBE", None)
    if probe:
        env["PDFSCAN_PROBE"] = probe
    subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", sets_path, out], env=env, check=True, cwd=ROOT)
    r = json.load(open(out)); os.unlink(out)
    return r


def main():
    sets_path = os.path.join(HERE, "data", "threshold_sets.json")
    if not os.path.exists(sets_path):
        build(sets_path)
    cands = [("installed (base)", None)] + [
        (os.path.basename(d), os.path.join(d, "probe.json"))
        for d in sorted(glob.glob(os.path.join(HERE, "data", "candidates", "candidates", "*")))]
    cols = ["test_pdf", "hard_neg", "generated", "emails", "wiki", "deepset_inj", "gandalf_sum_inj"]
    rows = {}
    for fpr in (0.01, 0.003):
        print(f"\n== threshold re-set so that <= {fpr:.1%} of calibration-PDF sentences are flagged "
              f"({', '.join(CALIB_FILES)}); measured on held-out files {', '.join(TEST_FILES)} and the other sets")
        print(f"{'candidate':18}" + "".join(f"{c[:13]:>14}" for c in cols) + f"{'probes':>9}")
        for label, probe in cands:
            r = rows.get(label) or scores_for(probe, sets_path)
            rows[label] = r
            cal = np.sort(np.array(r["calib_pdf"]))
            thr = float(cal[int(np.ceil((1 - fpr) * len(cal))) - 1])
            vals = [np.mean(np.array(r[c]) > thr) for c in cols]
            pr = "".join("Y" if s > thr else "-" for s in r["probes"])
            print(f"{label:18}" + "".join(f"{v:>14.3f}" for v in vals) + f"{pr:>9}")
    print("\n(test_pdf/hard_neg/generated/emails/wiki = false-positive rates; *_inj = catch rates)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main()
