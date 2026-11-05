"""Local acceptance tests for embedding-probe candidates (GPU notebook output) vs. the installed model.

  python research/accept_models.py C:/path/pdfscan_ft_candidates.zip

Everything here runs locally; the PDFs and hand-written/generated sentences never leave the machine.
Each candidate is scored in its own process (PDFSCAN_PROBE points at the candidate's probe.json),
exactly the way the scanner scores text at runtime (per sentence/window, max per text).

Benign side (must stay quiet)              Injection side (held out from training)
  local PDF sentences (your documents)       deepset/prompt-injections test split
  hand-written hard negatives                Lakera/gandalf_summarization (never trained on)
  generated instructions to human readers    audit fixture spans + a private document's hidden sentence (probes)
  Wikipedia test articles, benign emails
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

# strong-tier false-positive limits; generated reader instructions: 3% = what the installed base model
# itself does (2.6%), i.e. "no worse than today"
LIMITS = {"local_pdf_strong": 0.01, "hard_neg_strong": 0.05, "generated_strong": 0.03, "wiki_strong": 0.005,
          "emails_strong": 0.02}


def build_testsets(path: str) -> None:
    import pandas as pd
    import eval_embed as E
    import visible_diag as D
    from benign_gen import generate
    from hard_negatives import HARD_NEGATIVES
    sets = {}
    pdfs = D.FILES
    sets["local_pdf"] = [s for p in pdfs for s in D.doc_sentences(p)]
    sets["hard_neg"] = list(HARD_NEGATIVES)
    sets["generated"] = generate(n_sent=800, n_frag=0, seed=5)
    sets["wiki"] = [E.normalize(x) for x in E.wiki_sentences("test", 1500, 1)]
    sets["emails"] = json.load(open(os.path.join(HERE, "data", "llmail", "data", "emails_for_fp_tests.json"), encoding="utf-8"))
    ds = pd.read_parquet(glob.glob(os.path.join(HERE, "data", "deepset", "data", "test-*.parquet"))[0])
    sets["deepset_inj"] = [t for t, l in zip(ds.text, ds.label) if l == 1]
    sets["gandalf_sum_inj"] = [t for f in glob.glob(os.path.join(HERE, "data", "gandalf_sum", "data", "*.parquet"))
                               for t in pd.read_parquet(f).text]
    sets["probes"] = [s for _, s in E.probes()]
    json.dump(sets, open(path, "w", encoding="utf-8"))
    print({k: len(v) for k, v in sets.items()})


def worker(sets_path: str, out_path: str) -> None:
    """Runs inside a process whose PDFSCAN_PROBE selects the candidate."""
    from pdfscan import embedding
    sig = embedding.get()
    if sig is None:
        json.dump({"error": embedding.last_error()}, open(out_path, "w")); return
    sets = json.load(open(sets_path, encoding="utf-8"))
    res = {"model_dir": os.path.basename(sig.model_dir), "thresholds": sig.thr}
    for name, texts in sets.items():
        r = sig.score_units(texts)
        tiers = [x["tier"] for x in r]
        res[name] = {"n": len(texts), "moderate": float(np.mean([t != "none" for t in tiers])),
                     "strong": float(np.mean([t == "strong" for t in tiers]))}
        if name == "probes":
            res[name]["tiers"] = "".join({"strong": "S", "moderate": "M", "none": "-"}[t] for t in tiers)
    json.dump(res, open(out_path, "w"))


def run_candidate(label: str, probe: str | None, sets_path: str) -> dict:
    out = tempfile.mktemp(suffix=".json")
    env = dict(os.environ)
    if probe:
        env["PDFSCAN_PROBE"] = probe
    else:
        env.pop("PDFSCAN_PROBE", None)
    subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", sets_path, out], env=env, check=True,
                   cwd=ROOT)
    r = json.load(open(out)); os.unlink(out)
    r["label"] = label
    return r


def verdict(r: dict) -> tuple[bool, list[str]]:
    fails = []
    for key, lim in LIMITS.items():
        name, tier = key.rsplit("_", 1)
        v = r[name][tier]
        if v > lim:
            fails.append(f"{name} {tier} {v:.3f} > {lim}")
    return not fails, fails


def main(zip_path: str):
    work = os.path.join(HERE, "data", "candidates")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)
    sets_path = os.path.join(HERE, "data", "acceptance_sets.json")
    if not os.path.exists(sets_path):
        build_testsets(sets_path)
    cands = [("installed (base)", None)] + [
        (os.path.basename(d), os.path.join(d, "probe.json"))
        for d in sorted(glob.glob(os.path.join(work, "candidates", "*"))) if os.path.exists(os.path.join(d, "probe.json"))]
    results = [run_candidate(lbl, p, sets_path) for lbl, p in cands]
    cols = ["local_pdf", "hard_neg", "generated", "wiki", "emails", "deepset_inj", "gandalf_sum_inj"]
    print(f"\n{'candidate':18}" + "".join(f"{c[:13]:>14}" for c in cols) + f"{'probes':>8}  accept")
    print(f"{'(strong tier)':18}" + "".join(f"{'FPR' if 'inj' not in c else 'TPR':>14}" for c in cols))
    for r in results:
        if "error" in r:
            print(f"{r['label']:18} ERROR {r['error']}"); continue
        ok, fails = verdict(r)
        r["accepted"], r["fails"] = ok, fails
        print(f"{r['label']:18}" + "".join(f"{r[c]['strong']:>14.3f}" for c in cols)
              + f"{r['probes'].get('tiers', ''):>8}  {'PASS' if ok else 'FAIL'}")
    print(f"\n{'(moderate tier)':18}" + "".join(f"{c[:13]:>14}" for c in cols))
    for r in results:
        if "error" not in r:
            print(f"{r['label']:18}" + "".join(f"{r[c]['moderate']:>14.3f}" for c in cols))
    for r in results:
        if "error" not in r and not r["accepted"]:
            print(f"  {r['label']}: " + "; ".join(r["fails"]))
    ok = [r for r in results if "error" not in r and r["accepted"]]
    best = max(ok, key=lambda r: (r["deepset_inj"]["strong"] + r["gandalf_sum_inj"]["strong"]) / 2, default=None)
    print("\nbest accepted:", best["label"] if best else "none", "(by mean strong-tier TPR on held-out injections)")
    json.dump(results, open(os.path.join(HERE, "results_acceptance.json"), "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1])
