"""Private benign documents used for local evaluation and calibration. They are never committed.

Copy research/local_docs.example.json to research/local_docs.json (git-ignored) and list your own
PDFs by role:

  train    benign sentences for the base probe (train_probe.py, eval_embed.py)
  test     held-out benign documents (eval_embed.py)
  calib    threshold calibration for ensemble members (install_ensemble.py, threshold_analysis.py)
  heldout  documents for checking that calibration (threshold_analysis.py); keep disjoint from calib
  probe    documents containing a known hidden instruction to report scores on (eval_embed.probes)

Without the file every role is empty and the scripts fall back to public data and the test fixtures.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG = os.path.join(HERE, "local_docs.json")
FIXTURES = os.path.join(ROOT, "tests", "fixtures", "audit")
AUDIT_FIXTURES = [os.path.join(FIXTURES, f) for f in ("hidden-paraphrase-bypass.pdf", "hidden-classic-control.pdf")]


def _cfg() -> dict:
    if not os.path.exists(CONFIG):
        return {}
    with open(CONFIG, encoding="utf-8") as f:
        return json.load(f)


def docs(role: str) -> list[str]:
    """Absolute paths of the configured PDFs for a role (relative paths are relative to the repo)."""
    out = []
    for p in _cfg().get(role, []):
        p = os.path.expanduser(os.path.expandvars(p))
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if os.path.exists(p):
            out.append(p)
    return out


def all_docs() -> list[str]:
    """Every configured benign document once (all roles except 'probe')."""
    seen = []
    for role in ("train", "test", "calib", "heldout"):
        for p in docs(role):
            if p not in seen:
                seen.append(p)
    return seen
