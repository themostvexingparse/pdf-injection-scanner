"""Build the fine-tuning dataset: real licensed data + synthetic benign + mechanical augmentation.

No injection text is authored here. Positives come from licensed datasets; "synthetic" positives are
mechanical augmentations of those (planted between real benign document sentences, line-split,
bullet-prefixed, case-changed). Synthetic data on the benign side is generated (benign_gen.py).

Splits
  benign documents: split by DOCUMENT (no test sentence from a document seen in training)
  positives: split by example; Lakera/gandalf_summarization is held out entirely (out-of-distribution
  test: injections embedded in text to be summarised), as are the hand-written hard negatives.
Output: research/data/ft_dataset.parquet with columns text, label, source, split, domain
  domain = "doc" for document-style benign text (thresholds are calibrated on these), "chat" for chatbot
  user messages, "inj" for injections.
"""
from __future__ import annotations

import glob
import json
import os
import random
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(HERE, "data")
sys.path.insert(0, HERE)
from benign_gen import generate as benign_generate  # noqa: E402
from hard_negatives import HARD_NEGATIVES  # noqa: E402

RNG = random.Random(7)
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")
LOCAL_TEST_PDFS = {"cs211_hw2.pdf", "lecture11.pdf", "hwm_readme.pdf", "Try_premium_features.pdf",
                   "PDFSigQFormalRep.pdf"}


def norm(t: str) -> str:
    t = re.sub(r"([.\u2024\u2026\u00b7_\-=*])\1{3,}", " ", str(t))
    t = re.sub(r"[\u2022\u25aa\u25cf\u00b7]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def sentences(text: str, lo=5, hi=80) -> list[str]:
    out = []
    for s in SENT_RE.split(norm(text)):
        if lo <= len(s.split()) <= hi:
            out.append(s)
    return out


def split_of(key: str, val=0.1, test=0.1) -> str:
    r = random.Random(hash(key) & 0xFFFFFFFF).random()
    return "test" if r < test else "val" if r < test + val else "train"


rows: list[dict] = []


def add(text, label, source, split, domain):
    t = norm(text)
    if t:
        rows.append({"text": t[:2000], "label": int(label), "source": source, "split": split, "domain": domain})


# ------------------------------------------------------------------ benign documents (split by document)
def add_doc_corpus(source, docs, per_doc, cap):
    n = 0
    for i, doc in enumerate(docs):
        sp = split_of(f"{source}:{i}")
        ss = sentences(doc)
        for s in RNG.sample(ss, min(per_doc, len(ss))):
            add(s, 0, source, sp, "doc")
            n += 1
        if n >= cap:
            break


bill = pd.read_parquet(glob.glob(os.path.join(D, "billsum", "data", "train-*.parquet"))[0]).sample(frac=1, random_state=1)
add_doc_corpus("billsum", list(bill.text), 4, 4500)
def _paragraphs(node) -> list[str]:
    """gov_report 'reports' is a nested tree of sections with 'paragraphs' and 'subsections'."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [p for x in node for p in _paragraphs(x)]
    if isinstance(node, dict):
        return _paragraphs(node.get("paragraphs", [])) + _paragraphs(node.get("subsections", []))
    return []


gov = []
for f in glob.glob(os.path.join(D, "govreport", "data", "*.jsonl")):
    for line in open(f, encoding="utf-8"):
        gov.append(" ".join(_paragraphs(json.loads(line).get("reports")))[:30000])
RNG.shuffle(gov)
add_doc_corpus("gov_report", gov, 8, 4500)
lex = pd.read_parquet(glob.glob(os.path.join(D, "lexglue", "ecthr_a", "train-*.parquet"))[0]).sample(frac=1, random_state=2)
add_doc_corpus("lex_glue_ecthr", [" ".join(t) if not isinstance(t, str) else t for t in lex.text], 5, 4500)
wfiles = glob.glob(os.path.join(D, "wikitext", "wikitext-103-raw-v1", "train-*.parquet"))
wiki = [l for l in pd.read_parquet(wfiles[0]).text if len(l.strip()) > 200 and not l.strip().startswith("=")]
RNG.shuffle(wiki)
add_doc_corpus("wikitext", [re.sub(r"\s+@(.)@\s+", r"\1", w) for w in wiki], 3, 6000)

# local PDFs: split by file; known test documents always go to test
import fitz  # noqa: E402
pdfs = [os.path.join(ROOT, f) for f in ("CS223_Lab2.pdf", "cs211_hw1.pdf", "cs211_hw2.pdf")]
pdfs += []  # (further private local documents were listed here)
pdfs += [l.strip() for l in open(os.path.join(HERE, "data", "local_pdfs.txt"), encoding="utf-8")] \
    if os.path.exists(os.path.join(HERE, "data", "local_pdfs.txt")) else []
for p in dict.fromkeys(pdfs):
    try:
        text = " ".join(pg.get_text() for pg in fitz.open(p))
    except Exception:
        continue
    name = os.path.basename(p)
    sp = "test" if name in LOCAL_TEST_PDFS else "train"
    for s in sentences(text, 3, 80):
        if "museum room monitor" in s:   # the CS223 hidden instruction is a probe, not benign data
            continue
        add(s, 0, f"pdf:{name}", sp, "doc")

# synthetic benign (train/val only)
for i, s in enumerate(benign_generate(n_sent=5000, n_frag=1200, seed=11)):
    add(s, 0, "generated_benign", "val" if i % 10 == 0 else "train", "doc")
# hand-written hard negatives: test only
for s in HARD_NEGATIVES:
    add(s, 0, "hard_negatives", "test", "doc")

# chatbot-style benign
ds_tr = pd.read_parquet(glob.glob(os.path.join(D, "deepset", "data", "train-*.parquet"))[0])
ds_te = pd.read_parquet(glob.glob(os.path.join(D, "deepset", "data", "test-*.parquet"))[0])
for t in ds_tr.text[ds_tr.label == 0]:
    add(t, 0, "deepset", split_of("ds:" + t, val=0.15, test=0), "chat")
for t, l in zip(ds_te.text, ds_te.label):
    add(t, l, "deepset", "test", "chat" if l == 0 else "inj")
sp = pd.read_csv(os.path.join(D, "spml", "spml_prompt_injection.csv")).dropna(subset=["User Prompt"])
sp = sp.drop_duplicates(subset=["User Prompt"])
for t in sp[sp["Prompt injection"] == 0]["User Prompt"].sample(2000, random_state=3):
    add(t, 0, "spml", split_of("spml:" + t), "chat")
fp = json.load(open(os.path.join(D, "llmail", "data", "emails_for_fp_tests.json"), encoding="utf-8"))
for i, t in enumerate(fp):
    add(t, 0, "llmail_fp_emails", "train" if i % 2 else "test", "doc")

# ------------------------------------------------------------------ injections (licensed datasets)
for t in ds_tr.text[ds_tr.label == 1]:
    add(t, 1, "deepset", split_of("dsi:" + t, val=0.15, test=0), "inj")
for f in glob.glob(os.path.join(D, "gandalf", "data", "*.parquet")):
    s = "test" if "test-" in f else "val" if "validation" in f else "train"
    for t in pd.read_parquet(f).text:
        add(t, 1, "gandalf_ignore", s, "inj")
for t in sp[sp["Prompt injection"] == 1]["User Prompt"].sample(2500, random_state=4):
    add(t, 1, "spml", split_of("spmli:" + t), "inj")
ms = pd.read_parquet(glob.glob(os.path.join(D, "mosscap", "data", "train-*.parquet"))[0]).drop_duplicates("prompt")
ms = ms[ms.prompt.str.split().str.len() >= 8].sample(3500, random_state=5)
for t in ms.prompt:
    add(t, 1, "mosscap", split_of("ms:" + t), "inj")
ll = json.load(open(os.path.join(D, "llmail", "data", "labelled_unique_submissions_phase2.json"), encoding="utf-8"))
mails = [k for k, v in ll.items() if v.get("attack_attempt") in (True, "True") and 40 <= len(k) <= 2500]
RNG.shuffle(mails)
for k in mails[:6000]:
    m = re.match(r"\s*Subject of the email:\s*(.*?)\.?\s+Body:\s*(.*)$", k, re.S)
    body = (m.group(1) + ". " + m.group(2)) if m else k
    add(body, 1, "llmail", split_of("ll:" + k), "inj")
ya = pd.read_csv(os.path.join(D, "yanis", "prompt_injections.csv"))
for t in ya.prompt_injections.dropna().unique():
    add(t, 1, "yanis_multilingual", split_of("ya:" + t), "inj")
for f in glob.glob(os.path.join(D, "gandalf_sum", "data", "*.parquet")):
    for t in pd.read_parquet(f).text:
        add(t, 1, "gandalf_summarization_OOD", "test", "inj")

df = pd.DataFrame(rows).drop_duplicates(subset=["text", "label"])
# a text with both labels is ambiguous: drop it
amb = df.groupby("text").label.nunique()
df = df[~df.text.isin(amb[amb > 1].index)]

# ------------------------------------------------------------------ augmentation (train only)
aug = []
tr_pos = df[(df.split == "train") & (df.label == 1)].text.tolist()
tr_doc = df[(df.split == "train") & (df.domain == "doc") & (df.source != "generated_benign")].text.tolist()
for t in RNG.sample(tr_pos, min(3000, len(tr_pos))):
    k = RNG.choice([1, 1, 2])
    ctx = RNG.sample(tr_doc, k)
    pos = RNG.randint(0, k)
    aug.append({"text": " ".join(ctx[:pos] + [t] + ctx[pos:])[:2000], "label": 1, "source": "aug_planted",
                "split": "train", "domain": "inj"})
for t in RNG.sample(tr_pos, min(1500, len(tr_pos))):
    words = t.split()
    style = RNG.choice(["lower", "upper", "lines", "bullet"])
    if style == "lower":
        v = t.lower()
    elif style == "upper":
        v = t.upper()
    elif style == "lines":
        cut = max(1, len(words) // 2)
        v = " ".join(words[:cut]) + "\n" + " ".join(words[cut:])
    else:
        v = "- " + t
    aug.append({"text": norm(v), "label": 1, "source": "aug_format", "split": "train", "domain": "inj"})
# benign counterpart: pairs of real document sentences (so "longer = injection" can't be learned)
for _ in range(3000):
    k = RNG.choice([2, 2, 3])
    aug.append({"text": " ".join(RNG.sample(tr_doc, k))[:2000], "label": 0, "source": "aug_doc_pairs",
                "split": "train", "domain": "doc"})
df = pd.concat([df, pd.DataFrame(aug)], ignore_index=True)

# probes (audit fixtures, CS223 hidden sentence) must not leak into any split
probe_frag = ["ORANGE-CANARY", "museum room monitor"]
df = df[~df.text.str.contains("|".join(probe_frag), regex=True)]

out = os.path.join(D, "ft_dataset.parquet")
df.to_parquet(out, index=False)
summary = df.groupby(["split", "label"]).size().unstack(fill_value=0)
print(summary)
print(df.groupby(["split", "source"]).size().unstack(fill_value=0).T.to_string())
