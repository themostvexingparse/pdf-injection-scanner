"""Builds notebooks/pdfscan_finetune.ipynb from the cell sources below (keeps the code diffable).

Run:  python notebooks/build_notebook.py
"""
import json
import os

CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# pdfscan: fine-tune the injection-likeness signal (bge-small-en-v1.5) on GPU

**Run on Kaggle** (Settings: *Accelerator = GPU T4*, *Internet = on*) **or Colab** (*Runtime → Change runtime type → T4 GPU*), then *Run all*.

Everything is fetched automatically from the Hugging Face Hub (public, permissively licensed only):

| side | dataset | license |
|---|---|---|
| injections | deepset/prompt-injections | Apache-2.0 |
| injections | Lakera/gandalf_ignore_instructions | MIT |
| injections | Lakera/mosscap_prompt_injection | MIT |
| injections | microsoft/llmail-inject-challenge (phase-2 labelled, `attack_attempt == True`) | MIT |
| injections | reshabhs/SPML_Chatbot_Prompt_Injection | MIT |
| injections | yanismiraoui/prompt_injections | Apache-2.0 |
| injections, **test only (out-of-distribution)** | Lakera/gandalf_summarization | MIT |
| benign documents | Salesforce/wikitext (wikitext-103) | CC BY-SA 3.0 |
| benign documents | FiscalNote/billsum | CC0-1.0 |
| benign documents | launch/gov_report | CC BY 4.0 |
| benign documents | coastalcph/lex_glue (ecthr_a) | CC BY 4.0 |
| benign emails | microsoft/llmail-inject-challenge (false-positive test emails) | MIT |
| benign chat | deepset + SPML benign user messages | as above |

No synthetic or augmented data, no local documents.

**Unit of training = what the scanner scores:** windows of 1-3 consecutive sentences, at most 64 tokens (no context filling).
**Multiple-instance learning:** each original example is a *bag* of windows and its score is the **max** window score.
An injection bag only needs *some* window to score high (its greeting sentences are not mislabelled as attacks);
every window of a benign bag must score low. The scanner decides the same way (max over sentences).

Outputs (`pdfscan_ft_candidates.zip`, one folder per experiment variant): fine-tuned ONNX encoder + tokenizer, `probe.json` (head weights, thresholds,
data provenance, metrics), `exemplars.npz`, `metrics.json`. Install locally with
`python research/accept_models.py pdfscan_ft_candidates.zip` (local acceptance tests), then install the chosen one.
""")

code("""
import subprocess, sys, os
SMOKE = bool(os.environ.get("PDFSCAN_SMOKE"))   # tiny end-to-end run for testing the notebook itself
if not SMOKE:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxruntime", "huggingface_hub>=0.23",
                    "transformers>=4.40", "scikit-learn", "pyarrow", "datasets"], check=True)
import os, re, json, time, math, random, hashlib, zipfile, glob
from pathlib import Path
import numpy as np, pandas as pd, torch
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import roc_auc_score

OUT = next((Path(p) for p in ("/kaggle/working", "/content") if Path(p).exists()), Path.cwd()) / "pdfscan_ft"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "(no GPU: this will be slow)")

CFG = dict(
    base_model="BAAI/bge-small-en-v1.5",
    window_tokens=64, window_max_sents=3, max_windows_per_bag=16,
    epochs=3, bag_batch=64, max_windows_per_step=320, lr_encoder=2e-5, lr_head=1e-3, weight_decay=0.01, warmup=0.06,
    eval_every=300, seed=0,
    # samples per source (bags)
    n_llmail=8000, n_mosscap=5000, n_spml_inj=3000, n_spml_benign=3000,
    n_wiki=15000, n_billsum=8000, n_govreport=8000, n_lexglue=8000,
    # real instruction-heavy web text (tutorials, course pages, how-tos, manuals, FAQs): the previous
    # run had none, so the model learned "addresses the reader = injection"
    n_fineweb=16000, n_c4=12000,
    chat_benign_weight=0.5,
)
# experiment variants (same data, same MIL loss); every variant is exported for local acceptance tests
VARIANTS = [
    dict(name="frozen", freeze_encoder=True, epochs=2),
    dict(name="ft_e1", epochs=1, lr_encoder=2e-5),
    dict(name="ft_e2_lowlr", epochs=2, lr_encoder=1e-5),
    dict(name="ft_freeze6", epochs=2, lr_encoder=2e-5, freeze_layers=6),
]
if SMOKE:
    CFG.update(epochs=1, eval_every=20, n_llmail=150, n_mosscap=150, n_spml_inj=100, n_spml_benign=100,
               n_wiki=300, n_billsum=150, n_govreport=150, n_lexglue=150, n_fineweb=200, n_c4=200, bag_batch=16)
    VARIANTS = [dict(v, epochs=1) for v in VARIANTS]
random.seed(CFG["seed"]); np.random.seed(CFG["seed"]); torch.manual_seed(CFG["seed"])
T0 = time.time()
""")

md("## 1. Fetch base model and datasets")
code("""
# plain local folders (no cache symlinks: those fail on Windows without admin rights)
HF = OUT.parent / "hf_data"

def ds(repo, filename):
    return hf_hub_download(repo, filename, repo_type="dataset", local_dir=HF / repo.replace("/", "__"))

def ds_glob(repo, pattern):
    root = snapshot_download(repo, repo_type="dataset", allow_patterns=[pattern], local_dir=HF / repo.replace("/", "__"))
    return sorted(glob.glob(os.path.join(root, pattern)))

model_dir = snapshot_download(CFG["base_model"], allow_patterns=["*.json", "*.txt", "model.safetensors"],
                              local_dir=HF / CFG["base_model"].replace("/", "__"))
tok = AutoTokenizer.from_pretrained(model_dir)

SRC = {
    "deepset": ds_glob("deepset/prompt-injections", "data/*.parquet"),
    "gandalf": ds_glob("Lakera/gandalf_ignore_instructions", "data/*.parquet"),
    "gandalf_sum": ds_glob("Lakera/gandalf_summarization", "data/*.parquet"),
    "mosscap": ds_glob("Lakera/mosscap_prompt_injection", "data/*.parquet"),
    "llmail": [ds("microsoft/llmail-inject-challenge", "data/labelled_unique_submissions_phase2.json")],
    "llmail_fp": [ds("microsoft/llmail-inject-challenge", "data/emails_for_fp_tests.json")],
    "spml": [ds("reshabhs/SPML_Chatbot_Prompt_Injection", "spml_prompt_injection.csv")],
    "yanis": [ds("yanismiraoui/prompt_injections", "prompt_injections.csv")],
    "wiki": [ds("Salesforce/wikitext", "wikitext-103-raw-v1/train-00000-of-00002.parquet")],
    "billsum": [ds("FiscalNote/billsum", "data/train-00000-of-00001.parquet")],
    "govreport": [ds("launch/gov_report", f) for f in ("data/crs_valid.jsonl", "data/gao_valid.jsonl", "data/crs_test.jsonl")],
    "lexglue": ds_glob("coastalcph/lex_glue", "ecthr_a/train-*.parquet"),
    "c4": [ds("allenai/c4", "en/c4-validation.00000-of-00008.json.gz")],
}
# FineWeb-Edu is huge: stream only the first documents of the 10BT sample
from datasets import load_dataset
fineweb_docs = []
for row in load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True):
    if len(row["text"]) > 400:
        fineweb_docs.append(row["text"][:20000])
    if len(fineweb_docs) >= CFG["n_fineweb"] // 3 + 500:
        break
base_onnx = hf_hub_download(CFG["base_model"], "onnx/model.onnx", local_dir=HF / CFG["base_model"].replace("/", "__"))
print({k: len(v) for k, v in SRC.items()}, f"{time.time()-T0:.0f}s")
""")

md("## 2. Sentence windows (1-3 sentences, <= 64 tokens) and bags")
code("""
SENT_RE = re.compile(r"(?<=[.!?])\\s+(?=[\\"'(\\[A-Z0-9])")

def clean(t):
    t = re.sub(r"([.\\u2024\\u2026\\u00b7_\\-=*])\\1{3,}", " ", str(t))
    return re.sub(r"\\s+", " ", t).strip()

def ntok(s):
    return len(tok(s, add_special_tokens=False)["input_ids"])

def windows(text):
    # greedy groups of consecutive sentences, <= window_tokens and <= window_max_sents;
    # a single over-long sentence is cut into overlapping token windows (as the scanner does)
    W, K = CFG["window_tokens"], CFG["window_max_sents"]
    out, cur, cur_n = [], [], 0
    for s in SENT_RE.split(clean(text)):
        s = s.strip()
        if not s or not any(c.isalpha() for c in s):
            continue
        n = ntok(s)
        if n > W:
            if cur: out.append(" ".join(cur)); cur, cur_n = [], 0
            ids = tok(s, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            for i in range(0, len(ids), W // 2):
                j = min(i + W, len(ids)) - 1
                out.append(s[ids[i][0]:ids[j][1]])
                if j == len(ids) - 1: break
            continue
        if cur and (cur_n + n > W or len(cur) >= K):
            out.append(" ".join(cur)); cur, cur_n = [], 0
        cur.append(s); cur_n += n
    if cur: out.append(" ".join(cur))
    return out

def sentences(text, lo=4, hi=80):
    return [s for s in SENT_RE.split(clean(text)) if lo <= len(s.split()) <= hi]

def hsplit(key, val=0.1, test=0.1):
    r = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if r < test else "val" if r < test + val else "train"
""")

code("""
bags = []   # dict(windows=[...], label, source, split, domain)

def add_bag(text, label, source, split, domain):
    w = windows(text)[:CFG["max_windows_per_bag"]]
    if w:
        bags.append(dict(windows=w, label=int(label), source=source, split=split, domain=domain))

def add_doc_windows(source, docs, per_doc, cap):
    # benign documents: split by DOCUMENT; each 1-3 sentence window is its own bag (all must be low)
    n = 0
    for i, d in enumerate(docs):
        sp = hsplit(f"{source}:{i}")
        ws = windows(d)
        for w in random.sample(ws, min(per_doc, len(ws))):
            bags.append(dict(windows=[w], label=0, source=source, split=sp, domain="doc")); n += 1
        if n >= cap: break

# ---- benign documents
wiki = [l for l in pd.read_parquet(SRC["wiki"][0]).text if len(l.strip()) > 200 and not l.strip().startswith("=")]
random.shuffle(wiki)
add_doc_windows("wikitext", [re.sub(r"\\s+@(.)@\\s+", r"\\1", w) for w in wiki], 3, CFG["n_wiki"])
bill = pd.read_parquet(SRC["billsum"][0]).sample(frac=1, random_state=1)
add_doc_windows("billsum", list(bill.text), 4, CFG["n_billsum"])
def paragraphs(node):
    if isinstance(node, str): return [node]
    if isinstance(node, list): return [p for x in node for p in paragraphs(x)]
    if isinstance(node, dict): return paragraphs(node.get("paragraphs", [])) + paragraphs(node.get("subsections", []))
    return []
gov = [" ".join(paragraphs(json.loads(l).get("reports")))[:30000] for f in SRC["govreport"] for l in open(f, encoding="utf-8")]
random.shuffle(gov)
add_doc_windows("gov_report", gov, 8, CFG["n_govreport"])
lex = pd.read_parquet(SRC["lexglue"][0]).sample(frac=1, random_state=2)
add_doc_windows("lex_glue_ecthr", [" ".join(t) if not isinstance(t, str) else t for t in lex.text], 5, CFG["n_lexglue"])
import gzip
c4 = [json.loads(l)["text"] for l in gzip.open(SRC["c4"][0], "rt", encoding="utf-8")]
c4 = [t for t in c4 if len(t) > 400]
random.shuffle(c4)
add_doc_windows("c4_web", c4, 4, CFG["n_c4"])
add_doc_windows("fineweb_edu", fineweb_docs, 3, CFG["n_fineweb"])
fp = json.load(open(SRC["llmail_fp"][0], encoding="utf-8"))
for i, t in enumerate(fp):
    add_bag(t, 0, "llmail_fp_emails", "train" if i % 2 else "test", "doc")

# ---- chatbot-style benign (down-weighted: not the scanner's domain)
for f in SRC["deepset"]:
    d = pd.read_parquet(f); is_test = "test-" in os.path.basename(f)
    for t, l in zip(d.text, d.label):
        sp = "test" if is_test else hsplit("ds:" + t, val=0.15, test=0)
        add_bag(t, l, "deepset", sp, "inj" if l else "chat")
spml = pd.read_csv(SRC["spml"][0]).dropna(subset=["User Prompt"]).drop_duplicates(subset=["User Prompt"])
for t in spml[spml["Prompt injection"] == 0]["User Prompt"].sample(CFG["n_spml_benign"], random_state=3):
    add_bag(t, 0, "spml", hsplit("spml:" + t), "chat")

# ---- injections
for f in SRC["gandalf"]:
    sp = "test" if "test-" in f else "val" if "validation" in f else "train"
    for t in pd.read_parquet(f).text: add_bag(t, 1, "gandalf_ignore", sp, "inj")
for t in spml[spml["Prompt injection"] == 1]["User Prompt"].sample(CFG["n_spml_inj"], random_state=4):
    add_bag(t, 1, "spml", hsplit("spmli:" + t), "inj")
for f in SRC["mosscap"]:
    sp = "test" if "test-" in f else "val" if "validation" in f else "train"
    m = pd.read_parquet(f).drop_duplicates("prompt")
    m = m[m.prompt.str.split().str.len() >= 8]
    n = CFG["n_mosscap"] if sp == "train" else CFG["n_mosscap"] // 8
    for t in m.prompt.sample(min(n, len(m)), random_state=5): add_bag(t, 1, "mosscap", sp, "inj")
ll = json.load(open(SRC["llmail"][0], encoding="utf-8"))
mails = [k for k, v in ll.items() if v.get("attack_attempt") in (True, "True") and 40 <= len(k) <= 4000]
random.shuffle(mails)
for k in mails[:CFG["n_llmail"]]:
    m = re.match(r"\\s*Subject of the email:\\s*(.*?)\\.?\\s+Body:\\s*(.*)$", k, re.S)
    add_bag((m.group(1) + ". " + m.group(2)) if m else k, 1, "llmail", hsplit("ll:" + k), "inj")
ya = pd.read_csv(SRC["yanis"][0])
for t in ya.prompt_injections.dropna().unique(): add_bag(t, 1, "yanis_multilingual", hsplit("ya:" + t), "inj")
for f in SRC["gandalf_sum"]:
    for t in pd.read_parquet(f).text: add_bag(t, 1, "gandalf_summarization_OOD", "test", "inj")

# drop exact duplicates across labels/splits (first occurrence wins)
seen, uniq = set(), []
for b in bags:
    k = (" ".join(b["windows"]), b["label"])
    if k not in seen: seen.add(k); uniq.append(b)
bags = uniq
summary = pd.DataFrame([{k: b[k] for k in ("label", "source", "split")} | {"windows": len(b["windows"])} for b in bags])
print(summary.groupby(["split", "label"]).size().unstack(fill_value=0))
print(summary.groupby(["source", "split"]).size().unstack(fill_value=0))
print("windows per bag (injections): median", summary[summary.label == 1].windows.median(),
      "p90", summary[summary.label == 1].windows.quantile(.9), f"| {time.time()-T0:.0f}s")
""")

md("## 3. Model: encoder → CLS → L2-normalise → linear head; MIL loss = BCE on max window logit")
code("""
class Clf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_dir)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        torch.nn.init.normal_(self.head.weight, std=0.02); torch.nn.init.zeros_(self.head.bias)
    def embed(self, **b):
        return torch.nn.functional.normalize(self.enc(**b).last_hidden_state[:, 0], dim=-1)
    def forward(self, **b):
        return self.head(self.embed(**b)).squeeze(-1)

def encode_batch(texts):
    return {k: v.to(DEV) for k, v in tok(texts, padding=True, truncation=True,
                                          max_length=CFG["window_tokens"] + 2, return_tensors="pt").items()}

def bag_logits(model, batch_bags):
    flat, owner = [], []
    for i, b in enumerate(batch_bags):
        flat += b["windows"]; owner += [i] * len(b["windows"])
    z = model(**encode_batch(flat)).float()
    owner = torch.tensor(owner, device=DEV)
    out = torch.full((len(batch_bags),), -1e4, device=DEV)
    return out.scatter_reduce(0, owner, z, reduce="amax")

@torch.no_grad()
def score_windows(model, texts, bs=512):
    model.eval(); out = np.zeros(len(texts), np.float32)
    order = np.argsort([len(t) for t in texts])
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
            out[idx] = model(**encode_batch([texts[j] for j in idx])).float().cpu().numpy()
    return out

def bag_scores(model, bag_list):
    flat, owner = [], []
    for i, b in enumerate(bag_list):
        flat += b["windows"]; owner += [i] * len(b["windows"])
    s = score_windows(model, flat)
    res = np.full(len(bag_list), -1e4, np.float32)
    np.maximum.at(res, np.array(owner), s)
    return res

def thr_at(neg, fpr):
    s = np.sort(neg); return float(s[min(len(s) - 1, int(math.ceil((1 - fpr) * len(s))) - 1)])

split = {s: [b for b in bags if b["split"] == s] for s in ("train", "val", "test")}
if SMOKE:   # keep the smoke run short: the frozen/fine-tune loops then take seconds
    split = {s: random.sample(v, min(len(v), 400)) for s, v in split.items()}
def val_metric(model):
    v = split["val"]; s = bag_scores(model, v)
    lab = np.array([b["label"] for b in v]); dom = np.array([b["domain"] for b in v])
    neg = s[(lab == 0) & (dom == "doc")]; pos = s[lab == 1]
    return float(np.mean([np.mean(pos > thr_at(neg, f)) for f in (0.001, 0.005, 0.01)]))
""")

code("""
def train(freeze_encoder=False, freeze_layers=0, epochs=None, lr_encoder=None, tag="", **_):
    model = Clf().to(DEV)
    if freeze_encoder:
        for p in model.enc.parameters(): p.requires_grad = False
    elif freeze_layers:
        # keep the embeddings and the lowest layers as pre-trained (regularisation: less room to learn
        # surface shortcuts such as "imperative mood")
        for p in model.enc.embeddings.parameters(): p.requires_grad = False
        for layer in model.enc.encoder.layer[:freeze_layers]:
            for p in layer.parameters(): p.requires_grad = False
    epochs = epochs or CFG["epochs"]
    lr_encoder = lr_encoder or CFG["lr_encoder"]
    tr = split["train"]
    y = torch.tensor([b["label"] for b in tr], dtype=torch.float32, device=DEV)
    w = torch.tensor([CFG["chat_benign_weight"] if b["domain"] == "chat" else 1.0 for b in tr], device=DEV)
    pos_weight = (y == 0).sum() / (y == 1).sum().clamp(min=1)
    params = [{"params": model.head.parameters(), "lr": CFG["lr_head"]}]
    if not freeze_encoder:
        params.append({"params": [p for p in model.enc.parameters() if p.requires_grad], "lr": lr_encoder})
    opt = torch.optim.AdamW(params, weight_decay=CFG["weight_decay"])

    def make_chunks():
        # batches of bags with similar window counts/lengths (little padding), capped by the TOTAL
        # number of windows: a fixed bag count let 64 long emails x 16 windows exhaust GPU memory
        order = sorted(range(len(tr)), key=lambda i: (len(tr[i]["windows"]), len(tr[i]["windows"][0])))
        chunks, cur, cur_w = [], [], 0
        for i in order:
            nw = len(tr[i]["windows"])
            if cur and (len(cur) >= CFG["bag_batch"] or cur_w + nw > CFG["max_windows_per_step"]):
                chunks.append(cur); cur, cur_w = [], 0
            cur.append(i); cur_w += nw
        if cur: chunks.append(cur)
        return chunks
    base_chunks = make_chunks()
    steps = epochs * len(base_chunks)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / (CFG["warmup"] * steps)) * max(0, (steps - s) / ((1 - CFG["warmup"]) * steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=DEV == "cuda")
    best, best_state, step, hist, t0 = -1, None, 0, [], time.time()
    for ep in range(epochs):
        chunks = list(base_chunks)
        random.shuffle(chunks)
        for idx in chunks:
            model.train()
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                z = bag_logits(model, [tr[i] for i in idx])
            ii = torch.tensor(idx, device=DEV)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y[ii], weight=w[ii], pos_weight=pos_weight)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1
            if step % CFG["eval_every"] == 0 or step == steps:
                m = val_metric(model); hist.append((step, round(m, 4)))
                print(f"{tag} ep {ep+1} step {step}/{steps} loss {loss.item():.4f} val {m:.4f}{' *' if m > best else ''} {time.time()-t0:.0f}s")
                if m > best: best, best_state = m, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"best_val": round(best, 4), "history": hist, "seconds": round(time.time() - t0, 1)}

MODELS, LOGS = {}, {}
for v in VARIANTS:
    m, log = train(**v, tag=f"[{v['name']}]")
    MODELS[v["name"]], LOGS[v["name"]] = m.cpu(), log      # park on CPU; keeps GPU memory free
    torch.cuda.empty_cache() if DEV == "cuda" else None
    print(f"== {v['name']}: best val {log['best_val']} in {log['seconds']}s | total {time.time()-T0:.0f}s")
""")

md("## 4. Test: per-source rates at thresholds calibrated on validation document text")
code("""
def evaluate(model, name):
    v = split["val"]; sv = bag_scores(model, v)
    lab = np.array([b["label"] for b in v]); dom = np.array([b["domain"] for b in v])
    neg = sv[(lab == 0) & (dom == "doc")]
    thr = {"moderate": thr_at(neg, 0.01), "strong": thr_at(neg, 0.001)}
    te = split["test"]; st = bag_scores(model, te); lt = np.array([b["label"] for b in te])
    res = {"thresholds": thr, "auc_test": round(float(roc_auc_score(lt, st)), 4), "by_source": {}}
    df = pd.DataFrame({"source": [b["source"] for b in te], "label": lt, "score": st})
    for (src, l), g in df.groupby(["source", "label"]):
        k = "tpr" if l == 1 else "fpr"
        res["by_source"][f"{src}{'' if l else ' (benign)'}"] = {"n": len(g), "label": int(l),
            f"{k}_moderate": round(float((g.score > thr["moderate"]).mean()), 4),
            f"{k}_strong": round(float((g.score > thr["strong"]).mean()), 4)}
    print(f"\\n== {name}: test AUC {res['auc_test']}")
    print(f"   {'source':40}{'n':>6}   moderate   strong")
    for k, r in sorted(res["by_source"].items(), key=lambda x: (-x[1]["label"], x[0])):
        m = "tpr" if r["label"] else "fpr"
        print(f"   {k:40}{r['n']:>6}   {m} {r[m+'_moderate']:.3f}  {r[m+'_strong']:.3f}")
    return res

R = {"config": CFG, "variants": VARIANTS}
for name, m in MODELS.items():
    m.to(DEV)
    R[name] = evaluate(m, name) | {"train": LOGS[name]}
    m.cpu()
json.dump(R, open(OUT / "metrics.json", "w"), indent=1)
# compact comparison: strong-tier rates per source, all variants side by side
rows = sorted(R[VARIANTS[0]["name"]]["by_source"].items(), key=lambda x: (-x[1]["label"], x[0]))
print(f"\\n{'source (strong tier)':40}" + "".join(f"{v['name']:>14}" for v in VARIANTS))
for k, r in rows:
    m = "tpr" if r["label"] else "fpr"
    print(f"{(m.upper() + ' ' + k):40}" + "".join(f"{R[v['name']]['by_source'][k][m + '_strong']:>14.3f}" for v in VARIANTS))
print(f"{'test AUC':40}" + "".join(f"{R[v['name']]['auc_test']:>14.4f}" for v in VARIANTS))
""")

md("## 5. Export every variant: ONNX encoder (+ parity check), tokenizer, probe.json, exemplars -> one zip")
code("""
import onnxruntime as ort

class EncOnly(torch.nn.Module):
    def __init__(self, enc): super().__init__(); self.enc = enc
    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.enc(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state

DATA_LICENSES = [
    {"name": "deepset/prompt-injections", "license": "Apache-2.0"}, {"name": "Lakera/gandalf_ignore_instructions", "license": "MIT"},
    {"name": "Lakera/mosscap_prompt_injection", "license": "MIT"}, {"name": "microsoft/llmail-inject-challenge", "license": "MIT"},
    {"name": "reshabhs/SPML_Chatbot_Prompt_Injection", "license": "MIT"}, {"name": "yanismiraoui/prompt_injections", "license": "Apache-2.0"},
    {"name": "Salesforce/wikitext (wikitext-103)", "license": "CC BY-SA 3.0"}, {"name": "FiscalNote/billsum", "license": "CC0-1.0"},
    {"name": "launch/gov_report", "license": "CC BY 4.0"}, {"name": "coastalcph/lex_glue ecthr_a", "license": "CC BY 4.0"},
    {"name": "allenai/c4 (en, validation shard 0)", "license": "ODC-By"},
    {"name": "HuggingFaceFW/fineweb-edu (sample-10BT, first documents)", "license": "ODC-By"}]

def export_onnx(model, sub_dir):
    (sub_dir / "onnx").mkdir(parents=True, exist_ok=True)
    enc_eager = AutoModel.from_pretrained(model_dir, attn_implementation="eager")   # SDPA may not export cleanly
    enc_eager.load_state_dict(model.enc.state_dict())
    enc = EncOnly(enc_eager).float().cpu().eval()
    dummy = tok(["export example sentence"], return_tensors="pt")
    path = sub_dir / "onnx" / "model.onnx"
    kw = dict(input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["last_hidden_state"],
              dynamic_axes={n: {0: "batch", 1: "seq"} for n in ("input_ids", "attention_mask", "token_type_ids", "last_hidden_state")},
              opset_version=17, do_constant_folding=True)
    args = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])
    try:
        torch.onnx.export(enc, args, str(path), dynamo=False, **kw)   # classic exporter (newer torch defaults to dynamo)
    except TypeError:
        torch.onnx.export(enc, args, str(path), **kw)
    tok.save_pretrained(sub_dir); model.enc.config.save_pretrained(sub_dir)
    return path

def parity(model, onnx_path):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    texts = [w for b in split["test"][:200] for w in b["windows"]][:200]
    b = tok(texts, padding=True, truncation=True, max_length=CFG["window_tokens"] + 2, return_tensors="np")
    h = sess.run(None, {k: b[k].astype(np.int64) for k in ("input_ids", "attention_mask", "token_type_ids")})[0][:, 0]
    h = h / np.linalg.norm(h, axis=1, keepdims=True)
    W = model.head.weight.detach().float().numpy()[0]; B = float(model.head.bias.detach().float())
    with torch.no_grad():
        ref = model.float().cpu().eval()(**{k: torch.tensor(v) for k, v in b.items()}).numpy()
    return float(np.max(np.abs(h @ W + B - ref)))

CAND = OUT / "candidates"
for v in VARIANTS:
    name = v["name"]; model = MODELS[name].float().cpu().eval(); res = R[name]
    cdir = CAND / name; cdir.mkdir(parents=True, exist_ok=True)
    if v.get("freeze_encoder"):
        # encoder untouched: the scanner's existing base ONNX is used; identify it by hash
        sub, onnx_path = "bge-small-en-v1.5", Path(base_onnx)
    else:
        sub = f"bge-small-en-v1.5-pdfscan-{name}"
        onnx_path = export_onnx(model, cdir / sub)
    p = parity(model, onnx_path); print(f"[{name}] ONNX vs torch max |logit diff| {p:.2e}")
    assert p < 1e-3, f"{name}: ONNX does not match the trained model"
    W = model.head.weight.detach().float().numpy()[0]; B = float(model.head.bias.detach().float())
    thr = res["thresholds"]
    probe = {
        "format": "pdfscan-embedding-probe/1", "created": time.strftime("%Y-%m-%d"), "variant": v,
        "embedding_model": {"name": f"BAAI/bge-small-en-v1.5 ({name}; MIL on 1-3 sentence windows)",
                            "model_subdir": sub, "onnx_sha256": hashlib.sha256(open(onnx_path, "rb").read()).hexdigest(),
                            "dim": len(W), "pooling": "cls", "normalize": True, "max_tokens": CFG["window_tokens"], "precision": "fp32"},
        "weights": [round(float(x), 7) for x in W], "bias": round(B, 7),
        "thresholds": {"moderate": round(thr["moderate"], 5), "strong": round(thr["strong"], 5)},
        "calibration": "validation document windows of all benign document sources (prose, legal, government, web, educational web), "
                       "split by document; moderate = 1% FPR, strong = 0.1% FPR",
        "training": {"method": "multiple-instance learning, bag score = max window logit", "config": CFG,
                     "best_val": LOGS[name]["best_val"], "train_seconds": LOGS[name]["seconds"]},
        "training_data": DATA_LICENSES, "held_out_metrics": res["by_source"], "test_auc": res["auc_test"],
    }
    json.dump(probe, open(cdir / "probe.json", "w"), indent=1)
    # exemplars (explanations only): embeddings of short training injections, stored with ids, not text
    ex = [(i, b) for i, b in enumerate(split["train"]) if b["label"] == 1 and len(b["windows"]) == 1][:3000]
    model.to(DEV).eval(); vecs = []
    with torch.no_grad():
        for i in range(0, len(ex), 256):
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEV == "cuda"):
                vecs.append(model.embed(**encode_batch([b["windows"][0] for _, b in ex[i:i + 256]])).float().cpu().numpy())
    model.cpu()
    np.savez_compressed(cdir / "exemplars.npz", vectors=np.vstack(vecs).astype(np.float16),
                        ids=np.array([f"{b['source']}:train-bag:{i}" for i, b in ex]))

zp = OUT.parent / "pdfscan_ft_candidates.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(OUT / "metrics.json", "metrics.json")
    for p in CAND.rglob("*"):
        if p.is_file(): z.write(p, p.relative_to(OUT))
print("artifacts:", zp, f"{zp.stat().st_size/2**20:.1f} MB | total {time.time()-T0:.0f}s")
try:
    from google.colab import files; files.download(str(zp))
except Exception:
    print("Kaggle: download pdfscan_ft_candidates.zip from the Output panel (/kaggle/working).")
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfscan_finetune.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, len(CELLS), "cells")
