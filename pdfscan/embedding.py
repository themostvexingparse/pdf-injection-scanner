"""Embedding signal: how instruction-like (AI-directed) a piece of text reads.

One signal among several, never the classifier:
  * it only ever ADDS evidence; a low score clears nothing (paraphrases the model misses are still
    caught by the deterministic novelty rules)
  * it never decides visibility, and never on its own escalates to MALICIOUS (policy.py)

Text is scored per sentence (long sentences in overlapping token windows): embedding a whole
paragraph dilutes an injection sentence until it is missed. A unit's score is its strongest sentence.

Model: BAAI/bge-small-en-v1.5, fp32 ONNX, CLS pooling, L2-normalised; probe = logistic regression
(pdfscan/models/probe.json, trained by research/train_probe.py). Exemplars (training injections)
are used only to name the nearest known example in explanations.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

import numpy as np

PKG = os.path.dirname(os.path.abspath(__file__))
# PDFSCAN_PROBE lets a candidate probe be evaluated without installing it (research/accept_models.py);
# its exemplars.npz is read from the same folder
PROBE = os.environ.get("PDFSCAN_PROBE") or os.path.join(PKG, "models", "probe.json")
EXEMPLARS = os.path.join(os.path.dirname(PROBE), "exemplars.npz")
MODEL_DIR = os.environ.get("PDFSCAN_EMBED_MODEL",
                           os.path.join(os.path.dirname(PKG), "models", "bge-small-en-v1.5"))
WINDOW, STRIDE = 64, 32          # tokens, for sentences longer than WINDOW
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


class Unavailable(Exception):
    pass


TIER_RANK = {"none": 0, "moderate": 1, "strong": 2}


class EmbeddingSignal:
    """One model + probe. The primary probe may list ensemble members ("ensemble": [probe files]),
    each a full EmbeddingSignal of its own; a text's tier is the highest tier any member gives it."""

    def __init__(self, check_hash: bool = True, probe_path: str | None = None, load_members: bool = True):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as e:
            raise Unavailable(f"missing dependency: {e.name}")
        probe_path = probe_path or PROBE
        if not os.path.exists(probe_path):
            raise Unavailable(f"probe not found ({probe_path})")
        with open(probe_path) as f:
            self.probe = json.load(f)
        self.name = self.probe.get("name") or os.path.splitext(os.path.basename(probe_path))[0]
        # the probe names the model it was calibrated with (base or fine-tuned folder under models/)
        sub = self.probe.get("embedding_model", {}).get("model_subdir")
        beside_probe = os.path.join(os.path.dirname(probe_path), sub) if sub else ""
        self.model_dir = os.environ.get("PDFSCAN_EMBED_MODEL") or (
            beside_probe if sub and os.path.isdir(beside_probe) else
            os.path.join(os.path.dirname(PKG), "models", sub) if sub else MODEL_DIR)
        onnx = os.path.join(self.model_dir, "onnx", "model.onnx")
        if not os.path.exists(onnx):
            raise Unavailable(f"model not found ({onnx})")
        if check_hash:
            h = hashlib.sha256()
            with open(onnx, "rb") as f:
                for b in iter(lambda: f.read(1 << 20), b""):
                    h.update(b)
            if h.hexdigest() != self.probe["embedding_model"]["onnx_sha256"]:
                # thresholds were calibrated on exactly this model; a different file (e.g. int8)
                # shifts scores silently
                raise Unavailable("ONNX model does not match the one the probe was calibrated on")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = min(4, os.cpu_count() or 4)
        self.sess = ort.InferenceSession(onnx, so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json"))
        # visible-sentence gate (nearest training injection cosine); depends on the embedding space,
        # so a fine-tuned model carries its own calibrated value in probe.json
        self.visible_min_cosine = float((self.probe.get("visible_gate") or {}).get("min_exemplar_cosine", 0.74))
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self.tok.enable_truncation(128)
        self.w = np.array(self.probe["weights"], dtype=np.float32)
        self.b = float(self.probe["bias"])
        self.thr = self.probe["thresholds"]
        self.ex_vec, self.ex_ids = None, None
        ex_path = os.path.join(os.path.dirname(probe_path), self.probe.get("exemplars", "exemplars.npz"))
        if os.path.exists(ex_path):
            z = np.load(ex_path)
            self.ex_vec, self.ex_ids = z["vectors"].astype(np.float32), list(z["ids"])
        self.members: list[EmbeddingSignal] = []
        self.missing_members: list[str] = []
        if load_members:
            for m in self.probe.get("ensemble", []):
                try:
                    self.members.append(EmbeddingSignal(check_hash, os.path.join(os.path.dirname(probe_path), m),
                                                        load_members=False))
                except Unavailable as e:
                    # an absent optional member (e.g. model not downloaded) must not disable the
                    # primary signal: scan with what is available and report it
                    self.missing_members.append(f"{m}: {e}")

    # ------------------------------------------------------------------ segmentation
    def pieces(self, text: str) -> list[str]:
        """Sentences; overlong ones cut into overlapping token windows."""
        out = []
        for s in _SENT_RE.split(re.sub(r"\s+", " ", text).strip()):
            s = s.strip()
            if len(s) < 3 or not any(c.isalpha() for c in s):
                continue
            enc = self.tok.encode(s, add_special_tokens=False)
            if len(enc.ids) <= WINDOW:
                out.append(s)
                continue
            offs = enc.offsets
            for i in range(0, len(offs), STRIDE):
                j = min(i + WINDOW, len(offs)) - 1
                out.append(s[offs[i][0]:offs[j][1]])
                if j == len(offs) - 1:
                    break
        return out

    # ------------------------------------------------------------------ scoring
    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = []
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding per batch
        for k in range(0, len(order), 64):
            idx = order[k:k + 64]
            enc = self.tok.encode_batch([texts[i] for i in idx])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0][:, 0]
            vecs.append((idx, h / np.linalg.norm(h, axis=1, keepdims=True)))
        out = np.zeros((len(texts), 384), np.float32)
        for idx, h in vecs:
            out[idx] = h
        return out

    def tier(self, score: float) -> str:
        return "strong" if score > self.thr["strong"] else "moderate" if score > self.thr["moderate"] else "none"

    def score_units(self, units: list[str], use_members: bool = True) -> list[dict]:
        """Score many texts at once. Returns per unit: score, tier, the strongest sentence, nearest
        exemplar (explanation only), which member produced the result and every member's tier.
        use_members=False scores with the primary model only (the scanner does this for visible
        sentences: they are ~95% of the work, their flags are review-only, and the fine-tuned member's
        gain is on hidden/unreviewed text)."""
        results = self._score_own(units)
        if not self.members or not use_members:
            return results
        others = [m._score_own(units) for m in self.members]
        merged = []
        for i, r in enumerate(results):
            cands = [r] + [o[i] for o in others]
            best = max(cands, key=lambda x: (TIER_RANK[x["tier"]], x["score"] is not None))
            best = dict(best)
            best["members"] = {c["member"]: c["tier"] for c in cands}
            merged.append(best)
        return merged

    def _score_own(self, units: list[str]) -> list[dict]:
        res = self._score_units_single(units)
        for r in res:
            r["member"] = self.name
            r["visible_min_cosine"] = self.visible_min_cosine
        return res

    def _score_units_single(self, units: list[str]) -> list[dict]:
        all_pieces, owner = [], []
        for u, text in enumerate(units):
            for p in self.pieces(text):
                all_pieces.append(p)
                owner.append(u)
        results = [{"score": None, "tier": "none", "sentence": None, "pieces": 0} for _ in units]
        if not all_pieces:
            return results
        V = self._embed(all_pieces)
        s = V @ self.w + self.b
        for i, u in enumerate(owner):
            r = results[u]
            r["pieces"] += 1
            if r["score"] is None or s[i] > r["score"]:
                r.update(score=float(s[i]), sentence=all_pieces[i], _vec=V[i])
        for r in results:
            if r["score"] is None:
                r.pop("_vec", None)
                continue
            r["tier"] = self.tier(r["score"])
            r["probability"] = round(1 / (1 + math.exp(-r["score"])), 3)
            v = r.pop("_vec")
            if self.ex_vec is not None and r["tier"] != "none":
                sims = self.ex_vec @ v
                j = int(np.argmax(sims))
                r["nearest_example"] = {"id": self.ex_ids[j], "cosine": round(float(sims[j]), 3)}
            r["score"] = round(r["score"], 3)
        return results


_INSTANCE: EmbeddingSignal | None = None
_ERROR: str | None = None


def get() -> EmbeddingSignal | None:
    """Shared instance, or None if unavailable (reason in last_error())."""
    global _INSTANCE, _ERROR
    if _INSTANCE is None and _ERROR is None:
        try:
            _INSTANCE = EmbeddingSignal()
        except Unavailable as e:
            _ERROR = str(e)
        except Exception as e:  # never let the optional signal break a scan
            _ERROR = f"{type(e).__name__}: {e}"
    return _INSTANCE


def last_error() -> str | None:
    return _ERROR
