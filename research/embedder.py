"""bge-small-en-v1.5 sentence embeddings via ONNX Runtime (no torch at inference time).

bge recipe: CLS-token pooling, L2 normalisation. No query instruction prefix (that prefix is for
retrieval queries; here sentences are compared symmetrically / classified).
"""
from __future__ import annotations

import os
import time

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "bge-small-en-v1.5")


class Embedder:
    def __init__(self, onnx_path: str | None = None, max_len: int = 128, threads: int | None = None):
        onnx_path = onnx_path or os.path.join(MODEL_DIR, "onnx", "model.onnx")
        t0 = time.perf_counter()
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.load_s = time.perf_counter() - t0

    def encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        """Embeddings for texts; uses the on-disk cache when enabled (research iterations)."""
        cache = getattr(self, "_cache", None)
        if cache is None:
            return self._encode(texts, batch)
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        if missing:
            for t, v in zip(missing, self._encode(missing, batch)):
                cache[t] = v
            self._dirty = True
        return np.vstack([cache[t] for t in texts]) if texts else np.zeros((0, 384), np.float32)

    def enable_cache(self, path: str) -> None:
        import pickle
        self._cache_path, self._dirty = path, False
        try:
            with open(path, "rb") as f:
                self._cache = pickle.load(f)
        except (OSError, EOFError):
            self._cache = {}

    def save_cache(self) -> None:
        import pickle
        if getattr(self, "_dirty", False):
            with open(self._cache_path, "wb") as f:
                pickle.dump(self._cache, f)
            self._dirty = False

    def _encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok.encode_batch(texts[i:i + batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            h = self.sess.run(None, feed)[0][:, 0]          # CLS
            out.append(h / np.linalg.norm(h, axis=1, keepdims=True))
        return np.vstack(out) if out else np.zeros((0, 384), np.float32)
