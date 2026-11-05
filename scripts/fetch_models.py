"""Download the embedding models the scanner's optional embedding signal uses (~250 MB).

  python scripts/fetch_models.py              # base model + fine-tuned ensemble member
  python scripts/fetch_models.py --base-only  # base model only (the scanner then runs without the member)

  * BAAI/bge-small-en-v1.5 (MIT), pinned revision, from the Hugging Face Hub -> models/bge-small-en-v1.5
  * the fine-tuned ensemble member (release asset of this repository)
                                               -> models/bge-small-en-v1.5-pdfscan-ft_e2_lowlr
Each ONNX file is checked against the SHA-256 recorded in pdfscan/models/probe*.json: the thresholds
were calibrated on exactly these files. The scanner works without any model (the embedding signal is
then skipped and reported as unavailable).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
PROBES = os.path.join(ROOT, "pdfscan", "models")

BASE_REPO = "BAAI/bge-small-en-v1.5"
BASE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BASE_FILES = ["onnx/model.onnx", "tokenizer.json", "tokenizer_config.json", "config.json",
              "special_tokens_map.json", "vocab.txt"]
MEMBER_URL = ("https://github.com/themostvexingparse/pdf-injection-scanner/releases/download/"
              "v0.1.0/bge-small-en-v1.5-pdfscan-ft_e2_lowlr.zip")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def expected_sha(probe_file: str) -> str:
    with open(os.path.join(PROBES, probe_file)) as f:
        return json.load(f)["embedding_model"]["onnx_sha256"]


def base():
    dest = os.path.join(MODELS, "bge-small-en-v1.5")
    onnx = os.path.join(dest, "onnx", "model.onnx")
    want = expected_sha("probe.json")
    if os.path.exists(onnx) and sha256(onnx) == want:
        print("base model: already present"); return
    print(f"base model: {BASE_REPO}@{BASE_REVISION[:10]}")
    for f in BASE_FILES:
        path = os.path.join(dest, *f.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = fetch(f"https://huggingface.co/{BASE_REPO}/resolve/{BASE_REVISION}/{f}")
        with open(path, "wb") as out:
            out.write(data)
    if sha256(onnx) != want:
        sys.exit("base model: ONNX hash does not match pdfscan/models/probe.json")
    print("base model: ok")


def member():
    name = "bge-small-en-v1.5-pdfscan-ft_e2_lowlr"
    onnx = os.path.join(MODELS, name, "onnx", "model.onnx")
    want = expected_sha("probe_ft_e2_lowlr.json")
    if os.path.exists(onnx) and sha256(onnx) == want:
        print("ensemble member: already present"); return
    print("ensemble member: fine-tuned ft_e2_lowlr")
    with zipfile.ZipFile(io.BytesIO(fetch(MEMBER_URL))) as z:
        z.extractall(MODELS)
    if sha256(onnx) != want:
        sys.exit("ensemble member: ONNX hash does not match pdfscan/models/probe_ft_e2_lowlr.json")
    print("ensemble member: ok")


if __name__ == "__main__":
    base()
    if "--base-only" not in sys.argv:
        member()
