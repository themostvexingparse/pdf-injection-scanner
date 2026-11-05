"""Shared helpers: findings, defanging, text normalisation, PDF string parsing."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field, asdict
from typing import Any

SEVERITY_ORDER = {"info": 0, "low": 1, "medium": 2, "high": 3}


@dataclass
class Finding:
    category: str                      # hidden_text, mismatched_text, offpage_content, structural, ...
    mechanism: list[str]               # specific reasons, e.g. ["near_zero_opacity"]
    severity: str = "low"
    page: int | None = None            # 1-based
    bbox: list[float] | None = None
    extractors: list[str] = field(default_factory=list)
    text: str = ""                     # raw attacker-controlled text; never printed raw
    source: str = ""                   # where it came from (e.g. "annotation:/Contents")
    benign: str | None = None          # benign-pattern match (downgrades, never drops)
    review: bool = False               # needs human review (doubtful / methods disagree)
    detail: dict[str, Any] = field(default_factory=dict)
    semantic: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_INVISIBLE_CATS = {"Cf", "Cc", "Co", "Cn", "Zl", "Zp"}
_URL_RE = re.compile(r"\b(https?|ftp|file)://", re.I)


def defang(text: str, limit: int = 300) -> str:
    """Make attacker text safe to show: visible code points for invisibles, broken URLs, truncated."""
    out = []
    for ch in text[: limit * 2]:
        if ch == "\n":
            out.append("\n")          # line structure is kept; it carries meaning
        elif ch in "\t\r":
            out.append(" ")
        elif unicodedata.category(ch) in _INVISIBLE_CATS or ch in "​‌‍⁠﻿":
            out.append(f"<U+{ord(ch):04X}>")
        else:
            out.append(ch)
    s = "".join(out)
    s = _URL_RE.sub(lambda m: m.group(1).replace("t", "x", 2) + "[:]//", s)
    s = re.sub(r"(?<=\w)\.(?=(com|net|org|io|ai|ru|cn|xyz|dev|app)\b)", "[.]", s)
    if len(s) > limit:
        s = s[:limit] + f"... [+{len(text) - limit} chars]"
    return s


def norm_words(text: str, min_len: int = 1) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return [w for w in re.findall(r"[^\W_]+", text) if len(w) >= min_len]


def norm_compact(text: str) -> str:
    return "".join(norm_words(text))


def segmentable(w: str, vocab: set[str], min_piece: int = 1) -> bool:
    """Can token w be split entirely into >=2 vocabulary words? Extractors that drop spaces glue
    neighbouring words together ('foren' = 'for' + 'en', '11void' = line number '11' + 'void')."""
    n = len(w)
    ok = [False] * (n + 1)
    parts = [0] * (n + 1)
    ok[0] = True
    for i in range(1, n + 1):
        for j in range(max(0, i - 30), i - min_piece + 1):
            if ok[j] and w[j:i] in vocab and (not ok[i] or parts[j] + 1 > parts[i]):
                ok[i], parts[i] = True, parts[j] + 1
    return ok[n] and parts[n] >= 2


def is_substantive(text: str) -> bool:
    """True if text has at least one letter/digit (not just whitespace/punctuation)."""
    return any(ch.isalnum() for ch in text)


def decode_pdf_string(raw: bytes) -> str:
    """Decode a PDF text string (UTF-16BE with BOM, UTF-8 with BOM, else PDFDoc/latin-1)."""
    if raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", "replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", "replace")
    return raw.decode("latin-1")


_ESC = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\f",
        b"(": b"(", b")": b")", b"\\": b"\\"}


def parse_literal_at(data: bytes, i: int) -> tuple[bytes, int]:
    """Parse a PDF literal string starting at data[i] == '('. Returns (bytes, index after ')')."""
    assert data[i:i + 1] == b"("
    depth, i, out = 1, i + 1, bytearray()
    n = len(data)
    while i < n:
        c = data[i:i + 1]
        if c == b"\\":
            nxt = data[i + 1:i + 2]
            if nxt in _ESC:
                out += _ESC[nxt]; i += 2; continue
            m = re.match(rb"[0-7]{1,3}", data[i + 1:i + 4])
            if m:
                out.append(int(m.group(), 8) & 0xFF); i += 1 + len(m.group()); continue
            if nxt in (b"\r", b"\n"):
                i += 2; continue
            i += 1; continue
        if c == b"(":
            depth += 1
        elif c == b")":
            depth -= 1
            if depth == 0:
                return bytes(out), i + 1
        out += c
        i += 1
    return bytes(out), n


def parse_hex_at(data: bytes, i: int) -> tuple[bytes, int]:
    j = data.find(b">", i)
    j = len(data) if j < 0 else j
    hx = re.sub(rb"[^0-9A-Fa-f]", b"", data[i + 1:j])
    if len(hx) % 2:
        hx += b"0"
    return bytes.fromhex(hx.decode()), j + 1


def wcag_contrast(rgb1, rgb2) -> float:
    """WCAG contrast ratio of two sRGB colours given as 0..1 floats."""
    def lum(c):
        def ch(v):
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = (ch(float(x)) for x in c)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    a, b = lum(rgb1), lum(rgb2)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)
