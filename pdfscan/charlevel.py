"""§1F Character-level tricks in any extracted text."""
from __future__ import annotations

import base64
import binascii
import re
import unicodedata

ZERO_WIDTH = set("​‌‍⁠᠎﻿­")
BIDI = set("‪‫‬‭‮⁦⁧⁨⁩‎‏؜")
B64_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
HEX_RE = re.compile(r"\b(?:[0-9a-fA-F]{2}[\s:]?){16,}\b")


def _script(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "?"
    for s in ("LATIN", "CYRILLIC", "GREEK", "ARMENIAN", "CHEROKEE"):
        if name.startswith(s):
            return s
    if "FULLWIDTH" in name or "MATHEMATICAL" in name:
        return "COMPAT"
    return "OTHER"


def _printable_ratio(s: str) -> float:
    return sum(c.isprintable() or c in "\n\r\t" for c in s) / max(1, len(s))


def scan(text: str) -> tuple[list[dict], list[str]]:
    """Returns (issues, decoded_hidden_texts). Decoded texts should be fed to semantic analysis."""
    issues, decoded = [], []
    tags = [c for c in text if 0xE0000 <= ord(c) <= 0xE007F]
    if tags:
        hidden = "".join(chr(ord(c) - 0xE0000) for c in tags if 0x20 <= ord(c) - 0xE0000 < 0x7F)
        issues.append({"mechanism": "unicode_tag_characters", "count": len(tags), "decoded": hidden})
        if hidden:
            decoded.append(hidden)
    zw = sum(c in ZERO_WIDTH for c in text)
    if zw:
        issues.append({"mechanism": "zero_width_characters", "count": zw})
    bd = sum(c in BIDI for c in text)
    if bd:
        issues.append({"mechanism": "bidi_controls", "count": bd})
    var = sum(0xFE00 <= ord(c) <= 0xFE0F or 0xE0100 <= ord(c) <= 0xE01EF for c in text)
    if var > 3:
        issues.append({"mechanism": "variation_selectors", "count": var})
    mixed = []
    for w in re.findall(r"\w{3,}", text):
        scripts = {_script(c) for c in w if c.isalpha()} - {"OTHER", "?"}
        if len(scripts) > 1 and "LATIN" in scripts:
            mixed.append(w)
    if mixed:
        issues.append({"mechanism": "mixed_script_homoglyphs", "count": len(mixed), "examples": mixed[:10]})
        decoded.append(unicodedata.normalize("NFKC", " ".join(mixed)))
    if any(_script(c) == "COMPAT" for c in text):
        nf = unicodedata.normalize("NFKC", text)
        if nf != text:
            issues.append({"mechanism": "compatibility_characters"})
            decoded.append(nf)
    for m in B64_RE.finditer(text):
        s = m.group()
        if s.isalpha() or s.isdigit():
            continue
        try:
            raw = base64.b64decode(s + "=" * (-len(s) % 4), validate=False)
            dec = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if len(dec) >= 12 and _printable_ratio(dec) > 0.95 and re.search(r"[a-zA-Z]{3,}\s", dec):
            issues.append({"mechanism": "base64_payload", "decoded": dec[:500]})
            decoded.append(dec)
    for m in HEX_RE.finditer(text):
        hx = re.sub(r"[\s:]", "", m.group())
        try:
            dec = bytes.fromhex(hx).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if _printable_ratio(dec) > 0.95 and re.search(r"[a-zA-Z]{3,}", dec):
            issues.append({"mechanism": "hex_payload", "decoded": dec[:500]})
            decoded.append(dec)
    return issues, decoded
