"""§3.6 Differential engine: agent view minus human view, with reasons."""
from __future__ import annotations

import re
import unicodedata

import fitz
from rapidfuzz import fuzz

from .tracer import Glyph, group_runs, join_glyphs
from .util import Finding, norm_compact, norm_words, is_substantive, segmentable


def meaningful(text: str) -> bool:
    """Does extracted text read as language (vs. glyph-id junk from missing/broken ToUnicode, math
    symbol soup, private-use codes)? Script-agnostic apart from a vowel test on Latin words."""
    s = [c for c in text if not c.isspace()]
    if len(s) < 4:
        return False
    bad = sum(c == "�" or 0xE000 <= ord(c) <= 0xF8FF or unicodedata.category(c) in ("Cc", "Co", "Cn")
              for c in s)
    if bad / len(s) > 0.1:
        return False
    if sum(c.isalpha() for c in s) / len(s) < 0.6:
        return False
    words = re.findall(r"[^\W\d_]{2,}", text)
    if len(words) < 2:
        return False
    latin = [w for w in words if all(ord(c) < 0x250 for c in w)]
    if latin:
        vow = sum(bool(re.search(r"[aeiouyAEIOUY]", w)) for w in latin) / len(latin)
        if vow < 0.6:
            return False
    return True


def glyph_findings(pno: int, glyphs: list[Glyph]) -> list[Finding]:
    out = []
    for run in group_runs(glyphs, lambda g: g.label != "visible"):
        if not is_substantive(run.text):
            continue
        mech = sorted(run.reasons, key=lambda k: -run.reasons[k])
        out.append(Finding("hidden_text", mech or ["not_perceptible"],
                           "high" if run.label == "invisible" else "medium",
                           page=pno + 1, bbox=run.bbox, text=run.text,
                           extractors=list(run.methods),
                           review=(run.label == "doubtful" or "render_engines_disagree" in run.reasons
                                   or "mupdf_fallback" in run.methods),
                           detail={"visibility": run.label, "reason_counts": run.reasons}))
    if sum(1 for g in glyphs if g.remap and g.label == "visible") >= 3:
        for run in group_runs(glyphs, lambda g: g.remap and g.label == "visible"):
            if len(norm_compact(run.text)) >= 2:
                out.append(Finding("mismatched_text", ["cmap_remap"], "high", page=pno + 1,
                                   bbox=run.bbox, text=run.text, extractors=["font_cmap_check"],
                                   detail={"note": "drawn glyph id differs from the font's own cmap "
                                                   "entry for the extracted character"}))
    return out


def multi_char_findings(pno: int, runs: list) -> list[Finding]:
    return [Finding("mismatched_text", ["multi_char_glyph_mapping"], "high", page=pno + 1,
                    bbox=[round(v, 2) for v in bbox], text=text, extractors=["pdfium"],
                    detail={"chars_from_one_glyph": len(text)})
            for text, bbox in runs if is_substantive(text)]


def ocr_mismatch(pno: int, page: fitz.Page, glyphs: list[Glyph], ocr_lines: list[dict]) -> list[Finding]:
    """Visible text whose extracted characters differ from what OCR reads at the same place."""
    out = []
    vis = [g for g in glyphs if g.label == "visible" and not g.ch.isspace()]
    for ln in ocr_lines:
        if ln["score"] < 0.85 or len(norm_compact(ln["text"])) < 5:
            continue
        box = fitz.Rect(ln["bbox"]) + (-2, -2, 2, 2)
        inside = [g for g in vis if box.contains(fitz.Point((g.bbox[0] + g.bbox[2]) / 2,
                                                            (g.bbox[1] + g.bbox[3]) / 2))]
        if len(inside) < 4:
            continue
        inside.sort(key=lambda g: (round(g.origin[1] / 3), g.origin[0]))
        extracted = join_glyphs(inside)
        a, b = norm_compact(extracted), norm_compact(ln["text"])
        if not a or not b:
            continue
        ratio = fuzz.ratio(a, b)
        if ratio >= 55:
            continue
        f = Finding("mismatched_text", ["extracted_differs_from_rendered"], "high",
                    page=pno + 1, bbox=[round(v, 2) for v in box], text=extracted,
                    extractors=["pymupdf_trace", "ocr"], review=ratio >= 40,
                    detail={"ocr_text": ln["text"], "similarity": round(ratio, 1)})
        if not meaningful(extracted):
            # junk extraction (no/broken ToUnicode, math, symbol fonts): a quality issue, not a
            # payload, unless it reads as language
            f.mechanism = ["unextractable_text"]
            f.severity, f.review = "info", False
            f.benign = "extraction_not_meaningful"
        out.append(f)
    return out


def unexplained_agent_text(pno: int, texts: dict[str, list[str]], ocr_lines: list[dict],
                           explained: list[str], visible_glyph_text: str) -> Finding | None:
    """Words some extractor returns that neither OCR, visible glyphs nor a reported finding explain."""
    ocr_text = " ".join(l["text"] for l in ocr_lines)
    human = set(norm_words(ocr_text, 3)) | set(norm_words(visible_glyph_text, 3))
    known = set()
    for t in explained:
        known |= set(norm_words(t, 3))
    # Whitespace-insensitive views: extractors (and OCR) merge or split words differently, e.g.
    # pypdf's "thestructuralstyle" for "the structural style". Compare character streams instead.
    human_compact = norm_compact(visible_glyph_text) + "|" + norm_compact(ocr_text)
    known_compact = "|".join(norm_compact(t) for t in explained)
    vocab = set(norm_words(visible_glyph_text + " " + ocr_text + " " + " ".join(explained), 1))
    unexplained: dict[str, set[str]] = {}
    for name, pages in texts.items():
        for w in norm_words(pages[pno] if pno < len(pages) else "", 4):
            if w in human or w in known or w in known_compact or (len(w) >= 6 and w in human_compact):
                continue
            if any(fuzz.ratio(w, h) >= 85 for h in human if abs(len(h) - len(w)) <= 2):
                continue
            # visible words glued together by an extractor that dropped the spaces
            if segmentable(w, vocab):
                continue
            # tolerate OCR misreads inside a merged token
            if len(w) >= 8 and fuzz.partial_ratio(w, human_compact) >= 90:
                continue
            unexplained.setdefault(w, set()).add(name)
    if len(unexplained) < 3:
        return None
    best = max(texts, key=lambda n: sum(n in v for v in unexplained.values()))
    lines = []
    for line in (texts[best][pno] if pno < len(texts[best]) else "").splitlines():
        ws = norm_words(line, 4)
        if ws and sum(w in unexplained for w in ws) >= 0.5 * len(ws):
            lines.append(line.strip())
    words = sorted(unexplained)
    text = ("\n".join(lines) or " ".join(words))[:3000]
    f = Finding("hidden_text", ["agent_only_text_unexplained"], "medium", page=pno + 1,
                text=text, review=True,
                extractors=sorted(set().union(*unexplained.values())),
                detail={"count": len(words), "note": "returned by extractors but not perceptible in "
                                                     "the render and not attributed by the tracer"})
    if not meaningful(text):
        f.mechanism.append("not_meaningful")
        f.severity, f.review, f.benign = "low", False, "extraction_not_meaningful"
    return f
