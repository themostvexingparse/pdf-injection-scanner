"""§3.7 Benign-pattern filter, producer-agnostic and order-aware.

Hidden text is judged by what it ADDS to what a person can see. The hidden text is aligned against
the human view (text of glyphs measured as visible, plus OCR where it ran) and the contiguous
stretches that do not occur there, in order, are extracted as "novel spans":

  * a hidden token counts as seen only if it is part of a run of NGRAM consecutive tokens that also
    occurs consecutively in the human view. Word-set comparisons are not enough: an attacker can pad
    a payload with a copy of the visible text (low share of new words) or compose it entirely from
    words that appear somewhere on the page (no new words at all).
  * matching tolerates common OCR confusions and merged/split words, so an OCR layer over its scan
    and ActualText for drawn glyphs come out redundant.

Hidden text with no meaningful novel span is redundant (downgraded, never dropped). Anything novel
stays a finding; semantic analysis then runs on the novel spans themselves.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from .util import Finding, norm_compact

NGRAM = 3
# mismatches whose extracted text can't vouch for itself (same ToUnicode): only OCR can confirm
OCR_ONLY_MECHS = {"cmap_remap", "multi_char_glyph_mapping", "extracted_differs_from_rendered"}
_TOKEN_RE = re.compile(r"[^\W_]+")
_OCR_FOLD = str.maketrans({"0": "o", "1": "l", "5": "s", "|": "l", "!": "l"})


def _norm(tok: str) -> str:
    return unicodedata.normalize("NFKC", tok).lower()


def _skel(tok: str) -> str:
    """OCR-tolerant canonical token (common confusions folded)."""
    return tok.translate(_OCR_FOLD).replace("rn", "m").replace("vv", "w").replace("cl", "d")


def _tokens_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(_norm(m.group()), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


class HumanView:
    """N-gram index of what a person can perceive on one page."""

    def __init__(self, texts: list[str]):
        self.grams: dict[int, set] = {n: set() for n in range(1, NGRAM + 1)}
        self.vocab: set[str] = set()
        compact = []
        for t in texts:
            toks = [_skel(_norm(m.group())) for m in _TOKEN_RE.finditer(t)]
            self.vocab.update(toks)
            for n in range(1, NGRAM + 1):
                for i in range(len(toks) - n + 1):
                    self.grams[n].add(tuple(toks[i:i + n]))
            compact.append(_skel(norm_compact(t)))
        self.compact = "|".join(compact)

    def __bool__(self):
        return bool(self.vocab)


def novel_spans(hidden: str, hv: HumanView) -> tuple[list[str], int]:
    """Contiguous stretches of `hidden` not found, in order, in the human view."""
    toks = _tokens_with_offsets(hidden)
    if not toks:
        return [], 0
    sk = [_skel(t) for t, _, _ in toks]
    n = min(NGRAM, len(sk))
    covered = [False] * len(sk)
    for i in range(len(sk) - n + 1):
        if tuple(sk[i:i + n]) in hv.grams[n]:
            for j in range(i, i + n):
                covered[j] = True
    # uncovered runs -> spans; a run can still be redundant if it occurs with different word
    # breaks (OCR merges/splits words): compare it whitespace-free against the human view
    spans, i = [], 0
    while i < len(sk):
        if covered[i]:
            i += 1
            continue
        j = i
        while j < len(sk) and not covered[j]:
            j += 1
        run = "".join(sk[i:j])
        if not (len(run) >= 4 and run in hv.compact):
            spans.append(hidden[toks[i][1]:toks[j - 1][2]])
        i = j
    return spans, len(sk)


def meaningful(span: str, hv: HumanView) -> bool:
    """Is a novel span real content (vs. OCR noise / stray characters)?"""
    toks = [_skel(_norm(m.group())) for m in _TOKEN_RE.finditer(span)]
    if len(toks) >= 2:
        return True
    if not toks:
        return False
    t = toks[0]
    if len(t) < 3 and not t.isdigit():
        return False
    # a single word is noise if it is a near-miss of something visible (OCR misread)
    if not t.isdigit() and any(fuzz.ratio(t, v) >= 80 for v in hv.vocab if abs(len(v) - len(t)) <= 2):
        return False
    return True


def apply(findings: list[Finding], human_by_page: dict[int, dict]) -> None:
    """human_by_page[p] = {"visible": text of visible glyphs, "ocr": OCR text or ""}."""
    cache: dict[tuple, HumanView] = {}
    for f in findings:
        if f.category == "offpage_content" and "alt_text" in f.mechanism and f.benign is None:
            f.benign = "accessibility_text"
            continue
        if f.category not in ("hidden_text", "mismatched_text") or not f.page or f.benign:
            continue
        p = f.page - 1
        h = human_by_page.get(p, {})
        ocr_only = bool(set(f.mechanism) & OCR_ONLY_MECHS)
        key = (p, ocr_only)
        if key not in cache:
            cache[key] = HumanView([h.get("ocr", "")] if ocr_only else [h.get("visible", ""), h.get("ocr", "")])
        hv = cache[key]
        if not hv:
            f.detail["verification"] = ("needs OCR to compare with what is rendered (run with --ocr full)"
                                        if ocr_only else "nothing visible on the page to compare with")
            f.review = True
            continue
        spans, total = novel_spans(f.text, hv)
        novel = [s for s in spans if meaningful(s, hv)]
        f.detail.update(total_words=total, novel_spans=novel[:20],
                        novel_words=sum(len(_TOKEN_RE.findall(s)) for s in novel))
        if total == 0:
            f.benign, f.severity, f.review = "no_word_content", "low", False
        elif not novel:
            f.benign, f.severity, f.review = "redundant_with_visible_content", "low", False
        elif "actual_text_override" in f.mechanism:
            f.mechanism.append("actual_text_not_visible")
            f.severity = "high"
        elif "invisible_render_mode" in f.detail.get("reason_counts", {}):
            f.mechanism.append("invisible_text_novel_vs_render")
