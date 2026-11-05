"""§3.3 Agent-view extraction: run several common extractors and keep the union."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile

from .util import Finding, norm_compact, norm_words, segmentable


def _pymupdf(doc, data):
    return [p.get_text("text") for p in doc]


def _pdfminer(doc, data):
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer
    pages = []
    for layout in extract_pages(io.BytesIO(data)):
        pages.append("\n".join(el.get_text() for el in layout if isinstance(el, LTTextContainer)))
    return pages


def _pypdf(doc, data):
    import pypdf
    r = pypdf.PdfReader(io.BytesIO(data), strict=False)
    return [(p.extract_text() or "") for p in r.pages]


def _pdfium(doc, data):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(data)
    try:
        out = []
        for i in range(len(pdf)):
            tp = pdf[i].get_textpage()
            out.append(tp.get_text_range())
        return out
    finally:
        pdf.close()


def _poppler(doc, data):
    exe = shutil.which("pdftotext")
    if not exe:
        raise RuntimeError("pdftotext not installed")
    # xpdf's pdftotext does not read stdin, so go through temp files (works for poppler too)
    with tempfile.TemporaryDirectory() as td:
        src, dst = os.path.join(td, "in.pdf"), os.path.join(td, "out.txt")
        with open(src, "wb") as f:
            f.write(data)
        r = subprocess.run([exe, "-enc", "UTF-8", src, dst], capture_output=True, timeout=120)
        if r.returncode != 0 or not os.path.exists(dst):
            raise RuntimeError(r.stderr.decode("utf-8", "replace")[:200])
        with open(dst, encoding="utf-8", errors="replace") as f:
            pages = f.read().split("\f")
    return pages[: doc.page_count] + [""] * max(0, doc.page_count - len(pages))


EXTRACTORS = {"pymupdf": _pymupdf, "pdfminer": _pdfminer, "pypdf": _pypdf,
              "pdfium": _pdfium, "poppler": _poppler}


def agent_view(doc, data: bytes, progress=None) -> tuple[dict[str, list[str]], dict[str, str], list[Finding]]:
    """Returns ({extractor: [page_text]}, {extractor: error}, findings)."""
    texts, errors, findings = {}, {}, []
    for k, (name, fn) in enumerate(EXTRACTORS.items()):
        if progress:
            progress(k / len(EXTRACTORS), f"extracting text ({name})")
        try:
            pages = fn(doc, data)
            if len(pages) != doc.page_count:
                findings.append(Finding("structural", ["extractor_page_count_mismatch"], "medium",
                                        extractors=[name], review=True,
                                        detail={"expected": doc.page_count, "got": len(pages)}))
                pages = (pages + [""] * doc.page_count)[: doc.page_count]
            texts[name] = pages
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"[:300]
    findings += _disagreement(texts, doc.page_count)
    return texts, errors, findings


def _disagreement(texts: dict[str, list[str]], npages: int) -> list[Finding]:
    """Words that some extractors return and others don't (parser-differential signal)."""
    out = []
    names = list(texts)
    if len(names) < 2:
        return out
    for p in range(npages):
        sets = {n: set(norm_words(texts[n][p], 3)) for n in names}
        union = set().union(*sets.values())
        if not union:
            continue
        # Extractors differ a lot in where they put whitespace (merged "thestructuralstyle",
        # split "thestruc turalstyle"), so a word only counts as missing from an extractor if its
        # characters don't occur in that extractor's text with all whitespace removed.
        compact = {n: norm_compact(texts[n][p]) for n in names}
        missing = {}
        for n in names:
            # ...nor if it is just words this extractor itself returns, glued together (e.g. a
            # code listing's line number fused to the code: '11void' = '11' + 'void')
            vocab = set(norm_words(texts[n][p], 1))
            m = {w for w in union - sets[n] if w not in compact[n] and not segmentable(w, vocab)}
            if m:
                missing[n] = m
        total_missing = set().union(*missing.values()) if missing else set()
        if len(total_missing) >= max(3, 0.05 * len(union)):
            only_in = {}
            for w in total_missing:
                have = tuple(sorted(n for n in names if w in sets[n]))
                only_in.setdefault(",".join(have), []).append(w)
            out.append(Finding("extractor_disagreement", ["parser_differential"], "low",
                               page=p + 1, extractors=names,
                               text=" ".join(sorted(total_missing))[:3000],
                               detail={"words_by_extractor_set": {k: sorted(v)[:30] for k, v in only_in.items()},
                                       "union_words": len(union)}))
    return out
