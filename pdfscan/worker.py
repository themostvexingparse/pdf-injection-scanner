"""Parsing stage. Runs in its own (sandboxed, time-limited) process and never calls models.

Usage: python -E -P -m pdfscan.worker <pdf> <out.json> [--zoom Z] [--ocr off|targeted|full] [--no-faint]
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
import sys
import tempfile
import time
import traceback

import fitz
from rapidfuzz import fuzz

from . import benign, charlevel, contrib, differential, extract, structure, tracer, vision
from .intake import Quarantine, intake
from .util import Finding, norm_words, is_substantive


# Overall progress budget: setup 0-10%, pages 10-95%, finalisation 95-100%.
SETUP, PAGES = 0.10, 0.85


def emit(frac: float | None, msg: str, label: str = "") -> None:
    """Progress line for the parent process (parsed from stderr, never shown raw)."""
    sys.stderr.write("@@PROGRESS " + json.dumps({"f": frac, "m": (label + msg) if label else msg}) + "\n")
    sys.stderr.flush()


def scan(path: str, zoom: float = 2.0, ocr_mode: str = "targeted", faint: bool = True,
         depth: int = 0, max_depth: int = 2, label: str = "") -> dict:
    t0 = time.time()
    top = depth == 0

    def prog(frac, msg):
        emit(frac if top else None, msg, label)
    prog(0.0, "intake and validation")
    result = {"file": os.path.abspath(path), "status": "ok", "findings": [], "pages": [],
              "children": [], "errors": {}, "ocr_available": False}
    try:
        doc, info, findings = intake(path)
    except Quarantine as q:
        result.update(status="quarantine", quarantine_reason=str(q))
        return result
    result["info"] = info
    try:
        result["metadata"] = structure.metadata_view(doc)
    except Exception as e:
        result["errors"]["metadata"] = f"{type(e).__name__}: {e}"[:300]
    with open(path, "rb") as f:
        data = f.read()

    prog(0.02, "structural analysis")
    s_findings, embedded = structure.analyze(doc, data)
    findings += s_findings
    texts, ext_errors, e_findings = extract.agent_view(
        doc, data, lambda f, m: prog(0.04 + 0.06 * f, m))
    findings += e_findings
    result["errors"]["extractors"] = ext_errors
    result["extractors"] = sorted(texts)

    try:
        hidden_layers = {o["name"] for o in (doc.get_ocgs() or {}).values() if not o.get("on", True)}
    except Exception:
        hidden_layers = set()

    # OCR is a safety net, not the core engine (it is ~30x slower than everything else):
    #   off      - never
    #   targeted - only crops where nothing else can decide (hidden text over images, i.e. an
    #              OCR layer to verify; glyph/extraction mismatches to confirm)
    #   full     - additionally whole-page OCR: rendered-vs-extracted comparison, faint-text pass
    ocr_ok = ocr_mode != "off" and vision.ocr_engine() is not None
    result["ocr_mode"] = ocr_mode
    result["ocr_available"] = ocr_ok
    result["ocr_regions"] = 0
    ocr_by_page, vis_by_page = {}, {}
    page_findings: list[Finding] = []
    n = max(1, doc.page_count)

    def pprog(pno, sub, msg):
        prog(SETUP + PAGES * (pno + sub) / n, f"page {pno + 1}/{doc.page_count}: {msg}")

    for pno in range(doc.page_count):
        page = doc[pno]
        pprog(pno, 0.0, "rendering")
        img = vision.render(page, zoom, annots=False)
        try:
            img_nt = vision.render_without_text(data, pno, zoom)
        except Exception as e:
            result["errors"][f"page{pno + 1}_notext_render"] = str(e)
            img_nt = img
        glyphs = tracer.trace_page(doc, page, img, img_nt, zoom, hidden_layers)
        # primary visibility decision: leave-one-out ink attribution (pdfium)
        pprog(pno, 0.3, "measuring each text object's rendered contribution")
        multi_runs, cinfo = [], {}
        try:
            chars, cinfo = contrib.analyze_page(data, pno, zoom)
            multi_runs = cinfo.pop("multi_char_glyphs", [])
        except Exception as e:
            result["errors"][f"page{pno + 1}_contrib"] = f"{type(e).__name__}: {e}"[:300]
            chars = []
        glyphs += tracer.apply_contributions(glyphs, chars)

        pf = differential.glyph_findings(pno, glyphs)
        pf += differential.multi_char_findings(pno, multi_runs)
        safe = _visible_text(page, glyphs)
        vis_by_page[pno] = safe

        ocr_lines: list[dict] = []
        if ocr_ok and ocr_mode == "full":
            pprog(pno, 0.5, "full-page OCR (safety net)")
            img_human = vision.render(page, zoom, annots=True)
            ocr_lines = vision.ocr(img_human, zoom, (page.rect.x0, page.rect.y0))
            pf += differential.ocr_mismatch(pno, page, glyphs, ocr_lines)
            if faint:
                pprog(pno, 0.75, "faint-text OCR pass (safety net)")
                pf += _faint_text(pno, img_human, zoom, page, ocr_lines, safe)
        elif ocr_ok:
            regions = _ocr_regions(page, pf)
            if regions:
                pprog(pno, 0.5, f"targeted OCR of {len(regions)} region(s)")
                for r in regions:
                    crop = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=r, alpha=False,
                                           annots=True, colorspace=fitz.csRGB)
                    ocr_lines += vision.ocr(vision.pix_to_np(crop), zoom, (r.x0, r.y0))
                result["ocr_regions"] += len(regions)
        ocr_by_page[pno] = ocr_lines

        # only findings that attribute text to a hiding mechanism count as explanations
        explained = [f.text for f in pf + findings if f.page == pno + 1
                     and f.category in ("hidden_text", "mismatched_text", "offpage_content")]
        u = differential.unexplained_agent_text(pno, texts, ocr_lines, explained, safe)
        if u:
            if hidden_layers:
                u.mechanism.append("document_has_layers_off_by_default")
            pf.append(u)
        page_findings += pf
        counts = {"visible": 0, "doubtful": 0, "invisible": 0}
        methods: dict[str, int] = {}
        for g in glyphs:
            if not g.ch.isspace():
                counts[g.label] += 1
                methods[g.method] = methods.get(g.method, 0) + 1
        result["pages"].append({"page": pno + 1, "glyphs": counts, "methods": methods,
                                "contrib": cinfo, "ocr_lines": len(ocr_lines),
                                "safe_text": safe,
                                "human_ocr_text": "\n".join(l["text"] for l in ocr_lines)})
    findings += page_findings

    prog(SETUP + PAGES, "redundancy analysis and character-level checks")
    benign.apply(findings, {p: {"visible": vis_by_page.get(p, ""),
                                "ocr": "\n".join(l["text"] for l in ocr_by_page.get(p, []))}
                            for p in range(doc.page_count)})
    findings += _char_level(findings, texts)

    for name, blob in embedded:
        prog(0.96, f"scanning embedded file {name}")
        findings += _embedded(name, blob, zoom, ocr_mode, faint, depth, max_depth, result)

    result["findings"] = [f.to_dict() for f in findings]
    result["elapsed_s"] = round(time.time() - t0, 2)
    doc.close()
    return result


def _visible_text(page: fitz.Page, glyphs) -> str:
    """Reading-ordered text containing only glyphs labelled visible (safe-text export)."""
    def k(ch, o):
        return (ch, round(o[0], 1), round(o[1], 1))

    keep = set()
    # characters a visible glyph draws beyond its first one (ligatures: the "i" of "fi"); the two
    # MuPDF APIs place these at different positions, so they are matched by sequence instead
    cont_after: dict[tuple, list[str]] = {}
    base = None
    for g in glyphs:
        if g.cont:
            if base is not None and g.label == "visible":
                cont_after.setdefault(base, []).append(g.ch)
            continue
        base = k(g.ch, g.origin) if g.label == "visible" else None
        if g.label == "visible":
            keep.add(base)
    lines_out = []
    try:
        # PRESERVE_LIGATURES is essential: explicit flags replace MuPDF's defaults, and without it
        # rawdict splits a ligature glyph (U+FB01 "ﬁ") into "f"+"i" while the glyph trace keeps one
        # "ﬁ" -> keys never match and every ligature word loses letters ("file" -> "le").
        raw = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP
                            | fitz.TEXT_PRESERVE_LIGATURES)
    except Exception:
        return ""
    for b in raw.get("blocks", []):
        for ln in b.get("lines", []):
            s = []
            pending: list[str] = []
            for sp in ln.get("spans", []):
                for c in sp.get("chars", []):
                    key = k(c["c"], c["origin"])
                    if key in keep:
                        s.append(c["c"])
                        pending = list(cont_after.get(key, []))
                    elif pending and c["c"] == pending[0]:
                        s.append(c["c"])
                        pending.pop(0)
                    elif c["c"].isspace() and s:
                        s.append(" ")
                        pending = []
                    else:
                        pending = []
            # removed hidden words leave their neighbouring spaces behind; collapse them
            # expand typographic ligatures (U+FB00-FB06) so the exported text reads "file", not "ﬁle"
            line = re.sub(r"[ﬀ-ﬆ]", lambda m: unicodedata.normalize("NFKC", m.group()), "".join(s))
            line = re.sub(r" {2,}", " ", line).strip()
            if line:
                lines_out.append(line)
        lines_out.append("")
    return "\n".join(lines_out).strip()


def _faint_text(pno, img, zoom, page, ocr_lines, safe) -> list[Finding]:
    """§1E: text a vision model may read in a contrast-enhanced view that plain OCR/humans miss."""
    enh = vision.ocr(vision.enhance_faint(img), zoom, (page.rect.x0, page.rect.y0))
    seen = " ".join(l["text"] for l in ocr_lines) + " " + safe
    seen_words = set(norm_words(seen, 3))
    seen_boxes = [fitz.Rect(l["bbox"]) for l in ocr_lines]
    out = []
    for l in enh:
        words = norm_words(l["text"], 3)
        if l["score"] < 0.8 or len(words) < 2:
            continue
        r = fitz.Rect(l["bbox"])
        # enhancement garbles normal-contrast text; only regions plain OCR found empty count
        if any((r & b).get_area() > 0.3 * max(r.get_area(), 1e-6) for b in seen_boxes):
            continue
        new = [w for w in words if w not in seen_words
               and not any(fuzz.ratio(w, s) >= 85 for s in seen_words if abs(len(s) - len(w)) <= 2)]
        if len(new) >= max(2, 0.6 * len(words)):
            out.append(Finding("vision", ["faint_text_in_render"], "medium", page=pno + 1,
                               bbox=[round(v, 2) for v in l["bbox"]], text=l["text"], review=True,
                               extractors=["ocr_enhanced"],
                               detail={"note": "readable only after contrast enhancement"}))
    return out


def _char_level(findings: list[Finding], texts: dict) -> list[Finding]:
    out = []
    # (dedup group, readable location, page, text)
    sources = [("agent_view", f"page text ({n})", None, "\n".join(p)) for n, p in texts.items()]
    sources += [(f.category, f.source or (f"page {f.page}" if f.page else f.category), f.page, f.text)
                for f in findings if f.text]
    seen = set()
    for group, src, page, text in sources:
        issues, decoded = charlevel.scan(text)
        for iss in issues:
            key = (iss["mechanism"], iss.get("decoded", ""), group)
            if key in seen:
                continue
            seen.add(key)
            sev = "high" if iss["mechanism"] in ("unicode_tag_characters", "bidi_controls") else \
                  "medium" if iss.get("decoded") else "low"
            out.append(Finding("char_level", [iss["mechanism"]], sev, source=src, page=page,
                               text=iss.get("decoded", "") or "",
                               detail={k: v for k, v in iss.items() if k not in ("mechanism", "decoded")}))
    return out


def _embedded(name, blob, zoom, ocr_mode, faint, depth, max_depth, result) -> list[Finding]:
    if blob[:1024].find(b"%PDF-") >= 0:
        if depth >= max_depth:
            return [Finding("attachment", ["nesting_limit_reached"], "medium", text=name, review=True)]
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(blob)
        try:
            child = scan(tf.name, zoom, ocr_mode, faint, depth + 1, max_depth, label=f"[{name}] ")
        finally:
            os.unlink(tf.name)
        child["file"] = f"embedded:{name}"
        result["children"].append(child)
        return []
    try:
        txt = blob.decode("utf-8")
        if is_substantive(txt):
            return [Finding("offpage_content", ["attachment_text"], "medium", text=txt[:20000],
                            source=f"attachment:{name}")]
        return []
    except UnicodeDecodeError:
        return [Finding("attachment", ["attachment_not_scanned"], "medium", text=name,
                        source=f"attachment:{name}", detail={"bytes": len(blob)})]


def _ocr_regions(page: fitz.Page, page_findings: list[Finding]) -> list[fitz.Rect]:
    """Crops where only OCR can decide: hidden text lying over an image (is it an OCR layer that
    matches the scan beneath?) and glyph/extraction mismatches (what do the glyphs really show?)."""
    try:
        images = [fitz.Rect(i["bbox"]) for i in page.get_image_info()]
    except Exception:
        images = []
    rects = []
    for f in page_findings:
        if not f.bbox:
            continue
        r = fitz.Rect(f.bbox)
        if r.is_empty:
            continue
        need = (f.category == "mismatched_text" and set(f.mechanism) & benign.OCR_ONLY_MECHS) or (
            f.category == "hidden_text"
            and any((r & im).get_area() >= 0.3 * r.get_area() for im in images))
        if need:
            rects.append((r + (-4, -4, 4, 4)) & page.rect)
    merged: list[fitz.Rect] = []
    for r in sorted(rects, key=lambda r: (r.y0, r.x0)):
        for m in merged:
            if m.intersects(r + (-6, -6, 6, 6)):
                m |= r
                break
        else:
            merged.append(fitz.Rect(r))
    if sum(m.get_area() for m in merged) > 0.6 * page.rect.get_area():
        return [fitz.Rect(page.rect)]
    return [m for m in merged if not m.is_empty]


def main(argv):
    path, out = argv[0], argv[1]
    zoom = float(argv[argv.index("--zoom") + 1]) if "--zoom" in argv else 2.0
    ocr_mode = argv[argv.index("--ocr") + 1] if "--ocr" in argv else "targeted"
    try:
        res = scan(path, zoom, ocr_mode, "--no-faint" not in argv)
    except Exception as e:
        res = {"file": path, "status": "quarantine",
               "quarantine_reason": f"parser crash: {type(e).__name__}: {e}",
               "traceback": traceback.format_exc()[-3000:]}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1:])
