"""§3.5 Glyph-level visibility tracer.

Decision vs explanation:
  * The visibility LABEL of a glyph comes from rendered pixels only: primarily the leave-one-out
    ink attribution in contrib.py (pdfium), falling back to MuPDF's all-text-removed diff for glyphs
    pdfium does not attribute.
  * The drawing-state checks below (render mode, opacity, colour, page boxes, layers, occlusion,
    collapsed metrics, font cmap) only EXPLAIN why a glyph is invisible. They never decide on their
    own, because they depend on metrics producers write inconsistently (widths, sizes, bboxes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz
import numpy as np

from .util import wcag_contrast

T = {
    "microscopic_pt": 1.0,
    "zero_opacity": 0.05,
    "low_opacity": 0.25,
    "contrast_invisible": 1.2,
    "contrast_doubtful": 1.8,
    "occlusion_cover": 0.9,
    "px_noise": 10,
    "match_dist_pt": 1.5,
}


@dataclass
class Glyph:
    ch: str
    bbox: tuple
    origin: tuple
    size: float
    seqno: int
    span_idx: int
    reasons: list[str] = field(default_factory=list)   # explanations (drawing state)
    px: str = "visible"                                 # MuPDF all-text diff (fallback)
    px_contrast: float = 0.0
    label: str = "visible"
    method: str = "mupdf_fallback"
    remap: bool = False
    cont: bool = False                                  # ligature continuation (gid -1)
    dir: tuple = (1.0, 0.0)                             # writing direction of the line


def _to_rgb(color):
    if color is None:
        return (0.0, 0.0, 0.0)
    c = list(color)
    if len(c) == 1:
        return (c[0], c[0], c[0])
    if len(c) == 4:
        k = c[3]
        return tuple((1 - min(1, x + k)) for x in c[:3])
    return tuple(c[:3])


class FontChecker:
    """Compare drawn glyph id with the font's own cmap for the extracted Unicode.

    Several subsets of one font often share a base name (e.g. IXFAWU+LMRoman10-Regular and
    PVTTLK+LMRoman10-Regular), each numbering its glyphs differently, while the glyph trace only
    reports the shared name. All candidates are kept; a remap is reported only if NONE of them maps
    the character to the glyph actually drawn."""

    def __init__(self, doc: fitz.Document, page: fitz.Page):
        self.fonts: dict[str, list] = {}
        for f in page.get_fonts(full=True):
            xref, ext, ftype, basefont, name = f[0], f[1], f[2], f[3], f[4]
            if ftype == "Type3" or ext in ("n/a", ""):
                continue
            try:
                buf = doc.extract_font(xref)[3]
                if not buf:
                    continue
                font = fitz.Font(fontbuffer=buf)
            except Exception:
                continue
            for key in {basefont, basefont.split("+")[-1], name}:
                self.fonts.setdefault(key, []).append(font)

    def is_remapped(self, fontname: str, ch: str, gid: int) -> bool:
        """True only if some candidate font maps ch to a glyph AND none maps it to the drawn gid."""
        cands = self.fonts.get(fontname) or self.fonts.get(fontname.split("+")[-1]) or []
        expected = set()
        for font in cands:
            try:
                g = font.has_glyph(ord(ch), fallback=False)
            except Exception:
                g = 0
            if g == gid:
                return False
            if g:
                expected.add(g)
        return bool(expected)


def trace_page(doc, page: fitz.Page, img_full: np.ndarray, img_notext: np.ndarray,
               zoom: float, hidden_layers: set[str]) -> list[Glyph]:
    try:
        spans = page.get_texttrace()
    except Exception:
        return []
    page_rect = page.rect
    mb, cb = fitz.Rect(page.mediabox), fitz.Rect(page.cropbox)
    media_in_page = fitz.Rect(mb.x0 - cb.x0, mb.y0 - cb.y0, mb.x1 - cb.x0, mb.y1 - cb.y0)

    occluders = []
    try:
        for d in page.get_drawings():
            if d.get("fill") is not None and (d.get("fill_opacity") or 1) >= 0.9 and d.get("type") in ("f", "fs"):
                occluders.append((d["seqno"], fitz.Rect(d["rect"]), "shape"))
    except Exception:
        pass
    try:
        for i, (kind, r) in enumerate(page.get_bboxlog()):
            if kind in ("fill-image", "fill-imgmask"):
                occluders.append((i, fitz.Rect(r), "image"))
    except Exception:
        pass

    fc = FontChecker(doc, page)
    H, W = img_full.shape[:2]
    glyphs: list[Glyph] = []

    for si, s in enumerate(spans):
        chars = s.get("chars") or ()
        stype = s.get("type", 0)
        opacity = s.get("opacity", 1.0)
        rgb = _to_rgb(s.get("color"))
        layer = s.get("layer") or ""
        seq = s.get("seqno", -1)
        prev_origin = None
        for k, (ucs, gid, origin, bbox) in enumerate(chars):
            ch = chr(ucs) if ucs >= 0 else "�"
            r = fitz.Rect(bbox)
            g = Glyph(ch, tuple(r), tuple(origin), float(s.get("size", 0)), seq, si,
                      cont=(gid == -1 and k > 0), dir=tuple(s.get("dir") or (1.0, 0.0)))
            glyphs.append(g)
            if ch.isspace() or g.cont:
                continue
            ex = g.reasons
            if stype == 3:
                ex.append("invisible_render_mode")
            if opacity < T["zero_opacity"]:
                ex.append("zero_opacity")
            elif opacity < T["low_opacity"]:
                ex.append("low_opacity")
            if r.width < 0.05 or r.height < 0.05:
                ex.append("collapsed_metrics")
            if 0 < g.size < T["microscopic_pt"]:
                ex.append("tiny_font_size")
            if not r.intersects(page_rect) and not (r.is_empty and page_rect.contains(r.tl)):
                ex.append("outside_cropbox")
                if r.intersects(media_in_page):
                    ex.append("between_cropbox_and_mediabox")
            if layer and layer in hidden_layers:
                ex.append("hidden_layer")
            if prev_origin is not None and abs(origin[0] - prev_origin[0]) < 0.01 \
                    and abs(origin[1] - prev_origin[1]) < 0.01:
                ex.append("stacked_glyphs")
            prev_origin = origin
            for oseq, orect, kind in occluders:
                if oseq > seq and r.width > 0 and r.height > 0:
                    inter = fitz.Rect(r) & orect
                    if not inter.is_empty and inter.get_area() >= T["occlusion_cover"] * r.get_area():
                        ex.append("covered_by_later_shape" if kind == "shape" else "covered_by_later_image")
                        break
            if gid >= 0 and ucs > 32 and not (k + 1 < len(chars) and chars[k + 1][1] == -1):
                if fc.is_remapped(s.get("font", ""), ch, gid):
                    g.remap = True

            # fallback pixel method (MuPDF, all text removed at once)
            x0, y0 = int(np.floor((r.x0 - page_rect.x0) * zoom)), int(np.floor((r.y0 - page_rect.y0) * zoom))
            x1, y1 = int(np.ceil((r.x1 - page_rect.x0) * zoom)), int(np.ceil((r.y1 - page_rect.y0) * zoom))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(W, max(x1, x0 + 1)), min(H, max(y1, y0 + 1))
            if x0 >= W or y0 >= H or x1 <= 0 or y1 <= 0:
                g.px, g.px_contrast = "invisible", 1.0
                continue
            a = img_full[y0:y1, x0:x1].astype(np.int16)
            b = img_notext[y0:y1, x0:x1].astype(np.int16)
            if stype != 3 and opacity >= T["zero_opacity"]:
                bg = np.median(b.reshape(-1, 3), axis=0) / 255.0
                c = wcag_contrast(rgb, bg)
                if c < T["contrast_invisible"]:
                    ex.append("fill_color_matches_background")
                elif c < T["contrast_doubtful"]:
                    ex.append("fill_color_low_contrast")
            diff = np.abs(a - b).max(axis=2)
            if diff.max() <= T["px_noise"]:
                g.px, g.px_contrast = "invisible", 1.0
            else:
                yy, xx = np.unravel_index(np.argmax(diff), diff.shape)
                pc = wcag_contrast(a[yy, xx] / 255.0, b[yy, xx] / 255.0)
                g.px_contrast = round(pc, 3)
                g.px = ("invisible" if pc < T["contrast_invisible"] else
                        "doubtful" if pc < T["contrast_doubtful"] else "visible")
    return glyphs


def apply_contributions(glyphs: list[Glyph], chars) -> list[Glyph]:
    """Set each glyph's label from leave-one-out measurements; returns pdfium-only glyphs to append."""
    by_ch: dict[str, list[int]] = {}
    for i, c in enumerate(chars):
        if c.label is not None and not c.ch.isspace():
            by_ch.setdefault(c.ch, []).append(i)
    used = set()
    for g in glyphs:
        if g.ch.isspace() or g.cont:
            continue
        best, bd = None, T["match_dist_pt"]
        for i in by_ch.get(g.ch, ()):
            if i in used:
                continue
            c = chars[i]
            d = abs(c.origin[0] - g.origin[0]) + abs(c.origin[1] - g.origin[1])
            if d < bd:
                best, bd = i, d
        if best is None:
            g.label = g.px
            g.method = "mupdf_fallback"
            if g.label != "visible":
                g.reasons = ["no_visible_pixels" if g.px == "invisible" else "low_pixel_contrast"] + g.reasons
            continue
        used.add(best)
        c = chars[best]
        g.method = "leave_one_out"
        g.label = c.label
        g.reasons = list(c.reasons) + g.reasons
        if c.label == "visible" and g.px == "invisible":
            g.label = "doubtful"
            g.reasons.append("render_engines_disagree")
        elif c.label == "invisible" and g.px == "visible":
            g.reasons.append("overdrawn_by_other_text")
    # ligature continuations inherit from the glyph that drew them
    for i, g in enumerate(glyphs):
        if g.cont:
            prev = next((p for p in reversed(glyphs[:i]) if p.span_idx == g.span_idx and not p.cont
                         and not p.ch.isspace()), None)
            if prev is not None:
                g.label, g.reasons, g.method = prev.label, list(prev.reasons), prev.method
    # characters pdfium extracts that MuPDF does not: agents using pdfium-based tools see them
    extra = []
    for i, c in enumerate(chars):
        if i in used or c.ch.isspace() or c.label is None or c.label == "visible":
            continue
        g = Glyph(c.ch, c.bbox, c.origin, 0.0, -1, -1000 - c.obj, reasons=list(c.reasons) + ["pdfium_only"],
                  label=c.label, method="leave_one_out")
        extra.append(g)
    return extra


@dataclass
class Run:
    text: str
    bbox: list
    label: str
    reasons: dict
    remap: bool = False
    methods: tuple = ()


WORD_GAP_EM = 0.15    # gap (in font sizes) above which two glyphs are separate words
LINE_SHIFT_EM = 0.6   # baseline shift (in font sizes) treated as a new line (superscripts stay)


def _em(g: Glyph) -> float:
    """Effective font size: declared size, or the glyph box when the size is missing/scaled away."""
    w, h = abs(g.bbox[2] - g.bbox[0]), abs(g.bbox[3] - g.bbox[1])
    box = (abs(g.dir[0]) * h + abs(g.dir[1]) * w) / 1.2
    return max(g.size, box, 0.1)


def separator(a: Glyph, b: Glyph) -> str:
    """Whitespace implied by geometry between consecutive glyphs a -> b: '', ' ' or '\\n'.

    Many producers (TeX, Skia, PDF libraries) position words without drawing space glyphs, so
    spacing must be reconstructed from where glyphs sit, measured along the line's own direction.
    """
    dx, dy = a.dir
    ox, oy = b.origin[0] - a.origin[0], b.origin[1] - a.origin[1]
    along = ox * dx + oy * dy
    perp = -ox * dy + oy * dx
    em = _em(a)
    if abs(perp) > LINE_SHIFT_EM * em or tuple(a.dir) != tuple(b.dir):
        return "\n"
    adv = abs(dx) * abs(a.bbox[2] - a.bbox[0]) + abs(dy) * abs(a.bbox[3] - a.bbox[1])
    gap = along - adv
    if along < -em:          # jumped backwards on the same baseline: new column/segment
        return " "
    return " " if gap > WORD_GAP_EM * em else ""


def join_glyphs(glyphs: list[Glyph]) -> str:
    """Text of a glyph sequence with whitespace reconstructed from geometry."""
    out, prev = [], None
    for g in glyphs:
        if g.ch.isspace():
            if g.ch == "\n" or (out and out[-1] not in " \n"):
                out.append("\n" if g.ch == "\n" else " ")
            continue
        if prev is not None and out and out[-1] not in " \n":
            out.append(separator(prev, g))
        elif prev is not None and out and out[-1] == " " and separator(prev, g) == "\n":
            out[-1] = "\n"
        out.append(g.ch)
        prev = g
    return re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r" {2,}", " ", "".join(out))).strip()


def group_runs(glyphs: list[Glyph], want) -> list[Run]:
    """Group consecutive glyphs matching predicate `want` into runs (whitespace bridges)."""
    runs, cur = [], []

    def flush():
        if cur and any(not g.ch.isspace() for g in cur):
            while cur and cur[-1].ch.isspace():
                cur.pop()
            rect = fitz.Rect()
            reasons: dict = {}
            labels, methods = set(), set()
            for g in cur:
                if not g.ch.isspace():
                    gr = fitz.Rect(g.bbox)
                    rect |= gr if not gr.is_empty else fitz.Rect(g.origin, g.origin)
                    labels.add(g.label)
                    methods.add(g.method)
                    for rs in g.reasons:
                        reasons[rs] = reasons.get(rs, 0) + 1
            label = "invisible" if "invisible" in labels else ("doubtful" if "doubtful" in labels else "visible")
            runs.append(Run(join_glyphs(cur), [round(v, 2) for v in rect], label, reasons,
                            any(g.remap for g in cur), tuple(sorted(methods))))
        cur.clear()

    for g in glyphs:
        if g.ch.isspace():
            if cur:
                cur.append(g)
            continue
        if want(g):
            cur.append(g)
        else:
            flush()
    flush()
    return runs
