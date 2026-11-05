"""Leave-one-out ink attribution (the primary visibility decision).

For every text object pdfium extracts text from, re-render the page with only that object made
fully transparent and diff against the normal render. The pixels that change are exactly what the
object contributes to what a person sees, after everything the renderer applies: overdraw by later
content, clipping, soft masks, blend modes, opacity, layers, page boxes and blank or broken glyphs.
No font metrics, producer strings or font-specific rules are involved.

Objects whose regions don't overlap are measured in the same render (batches), so a typical page
costs a handful of renders. Characters are attributed to objects via FPDFText_GetTextObject.
"""
from __future__ import annotations

import ctypes
import unicodedata
from dataclasses import dataclass, field

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as R

from .util import wcag_contrast

T = {
    "noise": 10,                 # per-channel diff treated as rasterisation noise
    "contrast_invisible": 1.2,   # WCAG ratio of a char's strongest changed pixel
    "contrast_doubtful": 1.8,
    "ink_invisible_pt": 1.5,     # ink height of an object's letters, measured in rendered pixels
    "ink_small_pt": 3.0,
    "ink_small_rel": 0.45,       # ... and relative to the page's median line ink height
    "margin_pt": 1.0,
    "max_batches": 80,
}


@dataclass
class CharVis:
    ch: str
    bbox: tuple                  # MuPDF page coordinates
    origin: tuple
    obj: int                     # text object id (-1: not attributable)
    label: str | None = None     # visible / doubtful / invisible / None (not measured)
    reasons: list = field(default_factory=list)
    contrast: float = 0.0
    root: int = -1               # index of the char whose glyph also produced this one


def _addr(h) -> int:
    return ctypes.cast(h, ctypes.c_void_p).value or 0


def _text_objects(page_raw) -> list:
    """All text objects, including those nested in form XObjects, in deterministic order."""
    out = []

    def rec(count, get):
        for i in range(count):
            o = get(i)
            t = R.FPDFPageObj_GetType(o)
            if t == R.FPDF_PAGEOBJ_TEXT:
                out.append(o)
            elif t == R.FPDF_PAGEOBJ_FORM:
                rec(R.FPDFFormObj_CountObjects(o), lambda j, o=o: R.FPDFFormObj_GetObject(o, j))
    rec(R.FPDFPage_CountObjects(page_raw), lambda i: R.FPDFPage_GetObject(page_raw, i))
    return out


def _make_transparent(o) -> None:
    r, g, b, a = (ctypes.c_uint() for _ in range(4))
    refs = [ctypes.byref(x) for x in (r, g, b, a)]
    if R.FPDFPageObj_GetFillColor(o, *refs):
        R.FPDFPageObj_SetFillColor(o, r.value, g.value, b.value, 0)
    else:
        R.FPDFPageObj_SetFillColor(o, 0, 0, 0, 0)
    if R.FPDFPageObj_GetStrokeColor(o, *refs):
        R.FPDFPageObj_SetStrokeColor(o, r.value, g.value, b.value, 0)
    else:
        R.FPDFPageObj_SetStrokeColor(o, 0, 0, 0, 0)


def _render(page, zoom) -> np.ndarray:
    bm = page.render(scale=zoom, may_draw_forms=False, draw_annots=False)
    arr = bm.to_numpy()
    return arr[:, :, [2, 1, 0]].astype(np.int16)  # BGR(A) -> RGB


def _mapper(page_raw, w: int, h: int):
    """Affine map PDF user space -> device pixels, exactly as pdfium renders (handles /Rotate)."""
    k = 64  # oversample: FPDF_PageToDevice returns ints
    def dev(x, y):
        dx, dy = ctypes.c_int(), ctypes.c_int()
        R.FPDF_PageToDevice(page_raw, 0, 0, w * k, h * k, 0, x, y, ctypes.byref(dx), ctypes.byref(dy))
        return dx.value / k, dy.value / k
    o, ex, ey = dev(0, 0), dev(1000, 0), dev(0, 1000)
    a, b = (ex[0] - o[0]) / 1000, (ex[1] - o[1]) / 1000
    c, d = (ey[0] - o[0]) / 1000, (ey[1] - o[1]) / 1000
    return lambda x, y: (o[0] + a * x + c * y, o[1] + b * x + d * y)


def _box(tp, i, loose: bool, m):
    if loose:
        rf = R.FS_RECTF()
        if not R.FPDFText_GetLooseCharBox(tp, i, ctypes.byref(rf)):
            return None
        l, rt, b, t = rf.left, rf.right, rf.bottom, rf.top
    else:
        l, rt, b, t = (ctypes.c_double() for _ in range(4))
        if not R.FPDFText_GetCharBox(tp, i, *(ctypes.byref(v) for v in (l, rt, b, t))):
            return None
        l, rt, b, t = l.value, rt.value, b.value, t.value
    pts = [m(l, b), m(rt, t), m(l, t), m(rt, b)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _px(box, W, H, pad=0.0):
    x0, y0 = int(np.floor(box[0] - pad)), int(np.floor(box[1] - pad))
    x1, y1 = int(np.ceil(box[2] + pad)), int(np.ceil(box[3] + pad))
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, max(x1, x0 + 1)), min(H, max(y1, y0 + 1))
    return (x0, y0, x1, y1) if x0 < W and y0 < H and x1 > 0 and y1 > 0 else None


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def analyze_page(data: bytes, pno: int, zoom: float) -> tuple[list[CharVis], dict]:
    pdf = pdfium.PdfDocument(data)
    try:
        page = pdf[pno]
        tp = page.get_textpage()
        objs = _text_objects(page.raw)
        oid_of = {_addr(o): k for k, o in enumerate(objs)}
        base = _render(page, zoom)
        H, W = base.shape[:2]
        m = _mapper(page.raw, W, H)

        chars: list[CharVis] = []
        dev_boxes = []
        for i in range(tp.count_chars()):
            if R.FPDFText_IsGenerated(tp.raw, i) == 1:
                continue
            u = R.FPDFText_GetUnicode(tp.raw, i)
            ch = chr(u) if 0 < u < 0x110000 else "�"
            tight = _box(tp.raw, i, False, m)
            loose = _box(tp.raw, i, True, m)
            ox, oy = ctypes.c_double(), ctypes.c_double()
            R.FPDFText_GetCharOrigin(tp.raw, i, ctypes.byref(ox), ctypes.byref(oy))
            org = m(ox.value, oy.value)
            o = R.FPDFText_GetTextObject(tp.raw, i)
            cv = CharVis(ch, tuple(v / zoom for v in (tight or loose or (*org, *org))),
                         (org[0] / zoom, org[1] / zoom), oid_of.get(_addr(o), -1) if o else -1)
            if chars and cv.obj == chars[-1].obj and abs(cv.origin[0] - chars[-1].origin[0]) < 0.01 \
                    and abs(cv.origin[1] - chars[-1].origin[1]) < 0.01:
                cv.root = chars[-1].root if chars[-1].root >= 0 else len(chars) - 1
            chars.append(cv)
            dev_boxes.append((tight, loose))

        # region per object (device px)
        region: dict[int, tuple] = {}
        for cv, (tight, loose) in zip(chars, dev_boxes):
            b = tight if tight and (tight[2] - tight[0]) * (tight[3] - tight[1]) > 0.01 else loose
            if cv.obj < 0 or not b or cv.ch.isspace():
                continue
            r = region.get(cv.obj)
            region[cv.obj] = b if r is None else (min(r[0], b[0]), min(r[1], b[1]), max(r[2], b[2]), max(r[3], b[3]))
        pad = T["margin_pt"] * zoom
        region = {k: (r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad) for k, r in region.items()}

        batches: list[list[int]] = []
        for oid in sorted(region, key=lambda k: -(region[k][2] - region[k][0]) * (region[k][3] - region[k][1])):
            for bt in batches:
                if not any(_overlap(region[oid], region[o]) for o in bt):
                    bt.append(oid)
                    break
            else:
                batches.append([oid])
        measured = batches[: T["max_batches"]]

        removed_img: dict[int, np.ndarray] = {}
        for bt in measured:
            p2 = pdf[pno]
            try:
                objs2 = _text_objects(p2.raw)
                for oid in bt:
                    if oid < len(objs2):
                        _make_transparent(objs2[oid])
                img = _render(p2, zoom)
            finally:
                p2.close()
            for oid in bt:
                removed_img[oid] = img

        # per-object ink height over its letters/digits (for the size rule)
        ink_h: dict[int, float] = {}
        for oid, img in removed_img.items():
            rr = _px(region[oid], W, H)
            if not rr:
                continue
            x0, y0, x1, y1 = rr
            diff = np.abs(base[y0:y1, x0:x1] - img[y0:y1, x0:x1]).max(axis=2)
            rows = np.where((diff > T["noise"]).any(axis=1))[0]
            ink_h[oid] = (rows[-1] - rows[0] + 1) / zoom if len(rows) else 0.0
        alnum_objs = {cv.obj for cv in chars if cv.ch.isalnum()}
        alnum_count: dict[int, int] = {}
        for cv in chars:
            if cv.ch.isalnum():
                alnum_count[cv.obj] = alnum_count.get(cv.obj, 0) + 1
        sizes = [h for o, h in ink_h.items() if h > 0 and alnum_count.get(o, 0) >= 3]
        median_h = float(np.median(sizes)) if sizes else 0.0

        for idx, (cv, (tight, loose)) in enumerate(zip(chars, dev_boxes)):
            if cv.ch.isspace():
                continue
            if cv.root >= 0 or (unicodedata.combining(cv.ch) and idx > 0):
                continue  # inherits below
            img = removed_img.get(cv.obj)
            if img is None:
                continue  # not attributable / over batch limit -> caller falls back
            b = tight if tight and (tight[2] - tight[0]) * (tight[3] - tight[1]) > 0.01 else loose
            rr = _px(b, W, H) if b else None
            if rr is None:
                cv.label, cv.reasons, cv.contrast = "invisible", ["no_rendered_contribution", "outside_rendered_page"], 1.0
                continue
            x0, y0, x1, y1 = rr
            a, r = base[y0:y1, x0:x1], img[y0:y1, x0:x1]
            diff = np.abs(a - r).max(axis=2)
            if diff.max() <= T["noise"]:
                cv.label, cv.reasons, cv.contrast = "invisible", ["no_rendered_contribution"], 1.0
                continue
            yy, xx = np.unravel_index(np.argmax(diff), diff.shape)
            c = wcag_contrast(a[yy, xx] / 255.0, r[yy, xx] / 255.0)
            cv.contrast = round(c, 3)
            if c < T["contrast_invisible"]:
                cv.label, cv.reasons = "invisible", ["imperceptible_contrast"]
            elif c < T["contrast_doubtful"]:
                cv.label, cv.reasons = "doubtful", ["low_contrast"]
            else:
                cv.label = "visible"
            h = ink_h.get(cv.obj, 0.0)
            if cv.obj in alnum_objs and alnum_count.get(cv.obj, 0) >= 2 and cv.label != "invisible":
                if h < T["ink_invisible_pt"]:
                    cv.label = "invisible"; cv.reasons.append("illegibly_small")
                elif h < T["ink_small_pt"] and median_h and h < T["ink_small_rel"] * median_h:
                    cv.label = "doubtful"; cv.reasons.append("very_small_relative_to_body_text")

        # glyph-sharing chars (ligatures, multi-char ToUnicode) and combining marks inherit
        multi_runs = []
        for idx, cv in enumerate(chars):
            src = cv.root if cv.root >= 0 else (idx - 1 if unicodedata.combining(cv.ch) and idx > 0 else -1)
            if src >= 0 and cv.label is None:
                cv.label, cv.reasons, cv.contrast = chars[src].label, list(chars[src].reasons), chars[src].contrast
        i = 0
        while i < len(chars):
            j = i + 1
            while j < len(chars) and chars[j].root == i:
                j += 1
            if j - i > 3:  # one glyph extracting as >3 characters: beyond any ligature
                multi_runs.append(("".join(c.ch for c in chars[i:j]), chars[i].bbox))
            i = j

        info = {"text_objects": len(objs), "batches": len(batches), "measured_batches": len(measured),
                "unmeasured_chars": sum(1 for c in chars if c.label is None and not c.ch.isspace()),
                "median_line_ink_pt": round(median_h, 2), "multi_char_glyphs": multi_runs}
        return chars, info
    finally:
        pdf.close()
