"""§3.4 Human-view extraction: render pages, render with text removed, OCR."""
from __future__ import annotations

import fitz
import numpy as np

_OCR = None


def ocr_engine():
    global _OCR
    if _OCR is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _OCR = RapidOCR()
        except Exception:
            _OCR = False
    return _OCR or None


def pix_to_np(pix: fitz.Pixmap) -> np.ndarray:
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3].copy()


def render(page: fitz.Page, zoom: float, annots: bool) -> np.ndarray:
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, annots=annots,
                          colorspace=fitz.csRGB)
    return pix_to_np(pix)


def render_without_text(data: bytes, pno: int, zoom: float) -> np.ndarray:
    """Render the page with every text-showing operator removed (graphics/images untouched)."""
    d = fitz.open(stream=data, filetype="pdf")
    try:
        if d.needs_pass:
            d.authenticate("")
        p = d[pno]
        big = p.rect + (-1e4, -1e4, 1e4, 1e4)
        p.add_redact_annot(big, fill=False, cross_out=False)
        p.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
                           graphics=fitz.PDF_REDACT_LINE_ART_NONE,
                           text=fitz.PDF_REDACT_TEXT_REMOVE)
        return render(p, zoom, annots=False)
    finally:
        d.close()


def ocr(img: np.ndarray, zoom: float, origin=(0.0, 0.0)) -> list[dict]:
    """OCR an RGB array. Returns [{text, score, bbox(page coords)}]."""
    eng = ocr_engine()
    if eng is None:
        return []
    try:
        result, _ = eng(img)
    except Exception:
        return []
    out = []
    for box, text, score in result or []:
        xs = [pt[0] for pt in box]; ys = [pt[1] for pt in box]
        out.append({"text": text, "score": float(score),
                    "bbox": [min(xs) / zoom + origin[0], min(ys) / zoom + origin[1],
                             max(xs) / zoom + origin[0], max(ys) / zoom + origin[1]]})
    return out


def enhance_faint(img: np.ndarray) -> np.ndarray:
    """Amplify low-contrast detail relative to local background (for faint-text detection)."""
    from PIL import Image, ImageFilter
    im = Image.fromarray(img).convert("L")
    bg = np.asarray(im.filter(ImageFilter.MedianFilter(9)), dtype=np.int16)
    fg = np.asarray(im, dtype=np.int16)
    diff = np.clip(255 - np.abs(fg - bg) * 12, 0, 255).astype(np.uint8)
    return np.stack([diff] * 3, axis=-1)
