"""§3.10 Content disarm: safe-text export and flattened PDF."""
from __future__ import annotations

import fitz


def safe_text(result: dict) -> str:
    """Only human-perceptible text (glyphs both methods labelled visible), per page."""
    parts = []
    for p in result.get("pages", []):
        parts.append(f"--- page {p['page']} ---\n{p.get('safe_text', '')}")
    return "\n\n".join(parts) + "\n"


def flatten(src: str, dst: str, dpi: int = 150) -> None:
    """Re-render every page to an image-only PDF: drops metadata, annotations' hidden state,
    hidden layers, JavaScript, attachments, earlier revisions and all extractable text."""
    doc = fitz.open(src)
    if doc.needs_pass:
        doc.authenticate("")
    out = fitz.open()
    for page in doc:
        pix = page.get_pixmap(dpi=dpi, annots=True, alpha=False)
        np_ = out.new_page(width=page.rect.width, height=page.rect.height)
        np_.insert_image(np_.rect, stream=pix.tobytes("png"))
    out.set_metadata({})
    out.save(dst, garbage=4, deflate=True, no_new_id=True)
    out.close()
    doc.close()
