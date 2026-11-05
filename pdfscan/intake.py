"""§3.1 Intake: hashing, resource limits, structural validation, encryption handling."""
from __future__ import annotations

import io
import os
import warnings

import fitz

from .util import Finding, sha256_file

LIMITS = {
    "max_bytes": 200 * 1024 * 1024,
    "max_pages": 2000,
    "max_objects": 500_000,
    "max_decompressed_bytes": 1024 * 1024 * 1024,
    "max_decompress_ratio": 200.0,
    "heavy_repair_warnings": 5,
}


class Quarantine(Exception):
    """Raised when the file must be quarantined rather than scanned (never 'clean')."""


def intake(path: str, password: str = "") -> tuple[fitz.Document, dict, list[Finding]]:
    findings: list[Finding] = []
    size = os.path.getsize(path)
    info = {"path": os.path.abspath(path), "sha256": sha256_file(path), "bytes": size}
    if size > LIMITS["max_bytes"]:
        raise Quarantine(f"file too large ({size} bytes)")

    with open(path, "rb") as f:
        data = f.read()
    head = data[:1024]
    hdr = head.find(b"%PDF-")
    if hdr < 0:
        raise Quarantine("no %PDF- header in first 1024 bytes")
    if hdr > 0:
        findings.append(Finding("structural", ["data_before_header"], "medium",
                                text=data[:hdr].decode("latin-1", "replace"),
                                source="bytes before %PDF-", detail={"offset": hdr}))

    fitz.TOOLS.mupdf_warnings()  # reset
    fitz.TOOLS.mupdf_display_errors(False)
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:  # parser failure
        raise Quarantine(f"MuPDF failed to open file: {e}")

    if doc.needs_pass:
        if not doc.authenticate(password):
            raise Quarantine("encrypted and the empty/provided password did not work")
        info["encrypted"] = True
        findings.append(Finding("structural", ["encrypted_opened_with_empty_password"], "low",
                                source="encryption"))

    info["pages"] = doc.page_count
    info["objects"] = doc.xref_length()
    info["metadata_producer"] = (doc.metadata or {}).get("producer", "")
    info["metadata_creator"] = (doc.metadata or {}).get("creator", "")
    if doc.page_count > LIMITS["max_pages"]:
        raise Quarantine(f"too many pages ({doc.page_count})")
    if doc.xref_length() > LIMITS["max_objects"]:
        raise Quarantine(f"too many objects ({doc.xref_length()})")

    # Decompression bomb check: total decoded stream size and ratio.
    total = 0
    for x in range(1, doc.xref_length()):
        try:
            if doc.xref_is_stream(x):
                total += len(doc.xref_stream(x) or b"")
        except Exception:
            pass
        if total > LIMITS["max_decompressed_bytes"]:
            raise Quarantine("decompressed stream data exceeds limit")
    info["decompressed_bytes"] = total
    if size and total / size > LIMITS["max_decompress_ratio"]:
        raise Quarantine(f"decompression ratio {total / size:.0f}x exceeds limit")

    # Repair / recovery accounting (MuPDF + pypdf).
    mupdf_warn = [w for w in fitz.TOOLS.mupdf_warnings().splitlines() if w.strip()]
    pypdf_warn: list[str] = []
    try:
        import pypdf
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import logging
            logger = logging.getLogger("pypdf")
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            logger.addHandler(handler)
            try:
                r = pypdf.PdfReader(io.BytesIO(data), strict=False)
                _ = len(r.pages)
            finally:
                logger.removeHandler(handler)
            pypdf_warn = [str(c.message) for c in caught] + [l for l in stream.getvalue().splitlines() if l]
    except Exception as e:
        pypdf_warn.append(f"pypdf failed: {e}")
    info["repaired"] = bool(getattr(doc, "is_repaired", False))
    info["repair_warnings"] = {"mupdf": mupdf_warn[:50], "pypdf": pypdf_warn[:50]}
    nwarn = len(mupdf_warn) + len(pypdf_warn)
    if info["repaired"] or nwarn:
        heavy = info["repaired"] or nwarn >= LIMITS["heavy_repair_warnings"]
        findings.append(Finding("structural", ["parser_repair"], "medium" if heavy else "low",
                                source="intake", review=heavy,
                                detail={"repaired": info["repaired"], "warnings": nwarn,
                                        "heavy": heavy}))
    return doc, info, findings
