"""§3.2 Structural analyzer + §1C non-page content enumeration."""
from __future__ import annotations

import re
from collections import deque

import fitz

from .util import SEVERITY_ORDER, Finding, decode_pdf_string, parse_hex_at, parse_literal_at, norm_words, is_substantive

REF_RE = re.compile(r"(\d+)\s+0\s+R\b")
TEXT_SHOW_RE = re.compile(rb"(\((?:\\.|[^\\)])*\)|<[0-9A-Fa-f\s]*>)\s*(Tj|'|\")|\[(.*?)\]\s*TJ", re.S)
ACTION_KEYS = {
    "JS": "javascript", "JavaScript": "javascript", "Launch": "launch_action",
    "SubmitForm": "submit_form_action", "ImportData": "import_data_action",
    "GoToR": "remote_goto_action", "GoToE": "embedded_goto_action",
    "RichMedia": "rich_media", "URI": "uri_action",
}
STRING_KEYS = ["Alt", "ActualText", "E", "TU", "Lang"]


def analyze(doc: fitz.Document, data: bytes) -> tuple[list[Finding], list[tuple[str, bytes]]]:
    findings: list[Finding] = []
    embedded: list[tuple[str, bytes]] = []
    findings += _revisions(doc, data)
    findings += _orphans(doc)
    findings += _actions(doc)
    findings += _metadata(doc)
    findings += _annotations_and_fields(doc)
    findings += _navigation(doc)
    findings += _object_strings(doc)
    findings += _layers(doc)
    findings += _content_stream_markers(doc)
    f, embedded = _attachments(doc)
    findings += f
    return findings, embedded


# ---------------------------------------------------------------- revisions / EOF
def _revisions(doc, data: bytes) -> list[Finding]:
    out = []
    eofs = [m.end() for m in re.finditer(rb"%%EOF", data)]
    xref_tables = len(re.findall(rb"(?m)^\s*xref\s*$", data))
    startxrefs = len(re.findall(rb"startxref", data))
    if eofs:
        tail = data[eofs[-1]:]
        if tail.strip(b"\r\n\t \x00"):
            out.append(Finding("structural", ["data_after_eof"], "medium",
                               text=tail.decode("latin-1", "replace")[:5000], source="after final %%EOF",
                               detail={"bytes": len(tail)}))
    else:
        out.append(Finding("structural", ["missing_eof"], "low", source="trailer"))
    if len(eofs) > 1 or startxrefs > 1:
        out.append(Finding("structural", ["incremental_updates"], "low", source="revisions",
                           detail={"eof_markers": len(eofs), "startxref": startxrefs,
                                   "xref_tables": xref_tables}))
    # Diff text of earlier revisions against the final one (shadow attacks).
    final_words = _doc_words(doc)
    for i, end in enumerate(eofs[:-1]):
        try:
            # include trailing EOL after %%EOF
            rev = fitz.open(stream=data[:end + 2], filetype="pdf")
        except Exception:
            continue
        try:
            for pno in range(rev.page_count):
                txt = rev[pno].get_text()
                removed = [w for w in norm_words(txt, 3) if w not in final_words]
                if removed:
                    out.append(Finding("structural", ["earlier_revision_text_differs"], "medium",
                                       page=pno + 1, text=txt[:5000], source=f"revision {i + 1}",
                                       review=True,
                                       detail={"words_not_in_final": removed[:40]}))
        finally:
            rev.close()
    return out


def _doc_words(doc) -> set[str]:
    s = set()
    for p in doc:
        s.update(norm_words(p.get_text(), 3))
    return s


# ---------------------------------------------------------------- orphans
def _orphans(doc) -> list[Finding]:
    n = doc.xref_length()
    seen, q = set(), deque()
    trailer = doc.pdf_trailer() or ""
    for m in REF_RE.finditer(trailer):
        q.append(int(m.group(1)))
    while q:
        x = q.popleft()
        if x in seen or x <= 0 or x >= n:
            continue
        seen.add(x)
        try:
            obj = doc.xref_object(x, compressed=False)
        except Exception:
            continue
        for m in REF_RE.finditer(obj):
            q.append(int(m.group(1)))
    out = []
    for x in range(1, n):
        if x in seen:
            continue
        try:
            if not doc.xref_is_stream(x):
                continue
            obj = doc.xref_object(x)
            if re.search(r"/Type\s*/(ObjStm|XRef|Metadata)", obj):
                continue
            raw = doc.xref_stream(x) or b""
        except Exception:
            continue
        text = _content_stream_strings(raw)
        if text and is_substantive(text):
            out.append(Finding("structural", ["orphaned_content_stream"], "medium",
                               text=text[:5000], source=f"unreferenced object {x}",
                               detail={"xref": x}))
    return out


def _content_stream_strings(raw: bytes) -> str:
    """Best-effort: literal strings drawn by Tj/TJ in a raw content stream (font encoding ignored)."""
    parts = []
    for m in TEXT_SHOW_RE.finditer(raw):
        chunk = m.group(1) if m.group(1) else m.group(3)
        i = 0
        while i < len(chunk):
            c = chunk[i:i + 1]
            if c == b"(":
                s, i = parse_literal_at(chunk, i); parts.append(s.decode("latin-1"))
            elif c == b"<":
                s, i = parse_hex_at(chunk, i); parts.append(s.decode("latin-1"))
            else:
                # TJ kerning: a large negative adjustment (thousandths of an em) is a word gap
                m = re.match(rb"-?\d*\.?\d+", chunk[i:])
                if m:
                    if float(m.group()) < -200 and parts and not parts[-1].endswith(" "):
                        parts.append(" ")
                    i += len(m.group())
                else:
                    i += 1
        parts.append(" ")
    return re.sub(r" {2,}", " ", "".join(parts)).strip()


# ---------------------------------------------------------------- actions / JS
def _key_text(doc, x: int, key: str) -> str:
    """String value of key on object x, also looking inside an inline action dict (/A << ... >>)."""
    for path in (key, f"A/{key}"):
        try:
            t, v = doc.xref_get_key(x, path)
        except Exception:
            continue
        if t == "string":
            return v
        if t == "xref":
            try:
                return (doc.xref_stream(int(v.split()[0])) or b"").decode("latin-1")
            except Exception:
                pass
    return ""


def _actions(doc) -> list[Finding]:
    out, chain_xrefs = _open_action(doc)
    for x in range(1, doc.xref_length()):
        if x in chain_xrefs:
            continue  # already explained as part of the /OpenAction
        try:
            obj = doc.xref_object(x)
        except Exception:
            continue
        is_link = re.search(r"/Subtype\s*/Link\b", obj) is not None
        for key, mech in ACTION_KEYS.items():
            if not re.search(rf"/{key}\b", obj) and not re.search(rf"/S\s*/{key}\b", obj):
                continue
            if key == "JavaScript" and "/JS" in obj:
                continue  # reported via /JS
            if key == "URI" and is_link:
                continue  # clickable links are reported once per target by _links()
            text = _key_text(doc, x, key) if key in ("JS", "URI") else ""
            sev = {"javascript": "high", "launch_action": "high", "uri_action": "info"}.get(mech, "medium")
            out.append(Finding("action", [mech], sev, text=text[:5000],
                               source=f"object {x}", detail={"xref": x}))
    if doc.xref_get_key(doc.pdf_catalog(), "AA")[0] != "null":
        out.append(Finding("action", ["document_additional_actions"], "medium", source="catalog /AA"))
    return out


# What an /OpenAction of each type does, and how much it matters
ACTION_SEVERITY = {
    "GoTo": "info", "Thread": "info", "Trans": "info", "GoTo3DView": "info",
    "Named": "low", "ResetForm": "low", "Hide": "low", "SetOCGState": "low", "Sound": "low", "Movie": "low",
    "URI": "medium", "SubmitForm": "medium", "ImportData": "medium", "GoToR": "medium", "GoToE": "medium",
    "Rendition": "medium", "JavaScript": "high", "Launch": "high", "RichMediaExecute": "high",
}
VIEWS = {"Fit": "fit to window", "FitH": "fit width", "FitV": "fit height", "FitB": "fit content",
         "FitBH": "fit content width", "FitBV": "fit content height", "FitR": "fit a rectangle",
         "XYZ": "at a set position and zoom"}


def _inline_get(s: str, k: str) -> tuple[str, str]:
    """Minimal key lookup inside an inline dictionary string (same shape as xref_get_key)."""
    m = re.search(rf"/{k}\s*(/[^\s/<>\[\]()]+|\(((?:\\.|[^\\)])*)\)|(\d+)\s+0\s+R|\[[^\]]*\]|<<.*?>>)", s, re.S)
    if not m:
        return "null", "null"
    g = m.group(1)
    if g.startswith("/"):
        return "name", g
    if g.startswith("("):
        return "string", m.group(2)
    if g.startswith("["):
        return "array", g
    if g.startswith("<<"):
        return "dict", g
    return "xref", f"{m.group(3)} 0 R"


def _describe_dest(doc, t: str, v: str) -> str:
    if t in ("string", "name"):
        return f"named destination '{v.lstrip('/')[:60]}'"
    m = re.search(r"(\d+)\s+0\s+R\s*/(\w+)", v)
    if m:
        page = next((p.number + 1 for p in doc if p.xref == int(m.group(1))), None)
        view = VIEWS.get(m.group(2), m.group(2))
        return f"page {page}, {view}" if page else f"object {m.group(1)}, {view}"
    m = re.search(r"\[\s*(\d+)\s*/(\w+)", v)
    if m:
        return f"page index {int(m.group(1)) + 1}, {VIEWS.get(m.group(2), m.group(2))}"
    return v[:60]


def _filespec(get) -> str:
    t, v = get("F")
    if t == "string":
        return v
    if t == "dict":
        m = re.search(r"/(?:UF|F)\s*\(((?:\\.|[^\\)])*)\)", v)
        return m.group(1) if m else v[:80]
    return v if t != "null" else "?"


def _describe_action(doc, typ: str, get) -> tuple[str, str]:
    """(plain-language description, attacker-relevant text carried by the action)."""
    if typ == "GoTo":
        t, v = get("D")
        return f"go to {_describe_dest(doc, t, v)}", ""
    if typ == "URI":
        t, v = get("URI")
        return f"open website {v}", v
    if typ == "JavaScript":
        t, v = get("JS")
        code = v if t == "string" else ""
        if t == "xref":
            try:
                code = (doc.xref_stream(int(v.split()[0])) or b"").decode("latin-1")
            except Exception:
                pass
        return f"run JavaScript ({len(code)} characters)", code
    if typ == "Launch":
        f = _filespec(get)
        return f"launch {f}", f
    if typ in ("SubmitForm", "ImportData"):
        f = _filespec(get)
        return ("submit form data to " if typ == "SubmitForm" else "import data from ") + f, f
    if typ in ("GoToR", "GoToE"):
        f = _filespec(get)
        t, v = get("D")
        return f"open another document {f} at {_describe_dest(doc, t, v)}", f
    if typ == "Named":
        t, v = get("N")
        return f"run viewer command {v.lstrip('/')}", ""
    return f"{typ} action", ""


def _open_action(doc) -> tuple[list[Finding], set[int]]:
    """Resolve the catalog /OpenAction (and any /Next chain) into what it actually does."""
    cat = doc.pdf_catalog()
    t, v = doc.xref_get_key(cat, "OpenAction")
    if t == "null":
        return [], set()
    if t in ("array", "string", "name"):  # a plain destination, not an action
        step = f"open at {_describe_dest(doc, t, v)}"
        return [Finding("action", ["open_action", "open_action_goto"], "info", source="catalog /OpenAction",
                        detail={"steps": [step]})], set()
    steps, texts, mechs, xrefs = [], [], ["open_action"], set()
    sev = "info"
    cur = (t, v)
    for _ in range(10):
        t, v = cur
        if t == "xref":
            x = int(v.split()[0])
            if x in xrefs:
                steps.append("loops back to an earlier action")
                mechs.append("action_loop")
                break
            xrefs.add(x)
            get = (lambda k, x=x: doc.xref_get_key(x, k))
        elif t == "dict":
            get = (lambda k, s=v: _inline_get(s, k))
        else:
            break
        st, sv = get("S")
        typ = sv.lstrip("/") if st == "name" else "unknown"
        desc, text = _describe_action(doc, typ, get)
        steps.append(desc)
        if text:
            texts.append(text)
        if f"open_action_{typ.lower()}" not in mechs:
            mechs.append(f"open_action_{typ.lower()}")
        sev = max(sev, ACTION_SEVERITY.get(typ, "medium"), key=lambda s: SEVERITY_ORDER[s])
        nt, nv = get("Next")
        if nt == "array":
            refs = REF_RE.findall(nv)
            nt, nv = ("xref", f"{refs[0]} 0 R") if refs else ("null", "null")
            if len(refs) > 1:
                steps.append(f"(+{len(refs) - 1} further chained actions not followed)")
        if nt not in ("xref", "dict"):
            break
        if "action_chain" not in mechs:
            mechs.append("action_chain")
        if sev == "info":
            sev = "low"  # chains are unusual even when each step is harmless
        cur = (nt, nv)
    return [Finding("action", mechs, sev, source="catalog /OpenAction", text="\n".join(texts)[:5000],
                    detail={"steps": steps})], xrefs


# ---------------------------------------------------------------- metadata
# Fields written by authoring tools themselves (no user-supplied content)
TOOL_META_KEYS = {"Producer", "Creator", "CreationDate", "ModDate", "Trapped",
                  "producer", "creator", "creationDate", "modDate", "trapped", "format"}
DESCRIPTIVE_META_KEYS = {"Title", "Author", "Subject", "Keywords", "title", "author", "subject", "keywords"}


def _info_fields(doc) -> dict[str, str]:
    """Every entry of the document Info dictionary, including non-standard keys."""
    fields: dict[str, str] = {}
    try:
        t, v = doc.xref_get_key(-1, "Info")
        if t == "xref":
            ix = int(v.split()[0])
            for k in doc.xref_get_keys(ix):
                kt, kv = doc.xref_get_key(ix, k)
                if kt in ("string", "name") and kv.strip():
                    fields[k] = kv.lstrip("/") if kt == "name" else kv
    except Exception:
        pass
    if not fields:
        fields = {k: v for k, v in (doc.metadata or {}).items() if v and k not in ("encryption",)}
    return fields


def _metadata(doc) -> list[Finding]:
    out = []
    fields = _info_fields(doc)
    tool = {k: v for k, v in fields.items() if k in TOOL_META_KEYS}
    if tool:
        # one entry for all tool/date fields; semantic + character checks still read the values
        out.append(Finding("offpage_content", ["metadata_info"], "info",
                           text="\n".join(f"{k}: {v}" for k, v in tool.items()),
                           source="info: tool and date fields", benign="standard_tool_and_date_fields",
                           detail={"fields": sorted(tool)}))
    for k, v in fields.items():
        if k in TOOL_META_KEYS or not is_substantive(v):
            continue
        mech = "metadata_info" if k in DESCRIPTIVE_META_KEYS else "metadata_custom_field"
        out.append(Finding("offpage_content", [mech], "info", text=v, source=f"info:{k}"))
    props = xmp_properties(doc.get_xml_metadata() or "")
    if props:
        # checked in full (semantic analysis chunks long text); nothing is truncated here
        text = "\n".join(f"{k}: {v}" for k, v in props if is_substantive(v))
        if text:
            out.append(Finding("offpage_content", ["metadata_xmp"], "info", text=text,
                               source="XMP", detail={"properties": len(props)}))
    return out


_XML_ENT = {"&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'", "&amp;": "&"}


def _xml_unescape(s: str) -> str:
    s = re.sub(r"&#x([0-9a-fA-F]+);", lambda m: chr(int(m.group(1), 16)) if int(m.group(1), 16) < 0x110000 else "", s)
    s = re.sub(r"&#(\d+);", lambda m: chr(int(m.group(1))) if int(m.group(1)) < 0x110000 else "", s)
    for k, v in _XML_ENT.items():
        s = s.replace(k, v)
    return s


def xmp_properties(xmp: str) -> list[tuple[str, str]]:
    """XMP as (property, value) pairs, e.g. ('dc:title', 'Report'). Regex-based on purpose: the
    XML is attacker-controlled, so no XML parser (entity expansion, parser bugs) is involved.
    Covers both property encodings: attributes on rdf:Description and child elements (including
    rdf:Seq/Bag/Alt lists, whose rdf:li items are joined)."""
    if not xmp:
        return []
    xmp = re.sub(r"<!--.*?-->", "", xmp, flags=re.S)
    props: list[tuple[str, str]] = []
    for desc in re.finditer(r"<rdf:Description\b([^>]*)>", xmp):
        for m in re.finditer(r"([\w.\-]+:[\w.\-]+)\s*=\s*(\"([^\"]*)\"|'([^']*)')", desc.group(1)):
            name = m.group(1)
            if name.startswith(("xmlns:", "rdf:", "x:")):
                continue
            val = _xml_unescape(m.group(3) if m.group(3) is not None else m.group(4) or "")
            if val.strip():
                props.append((name, val))
    # wrappers (x:xmpmeta, rdf:RDF, rdf:Description, ...) are excluded in the pattern itself so a
    # match on them can't swallow the properties nested inside
    for m in re.finditer(r"<(?!rdf:|x:)([\w.\-]+:[\w.\-]+)\b[^>]*?(?<!/)>(.*?)</\1\s*>", xmp, re.S):
        name, inner = m.group(1), m.group(2)
        items = re.findall(r"<rdf:li\b[^>]*>(.*?)</rdf:li\s*>", inner, re.S)
        parts = items if items else ([inner] if "<" not in inner else [])
        vals = [_xml_unescape(re.sub(r"<[^>]+>", "", p)).strip() for p in parts]
        val = " | ".join(v for v in vals if v)
        if val:
            props.append((name, val))
    return props


def metadata_view(doc) -> dict:
    """Every metadata value, complete, for human review (the report shows these in full)."""
    xmp_raw = doc.get_xml_metadata() or ""
    return {"info": [[k, v] for k, v in _info_fields(doc).items()],
            "xmp": [[k, v] for k, v in xmp_properties(xmp_raw)],
            "xmp_bytes": len(xmp_raw.encode("utf-8", "replace"))}


# ---------------------------------------------------------------- annotations / forms
def _annotations_and_fields(doc) -> list[Finding]:
    out = []
    for page in doc:
        for a in page.annots() or []:
            flags = a.flags
            hidden = bool(flags & (fitz.PDF_ANNOT_IS_HIDDEN | fitz.PDF_ANNOT_IS_NO_VIEW))
            info = a.info or {}
            parts = [info.get(k, "") for k in ("content", "title", "subject")]
            try:
                rc = doc.xref_get_key(a.xref, "RC")
                if rc[0] == "string":
                    parts.append(re.sub(r"<[^>]+>", " ", rc[1]))
            except Exception:
                pass
            text = "\n".join(p for p in parts if p)
            if not is_substantive(text):
                continue
            mech = ["annotation_" + a.type[1].lower()]
            if hidden:
                mech.append("annotation_hidden_flag")
            r = a.rect
            if not r.intersects(page.rect):
                mech.append("annotation_off_page")
            out.append(Finding("offpage_content", mech, "medium" if len(mech) > 1 else "low",
                               page=page.number + 1, bbox=list(r), text=text,
                               source=f"annotation {a.type[1]}"))
        for w in page.widgets() or []:
            vals = {"value": w.field_value, "label": w.field_label, "name": w.field_name}
            try:
                dv = doc.xref_get_key(w.xref, "DV")
                if dv[0] == "string":
                    vals["default"] = dv[1]
            except Exception:
                pass
            text = "\n".join(f"{v}" for v in vals.values() if isinstance(v, str) and v)
            if is_substantive(text):
                out.append(Finding("offpage_content", ["form_field"], "low", page=page.number + 1,
                                   bbox=list(w.rect), text=text, source=f"widget {w.field_name}"))
    return out + _links(doc)


SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")
EMAIL_RE = re.compile(r"^[^@\s/]+@[^@\s/]+\.[A-Za-z]{2,}$")


def _links(doc) -> list[Finding]:
    """One entry per distinct link target (URL, email, file), listing every page it appears on."""
    targets: dict[str, list] = {}
    for page in doc:
        for ln in page.get_links():
            target = ln.get("uri") or ""
            if not target and ln.get("xref"):
                # MuPDF classifies scheme-less URIs (e.g. a bare e-mail address) as file launches
                # and doesn't expose them as "uri"; read the action's /URI directly
                target = _key_text(doc, ln["xref"], "URI")
            if not target:
                continue
            targets.setdefault(target, []).append((page.number + 1, list(ln["from"])))
    out = []
    for target, where in targets.items():
        mech = ["link_uri"]
        note = None
        if not SCHEME_RE.match(target):
            mech.append("uri_without_scheme")
            note = ("an e-mail address without 'mailto:'" if EMAIL_RE.match(target) else
                    "no scheme: some viewers resolve it as a relative file path")
        pages = sorted({p for p, _ in where})
        detail = {"pages": pages, "occurrences": len(where)}
        if note:
            detail["note"] = note
        out.append(Finding("action", mech, "info", page=pages[0],
                           bbox=where[0][1] if len(where) == 1 else None,
                           text=target, source="link", detail=detail))
    return out


# ---------------------------------------------------------------- navigation
def _navigation(doc) -> list[Finding]:
    out = []
    toc = doc.get_toc(simple=True)
    if toc:
        text = "\n".join(t[1] for t in toc)
        if is_substantive(text):
            out.append(Finding("offpage_content", ["outline_titles"], "info", text=text,
                               source="outline/bookmarks"))
    labels = []
    for p in doc:
        try:
            lab = p.get_label()
        except Exception:
            lab = ""
        if lab and not lab.strip().isdigit():
            labels.append(lab)
    if labels:
        out.append(Finding("offpage_content", ["page_labels"], "info", text="\n".join(labels),
                           source="page labels"))
    return out


def _object_strings(doc) -> list[Finding]:
    """Alt text, ActualText, expansions, tooltips stored in object dictionaries."""
    buckets: dict[str, list[str]] = {}
    for x in range(1, doc.xref_length()):
        for k in STRING_KEYS:
            try:
                t, v = doc.xref_get_key(x, k)
            except Exception:
                continue
            if t == "string" and v and is_substantive(v):
                buckets.setdefault(k, []).append(v)
    out = []
    for k, vals in buckets.items():
        if k == "Lang":
            continue
        mech = {"Alt": "alt_text", "ActualText": "actual_text_dict", "E": "abbreviation_expansion",
                "TU": "field_tooltip"}[k]
        out.append(Finding("offpage_content", [mech], "low", text="\n".join(vals)[:10000],
                           source=f"/{k} strings", detail={"count": len(vals)}))
    return out


# ---------------------------------------------------------------- optional content
def _layers(doc) -> list[Finding]:
    out = []
    try:
        ocgs = doc.get_ocgs() or {}
    except Exception:
        ocgs = {}
    for xref, o in ocgs.items():
        obj = doc.xref_object(xref)
        mech = []
        if not o.get("on", True):
            mech.append("layer_off_by_default")
        if re.search(r"/PrintState\s*/ON", obj) and re.search(r"/ViewState\s*/OFF", obj):
            mech.append("layer_print_only")
        if re.search(r"/ViewState\s*/ON", obj) and re.search(r"/PrintState\s*/OFF", obj):
            mech.append("layer_screen_only")
        if mech:
            out.append(Finding("structural", mech, "low", text=o.get("name", ""),
                               source=f"OCG {xref}", detail={"xref": xref, "name": o.get("name")}))
    return out


# ---------------------------------------------------------------- content-stream markers
def _content_stream_markers(doc) -> list[Finding]:
    """Raw scans that the glyph tracer cannot see: clip-mode text (7 Tr) and marked-content ActualText."""
    out = []
    for page in doc:
        xrefs = list(page.get_contents())
        for img in page.get_xobjects():
            xrefs.append(img[0])
        for x in xrefs:
            try:
                raw = doc.xref_stream(x) or b""
            except Exception:
                continue
            if re.search(rb"(?<![\d.])7\s+Tr\b", raw):
                out.append(Finding("hidden_text", ["clip_only_render_mode"], "medium",
                                   page=page.number + 1, source=f"content stream {x}",
                                   text=_content_stream_strings(_slice_after(raw, rb"7\s+Tr")),
                                   detail={"xref": x}, review=True))
            for m in re.finditer(rb"/ActualText\s*", raw):
                j = m.end()
                if raw[j:j + 1] == b"(":
                    s, _ = parse_literal_at(raw, j)
                elif raw[j:j + 1] == b"<":
                    s, _ = parse_hex_at(raw, j)
                else:
                    continue
                txt = decode_pdf_string(s)
                if is_substantive(txt):
                    out.append(Finding("mismatched_text", ["actual_text_override"], "low",
                                       page=page.number + 1, text=txt,
                                       source=f"marked content in stream {x}",
                                       detail={"needs_visual_check": True}))
    return out


def _slice_after(raw: bytes, pat: bytes) -> bytes:
    m = re.search(pat, raw)
    if not m:
        return b""
    end = raw.find(b"ET", m.end())
    return raw[m.end(): end if end > 0 else m.end() + 2000]


# ---------------------------------------------------------------- attachments
def _attachments(doc) -> tuple[list[Finding], list[tuple[str, bytes]]]:
    out, emb = [], []
    try:
        names = doc.embfile_names()
    except Exception:
        names = []
    for name in names:
        try:
            data = doc.embfile_get(name)
        except Exception:
            data = b""
        out.append(Finding("attachment", ["embedded_file"], "medium", text=name,
                           source=f"EmbeddedFiles:{name}", detail={"bytes": len(data)}))
        emb.append((name, data))
    for page in doc:
        for a in page.annots(types=[fitz.PDF_ANNOT_FILE_ATTACHMENT]) or []:
            try:
                data = a.get_file()
                fname = a.file_info.get("filename", "attachment")
            except Exception:
                continue
            out.append(Finding("attachment", ["file_attachment_annotation"], "medium",
                               page=page.number + 1, text=fname, source="FileAttachment annot",
                               detail={"bytes": len(data)}))
            emb.append((fname, data))
    if re.search(r"/Collection\b", doc.xref_object(doc.pdf_catalog())):
        out.append(Finding("attachment", ["pdf_portfolio"], "medium", source="catalog /Collection"))
    return out, emb
