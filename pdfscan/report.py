"""§3.10 Report rendering: structured JSON (default, for agents) and a human-readable report.

Attacker-controlled text only ever appears defanged (URLs broken, invisible characters shown as
<U+XXXX>) unless --include-raw is given, in which case raw text is added in separate fields.
"""
from __future__ import annotations

import os
import re
import textwrap

from . import charlevel, semantic
from . import labels as L
from .util import SEVERITY_ORDER, defang

JSON_TEXT_LIMIT = 4000
HUMAN_TEXT_LIMIT = 1500
INTERNAL_DETAIL = {"reason_counts", "needs_visual_check"}


def _sort_key(f: dict):
    sem = f.get("semantic") or {}
    return (-int(bool(sem.get("instruction_like"))), -SEVERITY_ORDER[f["severity"]],
            int(bool(f.get("benign"))), f.get("page") or 0)


def _round(v):
    if isinstance(v, float):
        return round(v, 2)
    if isinstance(v, (list, tuple)):
        return [_round(x) for x in v]
    if isinstance(v, dict):
        return {k: _round(x) for k, x in v.items()}
    return v


META_MECHS = {"metadata_info", "metadata_xmp", "metadata_custom_field"}
META_VALUE_LIMIT = 20000
_B64ISH = re.compile(r"^[A-Za-z0-9+/=\s]+$")


def meta_flags(value: str) -> list[str]:
    """What the automated checks see in one metadata value (pattern-based, so bypassable:
    the report shows the full value for a person to read regardless)."""
    flags = []
    issues, decoded = charlevel.scan(value)
    for iss in issues:
        s = L.mech(iss["mechanism"])
        if iss.get("decoded"):
            s += f" (decodes to: \"{defang(iss['decoded'], 120)}\")"
        elif iss.get("count"):
            s += f" ({iss['count']})"
        flags.append(s)
    for text in [value] + decoded:
        r = semantic.classify_long(text)
        if r["score"] >= semantic.THRESHOLDS["offpage"]:
            flags.append("reads like instructions to an AI: "
                         + ", ".join(L.SEMANTIC.get(x, x) for x in r["all_labels"]))
            break
        if r["all_labels"]:
            flags.append("weak instruction signals: " + ", ".join(L.SEMANTIC.get(x, x) for x in r["all_labels"]))
            break
    return flags


def metadata_json(result: dict) -> dict | None:
    md = result.get("metadata")
    if not md:
        return None

    def entries(pairs):
        out = []
        for k, v in pairs:
            e = {"name": k, "value": defang(v, META_VALUE_LIMIT), "length": len(v)}
            fl = meta_flags(v)
            if fl:
                e["flags"] = fl
            out.append(e)
        return out
    return {"info": entries(md.get("info", [])), "xmp": entries(md.get("xmp", [])),
            "xmp_bytes": md.get("xmp_bytes", 0)}


def is_inventory(f: dict) -> bool:
    """Info-level items that describe the document (links, metadata, bookmarks, a harmless
    OpenAction) rather than a problem. Kept out of 'findings' unless they read like instructions
    or need review."""
    return (f["severity"] == "info" and not (f.get("semantic") or {}).get("instruction_like")
            and not f.get("review"))


# ------------------------------------------------------------------------------------- JSON
def _item(f: dict, ident: str, include_raw: bool) -> dict:
    sem = f.get("semantic") or {}
    text = f.get("text") or ""
    item = {
        "id": ident,
        "severity": f["severity"],
        "category": f["category"],
        "summary": L.short(f, 3),
        "mechanisms": f["mechanism"],
        "page": f.get("page"),
        "bbox": _round(f.get("bbox")),
        "instruction_like": bool(sem.get("instruction_like")),
        "needs_review": bool(f.get("review")),
        "benign_match": f.get("benign"),
        "text": defang(text, JSON_TEXT_LIMIT) if text else None,
        "text_truncated": len(text) > JSON_TEXT_LIMIT or None,
        "detected_by": f.get("extractors") or None,
        "source": f.get("source") or None,
    }
    if sem:
        item["semantic"] = _round({k: v for k, v in sem.items() if k != "threshold"})
    details = {k: v for k, v in (f.get("detail") or {}).items() if k not in INTERNAL_DETAIL}
    if f.get("detail", {}).get("reason_counts"):
        details["mechanism_counts"] = f["detail"]["reason_counts"]
    if details:
        item["details"] = _round(details)
    if include_raw and text:
        item["raw_text"] = text
    return {k: v for k, v in item.items() if v is not None}


def _inventory_item(f: dict, ident: str, include_raw: bool) -> dict:
    """Compact form for inventory entries."""
    text = f.get("text") or ""
    item = {"id": ident, "category": f["category"], "summary": L.short(f, 3), "mechanisms": f["mechanism"],
            "page": f.get("page"), "source": f.get("source") or None,
            "text": defang(text, JSON_TEXT_LIMIT) if text else None,
            "benign_match": f.get("benign")}
    details = {k: v for k, v in (f.get("detail") or {}).items() if k not in INTERNAL_DETAIL}
    if details:
        item["details"] = _round(details)
    if f.get("semantic"):
        item["semantic_score"] = f["semantic"].get("heuristic_score")
    if include_raw and text:
        item["raw_text"] = text
    return {k: v for k, v in item.items() if v is not None}


def build_json(result: dict, include_raw: bool = False) -> dict:
    info = result.get("info", {})
    allf = sorted(result.get("findings", []), key=_sort_key)
    fs = [f for f in allf if not is_inventory(f)]
    inv = [f for f in allf if is_inventory(f)]
    findings = [_item(f, f"F{i}", include_raw) for i, f in enumerate(fs, 1)]
    inventory = [_inventory_item(f, f"I{i}", include_raw) for i, f in enumerate(inv, 1)]

    by_sev: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for f in fs:
        by_sev[f["severity"]] = by_sev.get(f["severity"], 0) + 1
        by_cat[f["category"]] = by_cat.get(f["category"], 0) + 1
    pages = []
    for p in result.get("pages", []):
        e = {"page": p["page"], "glyphs": p.get("glyphs"), "measured_by": p.get("methods"),
             "measurement": p.get("contrib"), "ocr_lines": p.get("ocr_lines")}
        if include_raw:
            e["safe_text"] = p.get("safe_text", "")
            e["ocr_text"] = p.get("human_ocr_text", "")
        pages.append(e)

    out = {
        "verdict": result.get("verdict"),
        "verdict_reasons": result.get("verdict_reasons", []),
        "file": {k: v for k, v in {
            "path": result.get("file"), "sha256": info.get("sha256"), "bytes": info.get("bytes"),
            "pages": info.get("pages"), "objects": info.get("objects"),
            "producer": defang(info.get("metadata_producer", ""), 200) or None,
            "creator": defang(info.get("metadata_creator", ""), 200) or None,
            "encrypted": info.get("encrypted"), "repaired": info.get("repaired")}.items() if v is not None},
        "summary": {
            "findings": len(fs),
            "inventory_items": len(inv),
            "by_severity": {s: by_sev[s] for s in ("high", "medium", "low", "info") if s in by_sev},
            "by_category": by_cat,
            "instruction_like": sum(1 for f in fs if (f.get("semantic") or {}).get("instruction_like")),
            "needs_review": sum(1 for f in fs if f.get("review")),
        },
        "findings": findings,
        "inventory": inventory,
        "metadata": metadata_json(result),
        "flagged_sentences": [{**{k: v for k, v in x.items() if k != "sentence"},
                               "sentence": defang(x.get("sentence") or "", 600)}
                              for x in result.get("flagged_sentences", [])],
        "pages": pages,
        "embedded": [build_json(c, include_raw) for c in result.get("children", [])],
        "scan": {k: v for k, v in {
            "status": result.get("status"), "quarantine_reason": result.get("quarantine_reason"),
            "extractors": result.get("extractors"),
            "extractor_errors": (result.get("errors") or {}).get("extractors") or None,
            "other_errors": {k: v for k, v in (result.get("errors") or {}).items() if k != "extractors"} or None,
            "ocr": result.get("ocr_available"), "embedding": result.get("embedding_status"),
            "llm_judge": result.get("judge_status"),
            "elapsed_s": result.get("elapsed_s")}.items() if v is not None},
    }
    return out


# ------------------------------------------------------------------------------------ human
# Colour design: hierarchy comes from contrast, not from colour count.
#   verdict badge (vibrant pastel background) > headings (bold, bright) > finding titles (bold,
#   default fg) > values (terminal default fg) > labels / secondary info (quiet slate).
# Severity is carried by one pastel hue each and only on small elements (tag, quote bar), so the
# page never turns into a wall of colour. No SGR "dim" (renders muddy) and no neon bright-red.
# Colours are 24-bit; a deeper variant is used on light terminal backgrounds.

def _rgb(hexcode: str, bg: bool = False) -> str:
    h = hexcode.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"{48 if bg else 38};2;{r};{g};{b}"


_INK = "#1b1d27"  # dark text on pastel badges (readable on both themes)

PALETTES = {
    "dark": {
        "title": ("1", _rgb("#c4b5fd")),          # lavender
        "heading": ("1", _rgb("#e5e9f0")),        # near-white
        "rule": (_rgb("#3e4556"),),               # subtle separator
        "label": (_rgb("#7d879c"),),              # quiet slate
        "muted": (_rgb("#8a93a6"),),
        "strong": ("1",),
        "quote": (_rgb("#ece4d0"),),              # soft cream for extracted/attacker text
        "sev_high": ("1", _rgb("#ff8a8a")),       # coral
        "sev_medium": ("1", _rgb("#ffcf70")),     # amber
        "sev_low": (_rgb("#86c8ff"),),            # sky
        "sev_info": (_rgb("#8a93a6"),),           # slate
        "good": (_rgb("#94e2b4"),),               # mint
        "review": (_rgb("#c9b0ff"),),             # lavender
        "alert": (_rgb("#ff9a9a"),),              # soft coral text
        "bullet": (_rgb("#86c8ff"),),
        "v_MALICIOUS": ("1", _rgb("#ff8a8a")), "v_SUSPICIOUS": ("1", _rgb("#ffcf70")),
        "v_QUARANTINE": ("1", _rgb("#c9b0ff")), "v_CLEAN": ("1", _rgb("#94e2b4")),
    },
    "light": {
        "title": ("1", _rgb("#6d28d9")),
        "heading": ("1", _rgb("#1f2937")),
        "rule": (_rgb("#cbd2dc"),),
        "label": (_rgb("#64748b"),),
        "muted": (_rgb("#64748b"),),
        "strong": ("1",),
        "quote": (_rgb("#3f3a2e"),),
        "sev_high": ("1", _rgb("#c2410c")),
        "sev_medium": ("1", _rgb("#a16207")),
        "sev_low": (_rgb("#0369a1"),),
        "sev_info": (_rgb("#64748b"),),
        "good": (_rgb("#15803d"),),
        "review": (_rgb("#7e22ce"),),
        "alert": (_rgb("#c2410c"),),
        "bullet": (_rgb("#0369a1"),),
        "v_MALICIOUS": ("1", _rgb("#c2410c")), "v_SUSPICIOUS": ("1", _rgb("#a16207")),
        "v_QUARANTINE": ("1", _rgb("#7e22ce")), "v_CLEAN": ("1", _rgb("#15803d")),
    },
}
# Badges: same vibrant pastels on both themes, dark ink on top
BADGES = {
    "b_MALICIOUS": ("1", _rgb(_INK), _rgb("#ff8a8a", bg=True)),
    "b_SUSPICIOUS": ("1", _rgb(_INK), _rgb("#ffd479", bg=True)),
    "b_QUARANTINE": ("1", _rgb(_INK), _rgb("#c9b0ff", bg=True)),
    "b_CLEAN": ("1", _rgb(_INK), _rgb("#94e2b4", bg=True)),
    "b_instr": ("1", _rgb(_INK), _rgb("#ffa3c8", bg=True)),   # pastel pink
}


def detect_theme() -> str:
    """'light' if the terminal reports a light background (COLORFGBG), else 'dark'."""
    fgbg = os.environ.get("COLORFGBG", "")
    try:
        bg = int(fgbg.split(";")[-1])
        return "light" if bg == 7 or 9 <= bg <= 15 else "dark"  # ANSI: 7 and 9-15 are light
    except ValueError:
        return "dark"


class _Style:
    """Role-based ANSI styling. Disabled -> every call returns the text unchanged."""

    def __init__(self, enabled: bool, theme: str = "dark"):
        self.on = enabled
        self.roles = {**PALETTES.get(theme, PALETTES["dark"]), **BADGES}

    def __call__(self, text: str, *roles: str) -> str:
        if not self.on or not text or not roles:
            return text
        codes = [c for r in roles for c in self.roles.get(r, ())]
        return "\x1b[" + ";".join(codes) + "m" + text + "\x1b[0m" if codes else text


_TOKEN = re.compile(r"<U\+[0-9A-F]{4,6}>|.", re.S)


def _split_long(line: str, avail: int) -> list[str]:
    """Hard-split an over-long line without cutting a <U+XXXX> marker in half."""
    out, cur = [], ""
    for t in _TOKEN.findall(line):
        if cur and len(cur) + len(t) > avail:
            out.append(cur)
            cur = ""
        cur += t
    return out + ([cur] if cur else [])


def _wrap(text: str, width: int, indent: str) -> list[str]:
    avail = max(10, width - len(indent))
    out = []
    for para in text.split("\n"):
        for line in textwrap.wrap(para, width=avail, break_long_words=False,
                                  replace_whitespace=False) or [""]:
            out += _split_long(line, avail) if len(line) > avail else [line]
    return [indent + l for l in out]


def _kv(key: str, value: str, width: int, S: "_Style | None" = None, *vstyle: str) -> list[str]:
    """'    Key        : value' with wrapping; styling applied after wrapping so widths stay right."""
    S = S or _Style(False)
    pad = "    " + f"{key:<11}: "
    lines = textwrap.wrap(value, width=width - len(pad)) or [""]
    first = "    " + S(f"{key:<11}", "label") + S(": ", "rule") + S(lines[0], *vstyle)
    return [first] + [" " * len(pad) + S(l, *vstyle) for l in lines[1:]]


def _detail_lines(f: dict, width: int, S: _Style) -> list[str]:
    d = f.get("detail") or {}
    out = []
    if "ocr_text" in d:
        out += _kv("OCR reads", f"\"{defang(d['ocr_text'], 300)}\"", width, S, "quote")
    if "similarity" in d:
        out += _kv("Similarity", f"{d['similarity']}% between extracted and rendered text", width, S)
    if d.get("novel_spans"):
        n = len(d["novel_spans"])
        out += _kv("Adds", f"{d.get('novel_words', '?')} of {d.get('total_words', '?')} words form "
                           f"{n} stretch{'es' if n != 1 else ''} not visible on the page (in this order):",
                   width, S)
        for sp in d["novel_spans"][:6]:
            for i, l in enumerate(_wrap(defang(sp, 600), width, " " * 17 + "> ")):
                out.append(" " * 17 + S("> " if i == 0 else "  ", "alert") + S(l[19:], "quote"))
    elif "total_words" in d and f.get("benign", "").startswith("redundant"):
        out += _kv("Adds", "nothing: every stretch of it is visible on the page", width, S, "good")
    if d.get("verification"):
        out += _kv("Unverified", d["verification"], width, S, "review")
    if "words_not_in_final" in d:
        out += _kv("Removed", ", ".join(d["words_not_in_final"][:20]), width, S)
    if "count" in d and f["category"] not in ("hidden_text", "char_level"):
        out += _kv("Count", str(d["count"]), width, S)
    if "bytes" in d:
        out += _kv("Size", f"{d['bytes']} bytes", width, S)
    if "words_by_extractor_set" in d:
        for k, v in list(d["words_by_extractor_set"].items())[:4]:
            out += _kv("Only in", f"{k or '(none)'}: {', '.join(v[:12])}", width, S)
    if f["category"] == "char_level":
        if "count" in d:
            out += _kv("Count", str(d["count"]), width, S)
        if d.get("examples"):
            out += _kv("Examples", ", ".join(defang(x, 40) for x in d["examples"][:8]), width, S)
    if d.get("steps"):
        out += _kv("Does", " -> then ".join(defang(s, 200) for s in d["steps"]), width, S)
    if d.get("pages") and len(d["pages"]) > 1:
        out += _kv("Pages", ", ".join(map(str, d["pages"])) + f" ({d.get('occurrences')} occurrences)", width, S)
    if d.get("note"):
        out += _kv("Note", d["note"], width, S, "muted")
    return out


def _inventory_line(f: dict, width: int, S: _Style) -> list[str]:
    """One compact entry per inventory item: what it is, where, and its text (first line)."""
    head = f"{L.category(f['category'])}: {'; '.join(L.mech(m) for m in f['mechanism'][:2])}"
    where = L.where(f)
    d = f.get("detail") or {}
    if d.get("pages") and len(d["pages"]) > 1:
        where = "pages " + ", ".join(map(str, d["pages"]))
    text = ""
    if d.get("steps"):
        text = " -> then ".join(d["steps"])
    elif f.get("text"):
        lines = [l for l in defang(f["text"], 400).split("\n") if l.strip()]
        text = lines[0] + (f"  (+{len(lines) - 1} more lines)" if len(lines) > 1 else "")
    wrapped = _wrap(f"{head}  [{where}]", width, "      ")
    first = wrapped[0][6:]
    if first.endswith(f"[{where}]"):
        first = first[: -len(where) - 2] + S(f"[{where}]", "muted")
    out = ["    " + S("•" if S.on else "-", "bullet") + " " + first] + wrapped[1:]
    if text:
        out += ["      " + S(l[6:], "muted") for l in _wrap(text, width, "      ")[:2]]
    return out


def _reason_lines(r: str, width: int, S: _Style, verdict: str = "") -> list[str]:
    """Verdict reason '[SUSPICIOUS] ...': wrapped plain, then the leading tag is coloured.
    The tag is dropped when it just repeats the overall verdict shown in the badge above."""
    if verdict and r.startswith(f"[{verdict}] "):
        r = r[len(verdict) + 3:]
        r = r[:1].upper() + r[1:]
    lines = _wrap(r, width, "    - ")[:1] + _wrap(r, width, "      ")[1:]
    if S.on and lines and lines[0].startswith("    - ") and not lines[0].startswith("    - ["):
        lines[0] = "    " + S("•", "rule") + lines[0][5:]
    if S.on and lines and lines[0].startswith("    - ["):
        end = lines[0].find("]") + 1
        tag = lines[0][6:end]
        lines[0] = "    " + S("•", "rule") + " " + S(tag, "v_" + tag.strip("[]")) + lines[0][end:]
    return lines


def _metadata_section(result: dict, width: int, S: _Style) -> list[str]:
    """Every metadata value in full, for a person to read. Automated checks are pattern-based
    (and an LLM judge can be argued with), so these are shown regardless of what they found."""
    md = result.get("metadata")
    if md is None:
        return []
    info, xmp = md.get("info", []), md.get("xmp", [])
    n = len(info) + len(xmp)
    judge = result.get("judge_status", "off")
    checked = "instruction patterns, hidden/encoded characters" + (", LLM judge" if judge.startswith("ok") else "")
    out = ["", "  " + S("Metadata", "heading") + "  " + S(str(n), "strong")
           + S("   shown in full for human review", "muted"), _rule(width, S)]
    out += ["  " + S(l[2:], "muted") for l in _wrap(
        f"Automated checks ({checked}) are pattern-based and can be bypassed; read the values "
        f"yourself. Invisible characters are shown as <U+XXXX>, links are defanged.", width, "  ")]
    out.append("")
    if not n:
        out.append("    " + S("none", "good"))
        return out
    names = [k for k, _ in info + xmp]
    col = min(max(len(k) for k in names), 24)
    indent = " " * (6 + col + 2)

    def block(title, pairs, note=""):
        if not pairs:
            return []
        lines = ["    " + S(title, "strong") + (S(f"  {note}", "muted") if note else "")]
        for k, v in pairs:
            key = k if len(k) <= col else k[: col - 1] + "~"
            if len(v) > 2000 and _B64ISH.match(v[:4000]):
                shown = f"[{len(v)} characters of base64-like data, not shown]"
                vstyle = "muted"
            else:
                shown, vstyle = defang(v, META_VALUE_LIMIT), "quote"
            wrapped = _wrap(shown, width, indent)
            lines.append("      " + S(f"{key:<{col}}", "label") + "  " + S(wrapped[0][len(indent):], vstyle))
            lines += [indent + S(l[len(indent):], vstyle) for l in wrapped[1:]]
            for fl in meta_flags(v):
                for i, l in enumerate(_wrap(fl, width, indent + "! ")):
                    lines.append(indent + S(("! " if i == 0 else "  ") + l[len(indent) + 2:], "alert"))
        return lines

    out += block("Info dictionary", info)
    if info and xmp:
        out.append("")
    out += block("XMP", xmp, f"{len(xmp)} properties, {md.get('xmp_bytes', 0)} bytes")
    return out


def _flagged_section(result: dict, width: int, S: _Style, limit: int = 25) -> list[str]:
    """Visible sentences the embedding model thinks may be addressed to an AI. For review only:
    they do not change the verdict (patterns/judge decide that)."""
    fl = result.get("flagged_sentences") or []
    if not fl:
        return []
    out = ["", "  " + S("Flagged sentences", "heading") + "  " + S(str(len(fl)), "strong")
           + S("   strong embedding signal, close to a known injection; for review, verdict unchanged", "muted"),
           _rule(width, S)]
    for x in fl[:limit]:
        tag = f"p{x.get('page')}".ljust(4)
        tier = (x.get("tier") or "").ljust(9)
        head = "    " + S(tag, "muted") + " " + S(tier, "sev_medium" if x.get("tier") == "strong" else "sev_low")             + S(f"{x.get('probability') or 0:.2f}", "muted") + "  "
        lines = _wrap(defang(x.get("sentence") or "", 400), width, " " * 24)
        out.append(head + S(lines[0][24:], "quote"))
        out += [" " * 24 + S(l[24:], "quote") for l in lines[1:]]
    if len(fl) > limit:
        out.append(S(f"    ... {len(fl) - limit} more in the JSON output", "muted"))
    return out


def _rule(width: int, S: _Style) -> str:
    return "  " + S(("─" if S.on else "-") * (width - 2), "rule")


def human_report(result: dict, show_info: bool = False, width: int = 100, color: bool = False,
                 theme: str = "dark", _nested: bool = False) -> str:
    S = _Style(color, theme)
    out: list[str] = []
    info = result.get("info", {})
    verdict = result.get("verdict", "?")
    if not _nested:
        out += ["", "  " + S("PDF prompt-injection scan", "title"), ""]

    def meta(key, value):
        out.append("  " + S(f"{key:<10}", "label") + S(": ", "rule") + value)

    meta("File", S(os.path.basename(str(result.get("file", ""))), "strong"))
    if not _nested:
        meta("Path", S(str(result.get("file")), "muted"))
    if info:
        meta("SHA-256", S(str(info.get("sha256")), "muted"))
        prod = defang(info.get("metadata_producer", ""), 60) or "unknown"
        meta("Pages", f"{info.get('pages')}  " + S(f"({info.get('objects')} objects, producer: {prod})", "muted"))
    ocr = S("yes", "good") if result.get("ocr_available") else S("NO", "sev_high")
    checks = [f"{len(result.get('extractors', []))} extractors "
              + S(f"({', '.join(result.get('extractors', []))})", "muted"),
              f"OCR {ocr}",
              "embedding " + (S("on", "good") if str(result.get("embedding_status", "off")).startswith("ok")
                              else S(str(result.get("embedding_status", "off")).split(" (")[0], "muted")),
              f"LLM judge {result.get('judge_status', 'off')}"]
    if result.get("elapsed_s") is not None:
        checks.append(S(f"{result['elapsed_s']} s", "muted"))
    meta("Checks", ", ".join(checks))
    errs = (result.get("errors") or {}).get("extractors") or {}
    for k, v in errs.items():
        out.append("  " + S(f"! extractor {k} failed: {v.splitlines()[0][:80]}", "sev_medium"))
    out.append("")
    badge = S(f" {verdict} ", "b_" + verdict) if S.on else verdict
    out.append("  " + S("Verdict", "heading") + "  " + badge)
    for r in result.get("verdict_reasons", [])[:12]:
        out += _reason_lines(r, width, S, verdict)
    if len(result.get("verdict_reasons", [])) > 12:
        out.append(S(f"    ... and {len(result['verdict_reasons']) - 12} more", "muted"))
    out.append("")

    allf = sorted(result.get("findings", []), key=_sort_key)
    shown = [f for f in allf if not is_inventory(f)]
    inventory = [f for f in allf if is_inventory(f)]
    if result.get("metadata") is not None:
        # metadata gets its own full section below; don't list it twice
        inventory = [f for f in inventory if not set(f["mechanism"]) & META_MECHS]
    counts = {}
    for f in shown:
        counts[f["severity"]] = counts.get(f["severity"], 0) + 1
    cs = S(", ", "rule").join(S(f"{counts[s]} {s}", "sev_" + s) for s in ("high", "medium", "low", "info")
                             if s in counts)
    head = "  " + S("Findings", "heading") + "  " + S(str(len(shown)), "strong") + (f"  {cs}" if cs else "")
    if inventory and not show_info:
        n = len(inventory)
        head += S(f"   inventory: {n} item{'s' if n != 1 else ''} (--show-info)", "muted")
    out += [head, _rule(width, S)]
    if not shown:
        out.append("    " + S("none", "good"))

    for i, f in enumerate(shown, 1):
        sem = f.get("semantic") or {}
        sev = f["severity"]
        flag = ""
        if sem.get("instruction_like"):
            flag = "  " + (S(" INSTRUCTION-LIKE ", "b_instr") if S.on else "** INSTRUCTION-LIKE **")
        out.append("")
        out.append("  " + S(f"{i:>2}", "muted") + "  " + S(f"{sev.upper():<6}", "sev_" + sev) + " "
                   + S(L.category(f["category"]), "strong") + S("  ·  " if S.on else "  -  ", "rule")
                   + S(L.where(f), "muted") + flag)
        out += _kv("How", "; ".join(L.mech(m) for m in f["mechanism"]), width, S)
        if f.get("bbox"):
            b = f["bbox"]
            out += _kv("Location", f"x {b[0]:.0f}-{b[2]:.0f}, y {b[1]:.0f}-{b[3]:.0f} pt (from top-left)", width, S)
        if f.get("extractors"):
            out += _kv("Found by", ", ".join(L.detector(x) for x in f["extractors"]), width, S)
        if f.get("benign"):
            out += _kv("Downgraded", f"benign pattern: {f['benign'].replace('_', ' ')}", width, S, "good")
        if f.get("review"):
            out += _kv("Review", "needs a human look", width, S, "review")
        if sem.get("labels") or sem.get("judge") or sem.get("embedding"):
            parts = []
            if sem.get("labels"):
                parts.append(f"patterns {sem.get('heuristic_score')} ("
                             + ", ".join(L.SEMANTIC.get(x, x) for x in sem["labels"]) + ")")
            else:
                parts.append("patterns: none")
            e = sem.get("embedding")
            if e:
                tier = {"strong": "strong", "moderate": "moderate", "none": "no signal"}[e.get("tier", "none")]
                via = f", {e['member']}" if len(e.get("members") or {}) > 1 and e.get("tier", "none") != "none" else ""
                parts.append(f"embedding {e.get('probability')} ({tier}{via})")
            if sem.get("judge"):
                j = sem["judge"]
                parts.append(f"judge: {j.get('category')} ({j.get('confidence')})")
            alert = sem.get("instruction_like") or (e or {}).get("tier", "none") != "none"
            out += _kv("Semantic", " · ".join(parts) if S.on else "; ".join(parts), width, S,
                       *(("alert",) if alert else ()))
            if e and e.get("tier", "none") != "none":
                if e.get("sentence") and e["sentence"].strip() != (f.get("text") or "").strip():
                    out += _kv("Strongest", f"\"{defang(e['sentence'], 300)}\"", width, S, "quote")
                if sem.get("embedding_only"):
                    out += _kv("Signals", "embedding only: flagged for review, not treated as an attack "
                                          "without a second signal", width, S, "review")
                elif sem.get("corroborated_by_embedding"):
                    out += _kv("Signals", "embedding and patterns agree", width, S, "alert")
        out += _detail_lines(f, width, S)
        text = f.get("text") or ""
        if text:
            out.append("    " + S(f"{'Text':<11}", "label") + S(":", "rule"))
            bar = S("│" if S.on else "|", "sev_" + sev if sev in ("high", "medium") else "rule")
            for line in _wrap(defang(text, HUMAN_TEXT_LIMIT), width, "      | "):
                out.append("      " + bar + " " + S(line[8:], "quote"))
    out += _flagged_section(result, width, S)
    out += _metadata_section(result, width, S)
    if inventory and show_info:
        out += ["", "  " + S("Inventory", "heading") + "  " + S(str(len(inventory)), "strong")
                + S("   what the document contains besides page text; nothing wrong found", "muted"),
                _rule(width, S)]
        for f in inventory:
            out += _inventory_line(f, width, S)
    for c in result.get("children", []):
        out += ["", _rule(width, S), "  " + S("Embedded file", "heading") + "  " + S(str(c.get("file")), "strong"),
                _rule(width, S)]
        out.append(human_report(c, show_info, width, color, theme, _nested=True))
    out.append("")
    return "\n".join(out)
