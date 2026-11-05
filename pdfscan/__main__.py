"""CLI: python -m pdfscan <file.pdf> [options]

Parsing runs in a separate isolated, time-limited Python process (the "parse sandbox"); model
calls (optional LLM judge) happen only in this parent process and only on extracted text.
Exit codes: 0 CLEAN, 1 SUSPICIOUS, 2 MALICIOUS, 3 QUARANTINE.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from . import differential, embedding, policy, report, sanitize, semantic
from .util import Finding

PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXIT = {"CLEAN": 0, "SUSPICIOUS": 1, "MALICIOUS": 2, "QUARANTINE": 3}


def _isolated(code: str, args: list[str], timeout: int, cwd: str) -> subprocess.CompletedProcess:
    boot = f"import sys; sys.path.insert(0, {PKG_PARENT!r}); {code}"
    env = {k: v for k, v in os.environ.items()
           if k.upper() in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "WINDIR", "HOME", "USERPROFILE", "LANG", "APPDATA", "LOCALAPPDATA")}
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run([sys.executable, "-E", "-P", "-c", boot, *args], cwd=cwd, env=env,
                          capture_output=True, timeout=timeout)


def _fmt_time(s: float) -> str:
    s = int(max(0, s))
    return f"{s // 60}:{s % 60:02d}"


class Progress:
    """Progress bar on stderr. Animated on a terminal; plain throttled lines otherwise."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.tty = sys.stderr.isatty()
        self.t0 = time.time()
        self.frac = 0.0
        self.msg = "starting"
        self.last_plain = (-1.0, "")
        self.width = 0

    def update(self, frac, msg):
        if frac is not None:
            self.frac = max(self.frac, min(1.0, frac))
        self.msg = msg
        self.draw()

    def draw(self):
        if not self.enabled:
            return
        el = time.time() - self.t0
        # ETA from overall rate; only once there is enough signal to be meaningful
        eta = f"ETA {_fmt_time(el / self.frac - el)}" if self.frac >= 0.12 else "ETA estimating..."
        if self.tty:
            cols = shutil.get_terminal_size((100, 20)).columns
            bar_w = 24
            filled = int(bar_w * self.frac)
            head = f"[{'#' * filled}{'-' * (bar_w - filled)}] {self.frac * 100:3.0f}%  {_fmt_time(el)} elapsed  {eta}  "
            line = (head + self.msg)[: cols - 1]
            sys.stderr.write("\r" + line + " " * max(0, self.width - len(line)))
            self.width = len(line)
            sys.stderr.flush()
        else:
            key = (round(self.frac, 1), self.msg.split(":")[0])
            if key != self.last_plain:
                self.last_plain = key
                sys.stderr.write(f"[{self.frac * 100:3.0f}%] {_fmt_time(el)} elapsed, {eta}: {self.msg}\n")
                sys.stderr.flush()

    def done(self, msg="done"):
        if not self.enabled:
            return
        self.frac = 1.0
        self.msg = msg
        self.draw()
        if self.tty:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _isolated_stream(code: str, args: list[str], timeout: int, cwd: str, progress: Progress):
    """Like _isolated, but streams worker stderr: progress lines update the bar, the rest is kept."""
    boot = f"import sys; sys.path.insert(0, {PKG_PARENT!r}); {code}"
    env = {k: v for k, v in os.environ.items()
           if k.upper() in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "WINDIR", "HOME", "USERPROFILE", "LANG",
                            "APPDATA", "LOCALAPPDATA")}
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.Popen([sys.executable, "-E", "-P", "-c", boot, *args], cwd=cwd, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    other: list[str] = []

    def reader():
        for raw in proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if line.startswith("@@PROGRESS "):
                try:
                    p = json.loads(line[11:])
                    progress.update(p.get("f"), p.get("m", ""))
                except ValueError:
                    pass
            else:
                other.append(line)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while proc.poll() is None:
        if time.time() > deadline:
            proc.kill()
            proc.wait()
            t.join(2)
            raise subprocess.TimeoutExpired(args, timeout)
        time.sleep(0.25)
        progress.draw()  # keeps elapsed/ETA ticking during long steps
    t.join(5)
    return proc.returncode, "\n".join(other)


def run_worker(pdf: str, zoom: float, ocr: str, faint: bool, timeout: int, progress: Progress) -> dict:
    work = tempfile.mkdtemp(prefix="pdfscan_")
    try:
        sample = os.path.join(work, "sample.pdf")
        shutil.copyfile(pdf, sample)
        out = os.path.join(work, "result.json")
        args = [sample, out, "--zoom", str(zoom), "--ocr", ocr] + ([] if faint else ["--no-faint"])
        try:
            code, err = _isolated_stream("from pdfscan.worker import main; main(sys.argv[1:])",
                                         args, timeout, work, progress)
        except subprocess.TimeoutExpired:
            return {"file": os.path.abspath(pdf), "status": "quarantine",
                    "quarantine_reason": f"parser timeout after {timeout}s"}
        if not os.path.exists(out):
            return {"file": os.path.abspath(pdf), "status": "quarantine",
                    "quarantine_reason": f"parser process died (exit {code}): {err[-500:]}"}
        with open(out, encoding="utf-8") as f:
            res = json.load(f)
        res["file"] = os.path.abspath(pdf) if res.get("file", "").endswith("sample.pdf") else res.get("file")
        return res
    finally:
        shutil.rmtree(work, ignore_errors=True)


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[A-Z0-9])")


_CODE_CHARS = set("{}();<>=[]|&")
_CODE_WORDS = re.compile(r"\b(int|void|return|cout|cin|endl|std|def|class|import|const|public|private|"
                         r"assign|module|endmodule|printf|include|elif|lambda|var|let|function)\b")


def looks_like_code(s: str) -> bool:
    """Source-code lines (listings in assignments, docs). Out of the embedding probe's domain."""
    s = s.strip()
    if not s:
        return False
    sym = sum(c in _CODE_CHARS for c in s) / len(s)
    return sym > 0.06 or (sym > 0.02 and len(_CODE_WORDS.findall(s)) >= 1) or len(_CODE_WORDS.findall(s)) >= 3


def visible_sentences(safe_text: str) -> list[str]:
    """Visible text as sentences, cut along the PDF's own blocks (blank lines in the safe text)
    first, so slide bullets / table cells / TOC lines are not glued into one string. Layout junk
    that does not read as language is skipped."""
    out = []
    for block in re.split(r"\n\s*\n", safe_text):
        block = " ".join(l.strip() for l in block.splitlines() if l.strip())
        block = re.sub(r"([.․…·_\-=*])\1{3,}", " ", block)
        # list items are separate statements: split on bullet characters too
        for item in re.split(r"\s*[•▪●◦‣⁃]\s*", block):
            for s in _SENT_SPLIT.split(item):
                s = s.strip()
                # a long single token still counts (e.g. a remapped font spelling out a run-together
                # phrase); patterns check it even when it is too short for the embedding
                # (run_embedding applies its own stricter "reads as language" filter)
                letters = sum(c.isalpha() for c in s)
                if (len(s.split()) >= 3 or len(s) >= 15) and letters >= 0.5 * len(s.replace(" ", "")):
                    out.append(s)
    return list(dict.fromkeys(out))


def add_visible_candidates(result: dict) -> None:
    """Visible body text gets semantic analysis too, sentence by sentence (higher threshold).
    Candidates without any signal are pruned again after scoring."""
    for p in result.get("pages", []):
        for s in visible_sentences(p.get("safe_text", "")):
            result["findings"].append(Finding("visible_text", ["visible_body_text"], "info",
                                              page=p["page"], text=s, extractors=["safe_text"]).to_dict())
    for c in result.get("children", []):
        add_visible_candidates(c)


# categories whose text is worth asking "does this read like instructions to an AI?"
EMBED_CATS = {"hidden_text", "mismatched_text", "offpage_content", "char_level", "vision", "visible_text",
              "structural"}


def run_embedding(result: dict) -> str:
    """Score every candidate text with the embedding signal in one batch (all nested results)."""
    sig = embedding.get()
    if sig is None:
        return f"unavailable ({embedding.last_error()})"
    targets = []

    def collect(r):
        for f in r.get("findings", []):
            if f["category"] not in EMBED_CATS or not f.get("text"):
                continue
            if f["category"] == "structural" and not set(f["mechanism"]) & policy.STRUCT_TEXT_MECH:
                continue
            if f.get("benign") == "standard_tool_and_date_fields":
                continue  # "Producer: ... CreationDate: D:2026..." is not prose; scores there are noise
            spans = (f.get("detail") or {}).get("novel_spans")
            if spans:
                units = spans
            elif set(f["mechanism"]) & {"metadata_xmp", "metadata_info"} and re.search(r"^[\w:.\-]+: ", f["text"], re.M):
                # "key: value" listings (XMP, Info): score each value on its own, never the glued blob
                units = [l.split(": ", 1)[1] for l in f["text"].splitlines() if ": " in l]
            else:
                units = [f["text"]]
            for u in units:
                # the probe was trained on prose: skip text that is not language or looks like code
                # (patterns still check these)
                if differential.meaningful(u) and not looks_like_code(u):
                    targets.append((f, u))
        for c in r.get("children", []):
            collect(c)
    collect(result)
    t = time.perf_counter()
    # visible sentences: primary model only (bulk of the work, review-only flags); everything else
    # (hidden text, metadata, annotations, decoded payloads...): the full ensemble
    vis = [(f, x) for f, x in targets if f["category"] == "visible_text"]
    rest = [(f, x) for f, x in targets if f["category"] != "visible_text"]
    scored = list(zip(vis, sig.score_units([x for _, x in vis], use_members=False))) + \
        list(zip(rest, sig.score_units([x for _, x in rest])))
    for (f, _), res in scored:
        if res["score"] is not None and (f.get("embedding") is None or res["score"] > f["embedding"]["score"]):
            f["embedding"] = res  # a finding with several units keeps its strongest
    members = 1 + len(sig.members)
    missing = f"; ensemble member unavailable: {sig.missing_members[0]}" if sig.missing_members else ""
    return (f"ok ({len(targets)} texts, {time.perf_counter() - t:.2f} s"
            + (f"; {members}-model ensemble on {len(rest)} non-visible texts" if sig.members else "")
            + missing + ")")


def prune_visible(result: dict) -> None:
    """Visible sentences: patterns/judge hits stay findings (verdict SUSPICIOUS); embedding-only hits
    become 'flagged sentences' for review (no verdict change); the rest is dropped."""
    keep, flagged = [], []
    for f in result["findings"]:
        if f["category"] == "visible_text":
            sem = f.get("semantic") or {}
            if sem.get("instruction_like"):
                f["severity"] = "medium"
                f["mechanism"] = ["visible_instruction_like"]
            elif sem.get("highlight"):
                e = sem.get("embedding") or {}
                flagged.append({"page": f.get("page"), "tier": e.get("tier"), "probability": e.get("probability"),
                                "sentence": f.get("text"), "nearest_example": e.get("nearest_example")})
                continue
            else:
                continue
        keep.append(f)
    result["findings"] = keep
    result["flagged_sentences"] = sorted(flagged, key=lambda x: (-(x["probability"] or 0), x["page"] or 0))
    for c in result.get("children", []):
        prune_visible(c)


def run_judge(result: dict, model: str) -> str:
    """Send candidate texts (non-benign-info) to the hardened judge; attach results."""
    items = []  # (result_dict, finding_index)

    def collect(r):
        for i, f in enumerate(r.get("findings", [])):
            if not f.get("text") or f["category"] == "attachment":
                continue
            if f["severity"] == "info" and f.get("benign") and f["category"] != "visible_text":
                continue
            if f["category"] == "visible_text" and (f.get("embedding") or {}).get("tier", "none") == "none" \
                    and not semantic.classify(f["text"])["labels"]:
                continue  # only visible sentences some cheaper signal already noticed
            items.append((r, i))
        for c in r.get("children", []):
            collect(c)
    collect(result)
    items = items[:120]
    if not items:
        return "ok (nothing to judge)"

    def fragment(f):
        # for hidden text, judge what it ADDS (novel spans), not the padding copied from the page
        spans = (f.get("detail") or {}).get("novel_spans")
        return "\n".join(spans) if spans else f["text"]
    res = semantic.llm_judge([fragment(r["findings"][i]) for r, i in items], model)
    if res.get("anomaly"):
        result["findings"].append(Finding("structural", ["judge_anomaly", res["anomaly"]], "medium",
                                          source="llm_judge", detail={"error": res.get("error")}).to_dict())
    if not res.get("ok") and not res.get("results"):
        return f"failed: {res.get('error')}"
    by_result: dict[int, dict] = {}
    for k, (r, i) in enumerate(items):
        if k in res.get("results", {}):
            by_result.setdefault(id(r), {})[i] = res["results"][k]
    result["_judge"] = by_result
    return "ok" if res.get("ok") else f"partial: {res.get('error')}"


def finalize(result: dict, judge_map: dict | None) -> None:
    policy.apply_semantics(result.get("findings", []), (judge_map or {}).get(id(result)))
    for c in result.get("children", []):
        finalize(c, judge_map)
        c["verdict"], c["verdict_reasons"] = policy.verdict(c)


def _enable_ansi() -> None:
    """Turn on ANSI escape processing in the Windows console (no-op elsewhere / on failure)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if k32.GetConsoleMode(h, ctypes.byref(mode)):
            k32.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        k32.SetConsoleOutputCP(65001)  # report is written as UTF-8 (rules, bullets)
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pdfscan", description="PDF prompt-injection scanner")
    ap.add_argument("pdf")
    ap.add_argument("--format", choices=["json", "human"], default="json",
                    help="stdout format: json (default, for agents) or human (readable report)")
    ap.add_argument("--human", action="store_const", const="human", dest="format",
                    help="shortcut for --format human")
    ap.add_argument("-o", "--output", help="write the report (in --format) to this file instead of stdout")
    ap.add_argument("--json", help="additionally write the JSON report to this file")
    ap.add_argument("--width", type=int, default=100, help="line width of the human report")
    ap.add_argument("--color", choices=["auto", "always", "never"], default="auto",
                    help="colour the human report (auto: only on a terminal, honours NO_COLOR)")
    ap.add_argument("--theme", choices=["auto", "dark", "light"], default="auto",
                    help="colour palette for the terminal background (auto: from COLORFGBG, else dark)")
    ap.add_argument("--safe-text", help="write human-perceptible text only (content disarm)")
    ap.add_argument("--flatten", help="write a re-rendered, image-only sanitized PDF")
    ap.add_argument("--include-raw", action="store_true", help="include raw (undefanged) text in JSON")
    ap.add_argument("--semantic", choices=["regex", "embed", "all"], default="all",
                    help="instruction-likeness signals: regex patterns, the local embedding model, or both "
                         "(default; the embedding is skipped with a note if its files are missing)")
    ap.add_argument("--llm", action="store_true", help="enable the hardened Claude judge")
    ap.add_argument("--llm-model", default="claude-opus-5-5")
    ap.add_argument("--zoom", type=float, default=2.0, help="render scale (2.0 = 144 dpi)")
    ap.add_argument("--ocr", choices=["off", "targeted", "full"], default="targeted",
                    help="OCR safety net: targeted (default; only regions nothing else can decide, e.g. "
                         "hidden text over a scan), full (whole pages: slow), off")
    ap.add_argument("--no-ocr", action="store_const", const="off", dest="ocr", help="same as --ocr off")
    ap.add_argument("--no-faint", action="store_true", help="with --ocr full: skip the faint-text OCR pass")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--show-info", action="store_true", help="human format: also list info-level findings")
    ap.add_argument("--quiet", action="store_true", help="no progress output on stderr")
    a = ap.parse_args(argv)

    progress = Progress(not a.quiet)
    result = run_worker(a.pdf, a.zoom, a.ocr, not a.no_faint, a.timeout, progress)
    if result.get("status") != "quarantine":
        add_visible_candidates(result)
        result["embedding_status"] = "off"
        if a.semantic in ("embed", "all"):
            progress.update(None, "embedding signal")
            result["embedding_status"] = run_embedding(result)
        result["judge_status"] = "off"
        if a.llm:
            progress.update(None, "LLM judge")
            result["judge_status"] = run_judge(result, a.llm_model)
        finalize(result, result.pop("_judge", None))
        prune_visible(result)
    verdict, reasons = policy.verdict(result)
    result["verdict"], result["verdict_reasons"] = verdict, reasons

    if a.safe_text and result.get("status") != "quarantine":
        with open(a.safe_text, "w", encoding="utf-8") as f:
            f.write(sanitize.safe_text(result))
    if a.flatten and result.get("status") != "quarantine":
        progress.update(None, "writing flattened PDF")
        try:
            _isolated("from pdfscan.sanitize import flatten; flatten(sys.argv[1], sys.argv[2])",
                      [os.path.abspath(a.pdf), os.path.abspath(a.flatten)], a.timeout, tempfile.gettempdir())
        except subprocess.TimeoutExpired:
            print("flatten timed out", file=sys.stderr)
    js = json.dumps(report.build_json(result, a.include_raw), indent=2, ensure_ascii=False)
    # keep numeric arrays (bboxes) on one line
    js = re.sub(r"\[\s+(-?[\d.]+(?:,\s+-?[\d.]+)*)\s+\]",
                lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", js)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(js + "\n")

    progress.done(f"done: {verdict}")
    color = a.format == "human" and (
        a.color == "always" or (a.color == "auto" and not a.output and sys.stdout.isatty()
                                and not os.environ.get("NO_COLOR")))
    if color:
        _enable_ansi()
    theme = report.detect_theme() if a.theme == "auto" else a.theme
    out = js if a.format == "json" else report.human_report(result, a.show_info, a.width, color, theme)
    if a.output:
        with open(a.output, "w", encoding="utf-8") as f:
            f.write(out + "\n")
    else:
        sys.stdout.buffer.write((out + "\n").encode("utf-8", "replace"))
        sys.stdout.flush()
    return EXIT[verdict]


if __name__ == "__main__":
    sys.exit(main())
