"""§3.8 context weighting + §3.9 deterministic policy engine."""
from __future__ import annotations

from . import embedding as _embedding
from . import labels, semantic

HIDDEN_CATS = {"hidden_text", "mismatched_text", "char_level", "vision"}
STRUCT_TEXT_MECH = {"data_after_eof", "orphaned_content_stream", "earlier_revision_text_differs",
                    "data_before_header", "clip_only_render_mode"}
RANK = {"CLEAN": 0, "SUSPICIOUS": 1, "QUARANTINE": 2, "MALICIOUS": 3}
# visible sentences are flagged for review only if the embedding is strong AND the sentence is this
# close (cosine) to a known training injection (tuned on 6 real PDFs: research/visible_diag.py)
VISIBLE_MIN_EXEMPLAR_COSINE = 0.74


def context_of(f: dict) -> str:
    if f["category"] in HIDDEN_CATS or set(f["mechanism"]) & STRUCT_TEXT_MECH:
        return "hidden"
    if f["category"] == "visible_text":
        return "visible"
    return "offpage"


def apply_semantics(findings: list[dict], judge: dict | None = None) -> None:
    """Attach semantic results. judge maps finding index -> judge item."""
    for i, f in enumerate(findings):
        if not f.get("text") or f["category"] in ("attachment",):
            continue
        ctx = context_of(f)
        spans = (f.get("detail") or {}).get("novel_spans")
        redundant = (f.get("benign") or "").startswith("redundant")
        # For hidden/mismatched text with a redundancy analysis, the question is what the hidden
        # text ADDS: classify the novel spans on their own. Padding a payload with a copy of the
        # visible text then neither dilutes it nor hides it.
        target = "\n".join(spans) if spans else f["text"]
        h = semantic.classify_long(target)
        sem = {"context": ctx, "heuristic_score": h["score"], "labels": h["all_labels"],
               "threshold": semantic.THRESHOLDS[ctx], "analysed": "novel_spans" if spans else "full_text"}
        regex_hit = h["score"] >= sem["threshold"]
        emb = f.get("embedding") or {}
        tier = emb.get("tier", "none")
        if emb:
            sem["embedding"] = {k: emb[k] for k in ("score", "probability", "tier", "sentence", "nearest_example", "member", "members")
                                if k in emb}
        judge_hit = False
        if judge and i in judge:
            j = judge[i]
            sem["judge"] = j
            judge_hit = bool(j.get("injection") and j.get("confidence", 0) >= 0.5)
            if (regex_hit or tier != "none") and not j.get("injection"):
                sem["judge_disagrees"] = True
        # The embedding only ever ADDS evidence. Alone it never makes text "instruction-like" in the
        # MALICIOUS sense for hidden/unreviewed text: that needs a second, independent signal
        # (regex patterns or the judge). Corroboration: a strong embedding hit plus any regex label,
        # or a moderate hit plus a non-trivial regex score.
        corroborated = (tier == "strong" and h["score"] > 0) or (tier != "none" and h["score"] >= 0.35) \
            or (tier != "none" and judge_hit)
        sem["corroborated_by_embedding"] = corroborated and not regex_hit and not judge_hit
        if ctx == "visible":
            # visible text: the embedding only flags sentences for review (any tier); the verdict
            # moves only on patterns or the judge. Real assignments/manuals legitimately contain
            # instruction-like sentences near the human/AI-directed boundary.
            instr = regex_hit or judge_hit
            # Only the STRONG tier (0.1% per-sentence FPR) flags visible sentences. A document has
            # 100-300 sentences, so the 1% "moderate" tier produced 1-8 unrelated flags per document;
            # with injections planted into real PDFs, strong-only cut false flags 19 -> 2 while still
            # catching 88% of them (research/visible_diag.py).
            # Second gate: the sentence must also be close to a known injection. Strong-only still
            # flagged FAQ/second-person course text ("Do I print with endl...?"): 7 unrelated flags on
            # 6 real PDFs (~880 sentences); with nearest-exemplar cosine >= 0.74: 0 flags, 60% of
            # planted injections still flagged. Precision first: these flags never change the verdict.
            near = ((emb.get("nearest_example") or {}).get("cosine") or 0.0)
            # the gate belongs to the ensemble member that produced the result (embedding spaces differ)
            sig = _embedding._INSTANCE
            cos_min = emb.get("visible_min_cosine",
                              sig.visible_min_cosine if sig is not None else VISIBLE_MIN_EXEMPLAR_COSINE)
            sem["highlight"] = tier == "strong" and near >= cos_min and not instr
        else:
            instr = regex_hit or judge_hit or corroborated
            sem["embedding_only"] = tier != "none" and not instr
        if redundant and instr:
            # the instructions are in the part a person can also see: a visible injection, not a
            # hidden one (still reported, at the visible-text level)
            sem["instruction_like_visible_copy"] = True
            instr = False
        sem["instruction_like"] = instr
        if instr and ctx in ("hidden", "offpage"):
            # instructions where nobody looks (hidden, metadata, annotations...) are the attack itself
            f["severity"] = "high"
        elif sem.get("embedding_only") and not redundant:
            # reads like instructions, but only one signal says so: review, not an attack verdict
            f["severity"] = "high" if tier == "strong" else max(
                f["severity"], "medium", key=lambda s: ["info", "low", "medium", "high"].index(s))
            f["review"] = True
        f["semantic"] = sem


def verdict(result: dict) -> tuple[str, list[str]]:
    if result.get("status") == "quarantine":
        return "QUARANTINE", [f"quarantine: {result.get('quarantine_reason')}"]
    reasons: list[tuple[str, str]] = []
    for f in result["findings"]:
        sem = f.get("semantic") or {}
        instr = sem.get("instruction_like", False)
        mech = set(f["mechanism"])
        cat = f["category"]
        benign = f.get("benign")
        tag = labels.short(f)
        if "honeypot_tool_attempt" in mech:
            reasons.append(("MALICIOUS", "honeypot tool attempt: " + tag))
        elif sem.get("instruction_like_visible_copy"):
            reasons.append(("SUSPICIOUS", "instruction-like text that is also visible on the page: " + tag))
        elif instr and context_of(f) in ("hidden", "offpage"):
            # benign matches (metadata field, minor/short text, ...) never excuse instructions
            reasons.append(("MALICIOUS", "instruction-like hidden/unreviewed text: " + tag))
        elif instr and cat == "visible_text":
            reasons.append(("SUSPICIOUS", "instruction-like visible text: " + tag))
        elif sem.get("embedding_only") and context_of(f) in ("hidden", "offpage") \
                and not (benign or "").startswith("redundant"):
            # one signal only (the embedding): enough for a human to look, not for MALICIOUS
            reasons.append(("SUSPICIOUS", "reads like instructions to an AI (embedding signal): " + tag))
        elif cat in ("hidden_text", "mismatched_text") and not benign:
            reasons.append(("SUSPICIOUS", "hidden/mismatched text: " + tag))
        elif benign:
            continue
        elif cat == "char_level" and f["severity"] in ("medium", "high"):
            reasons.append(("SUSPICIOUS", "character-level trick: " + tag))
        elif cat == "vision":
            reasons.append(("SUSPICIOUS", "faint text in render: " + tag))
        elif mech & {"javascript", "launch_action", "document_additional_actions"} or (
                cat == "action" and (f["severity"] == "high"
                                     or ("open_action" in mech and f["severity"] == "medium"))):
            # code/launch anywhere, or anything risky that runs automatically on open
            reasons.append(("SUSPICIOUS", "active content: " + tag))
        elif mech & (STRUCT_TEXT_MECH - {"clip_only_render_mode"}):
            reasons.append(("SUSPICIOUS", "structural hiding: " + tag))
        elif "attachment_not_scanned" in mech or "nesting_limit_reached" in mech:
            reasons.append(("QUARANTINE", "attachment could not be scanned: " + tag))
        elif "parser_repair" in mech and f["detail"].get("heavy"):
            reasons.append(("QUARANTINE", "heavy parser repair: " + tag))
        elif "judge_anomaly" in mech:
            reasons.append(("SUSPICIOUS", "LLM judge anomaly: " + tag))
    for child in result.get("children", []):
        cv, cr = verdict(child)
        if cv != "CLEAN":
            reasons.append((cv, f"embedded {child.get('file')}: {cv}"))
    # (no global "OCR unavailable" rule: OCR is a safety net. Findings that needed OCR to be cleared
    # simply stay unresolved and count above.)
    if not reasons:
        return "CLEAN", ["all checks ran and found nothing"]
    top = max(reasons, key=lambda r: RANK[r[0]])[0]
    ordered = sorted(reasons, key=lambda r: -RANK[r[0]])
    seen, out = set(), []
    for v, r in ordered:
        if r not in seen:
            seen.add(r)
            out.append(f"[{v}] {r}")
    return top, out
