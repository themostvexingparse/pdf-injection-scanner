"""Plain-language names for finding categories and mechanisms (used by reports and verdict reasons)."""
from __future__ import annotations

CATEGORY = {
    "hidden_text": "Hidden text",
    "mismatched_text": "Text that extracts differently than it looks",
    "offpage_content": "Text outside the page body",
    "structural": "Structural anomaly",
    "action": "Active content / link",
    "attachment": "Attachment",
    "extractor_disagreement": "Extractors disagree",
    "char_level": "Character-level trick",
    "vision": "Faint text in rendered page",
    "visible_text": "Visible instruction-like text",
}

MECHANISM = {
    # visibility measurement (decides)
    "no_rendered_contribution": "draws nothing visible on the page",
    "outside_rendered_page": "lies outside the visible page area",
    "imperceptible_contrast": "contrast against its background is imperceptible",
    "low_contrast": "very low contrast against its background",
    "illegibly_small": "rendered too small to read",
    "very_small_relative_to_body_text": "much smaller than the page's body text",
    "no_visible_pixels": "changes no pixels when rendered",
    "low_pixel_contrast": "renders with very low contrast",
    "render_engines_disagree": "rendering engines disagree on whether it is visible",
    "overdrawn_by_other_text": "drawn underneath other text",
    "pdfium_only": "only the pdfium extractor returns it",
    "not_perceptible": "not perceptible",
    # explanations (drawing state)
    "invisible_render_mode": "uses an invisible text render mode",
    "zero_opacity": "has zero opacity",
    "low_opacity": "has very low opacity",
    "collapsed_metrics": "has collapsed glyph metrics",
    "tiny_font_size": "uses a microscopic font size",
    "outside_cropbox": "is placed outside the visible page box",
    "between_cropbox_and_mediabox": "sits between the crop box and media box",
    "hidden_layer": "is on a layer that is off by default",
    "stacked_glyphs": "glyphs are stacked on top of each other",
    "covered_by_later_shape": "is covered by a shape drawn afterwards",
    "covered_by_later_image": "is covered by an image drawn afterwards",
    "fill_color_matches_background": "fill colour matches the background",
    "fill_color_low_contrast": "fill colour is close to the background",
    "clip_only_render_mode": "uses clip-only render mode (never painted)",
    "agent_only_text_unexplained": "is returned by extractors but is not visible in the render",
    "document_has_layers_off_by_default": "document has layers that are off by default",
    "not_meaningful": "extracted text does not read as language",
    "invisible_text_novel_vs_render": "invisible text says something the visible page does not",
    # mismatches
    "cmap_remap": "font character map is remapped",
    "extracted_differs_from_rendered": "extracted characters differ from what is rendered",
    "unextractable_text": "text cannot be extracted meaningfully (encoding issue, not hidden)",
    "multi_char_glyph_mapping": "a single glyph extracts as a long string",
    "actual_text_override": "replacement text (ActualText) overrides what is drawn",
    "actual_text_not_visible": "replacement text is not what the page shows",
    # off-page content
    "metadata_info": "document metadata",
    "metadata_xmp": "XMP metadata",
    "annotation_hidden_flag": "annotation flagged hidden",
    "annotation_off_page": "annotation placed off the page",
    "form_field": "form field value / label",
    "outline_titles": "bookmark titles",
    "page_labels": "page labels",
    "alt_text": "alternative text",
    "actual_text_dict": "replacement text in structure",
    "abbreviation_expansion": "abbreviation expansion",
    "field_tooltip": "form field tooltip",
    "attachment_text": "text of an attached file",
    # structural / actions / attachments
    "data_after_eof": "data after the end-of-file marker",
    "data_before_header": "data before the PDF header",
    "missing_eof": "missing end-of-file marker",
    "incremental_updates": "file has incremental updates (multiple revisions)",
    "earlier_revision_text_differs": "an earlier revision contains different text",
    "orphaned_content_stream": "unreferenced content stream with text",
    "parser_repair": "parsers had to repair the file",
    "layer_off_by_default": "layer is off by default",
    "layer_print_only": "layer only shows when printed",
    "layer_screen_only": "layer only shows on screen",
    "encrypted_opened_with_empty_password": "encrypted (opened with empty password)",
    "extractor_page_count_mismatch": "extractors disagree on page count",
    "judge_anomaly": "LLM judge behaved anomalously",
    "javascript": "JavaScript",
    "launch_action": "launches an external program",
    "submit_form_action": "submits form data",
    "import_data_action": "imports data",
    "remote_goto_action": "opens another document",
    "embedded_goto_action": "opens an embedded document",
    "rich_media": "rich media",
    "uri_action": "opens a URL",
    "link_uri": "link to a URL",
    "open_action": "does something automatically when the file is opened",
    "open_action_goto": "opens at a given page (navigation only, harmless)",
    "open_action_javascript": "runs JavaScript when opened",
    "open_action_launch": "launches a file or program when opened",
    "open_action_uri": "opens a website when opened",
    "open_action_submitform": "submits form data when opened",
    "open_action_importdata": "imports data when opened",
    "open_action_gotor": "opens another document when opened",
    "open_action_gotoe": "opens an embedded document when opened",
    "open_action_named": "runs a viewer command when opened",
    "action_chain": "chains several actions (/Next)",
    "action_loop": "action chain loops back on itself",
    "uri_without_scheme": "link target has no scheme",
    "metadata_custom_field": "non-standard metadata field",
    "document_additional_actions": "document-level triggered actions",
    "embedded_file": "embedded file",
    "file_attachment_annotation": "file attachment annotation",
    "pdf_portfolio": "PDF portfolio",
    "attachment_not_scanned": "attachment could not be scanned",
    "nesting_limit_reached": "nested attachments too deep to scan",
    # character level / vision
    "unicode_tag_characters": "invisible Unicode tag characters",
    "zero_width_characters": "zero-width characters",
    "bidi_controls": "bidirectional override characters",
    "variation_selectors": "variation selectors",
    "mixed_script_homoglyphs": "words mixing look-alike letters from different scripts",
    "compatibility_characters": "look-alike compatibility characters",
    "base64_payload": "base64-encoded text",
    "hex_payload": "hex-encoded text",
    "faint_text_in_render": "readable only after contrast enhancement",
    "parser_differential": "different extractors return different words",
    "visible_instruction_like": "visible text that reads like instructions to an AI",
    "visible_reads_like_instructions": "visible sentence that may be addressed to an AI (embedding, for review)",
    "visible_body_text": "visible body text",
}

SEMANTIC = {
    "override_previous_instructions": "override previous instructions",
    "new_instructions": "new/hidden instructions",
    "addresses_ai": "addresses an AI",
    "role_reassignment": "role reassignment",
    "prompt_markup": "prompt markup",
    "concealment": "asks to conceal",
    "exfiltration": "data exfiltration",
    "secrets": "mentions secrets",
    "tool_invocation": "tool invocation",
    "shell_command": "shell command",
    "evaluation_manipulation": "evaluation manipulation",
    "output_control": "output control",
    "second_person_imperative": "commands the reader",
}


DETECTOR = {
    "leave_one_out": "per-object render measurement",
    "mupdf_fallback": "all-text render diff (fallback)",
    "glyph_tracer": "glyph tracer",
    "pixel_diff": "pixel diff",
    "font_cmap_check": "font character-map check",
    "pymupdf_trace": "glyph trace",
    "ocr": "OCR",
    "ocr_enhanced": "contrast-enhanced OCR",
    "safe_text": "visible text",
}


def detector(d: str) -> str:
    return DETECTOR.get(d, d)


def mech(m: str) -> str:
    if m.startswith("layer_") and m not in MECHANISM:
        return m.replace("_", " ")
    if m.startswith("annotation_") and m not in MECHANISM:
        return m[len("annotation_"):].replace("_", " ") + " annotation"
    return MECHANISM.get(m, m.replace("_", " "))


def category(c: str) -> str:
    return CATEGORY.get(c, c.replace("_", " "))


def where(f: dict) -> str:
    if f.get("page"):
        return f"page {f['page']}"
    return f.get("source") or "document"


def short(f: dict, n: int = 2) -> str:
    """One-line description: 'Hidden text (draws nothing visible; ...) on page 3'."""
    ms = "; ".join(mech(m) for m in f["mechanism"][:n])
    return f"{category(f['category'])} ({ms}) at {where(f)}"
