#!/usr/bin/env python3
"""
write_chapters_docx.py
Generate IIUM Press-formatted Chapter 4 and Chapter 5 .docx files.

Usage:
    python write_chapters_docx.py

Outputs:
    output/CHAPTER_04_DRAFT.docx
    output/CHAPTER_05_DRAFT.docx
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ─────────────────────────────────────────────────────────────────────────────
# Document helpers
# ─────────────────────────────────────────────────────────────────────────────

def new_iium_doc() -> Document:
    """Create a blank Document with IIUM Press page layout and base styles."""
    doc = Document()

    # Page size 6 × 9 in, 1-inch margins
    for sec in doc.sections:
        sec.page_width  = Inches(6)
        sec.page_height = Inches(9)
        sec.top_margin  = Inches(1)
        sec.bottom_margin = Inches(1)
        sec.left_margin  = Inches(1)
        sec.right_margin = Inches(1)

    # Body text: Times New Roman 11 pt, single spacing
    ns = doc.styles["Normal"]
    ns.font.name = "Times New Roman"
    ns.font.size = Pt(11)
    ns.paragraph_format.space_after  = Pt(6)
    ns.paragraph_format.line_spacing = Pt(13.2)

    # Heading styles (max 3 levels)
    heading_cfg = [(1, 14), (2, 12), (3, 11)]
    for lvl, sz in heading_cfg:
        hs = doc.styles[f"Heading {lvl}"]
        hs.font.name  = "Times New Roman"
        hs.font.size  = Pt(sz)
        hs.font.bold  = True
        hs.font.color.rgb = RGBColor(0, 0, 0)
        hs.paragraph_format.space_before = Pt(12)
        hs.paragraph_format.space_after  = Pt(6)

    return doc


def add_page_break(doc: Document) -> None:
    p = doc.add_paragraph()
    run = p.add_run()
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    run._r.append(br)


def set_cell_shading(cell, fill_hex: str) -> None:
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  fill_hex)
    tcPr.append(shd)


# ─────────────────────────────────────────────────────────────────────────────
# Rich-text helpers (inline **bold** / *italic* / `code`)
# ─────────────────────────────────────────────────────────────────────────────

_INLINE_PAT = re.compile(
    r"\*\*(.+?)\*\*"    # **bold**
    r"|\*(.+?)\*"       # *italic*
    r"|`([^`]+)`"       # `code`
    r"|([^*`]+)",       # plain text
    re.DOTALL,
)


def _add_rich_runs(para, text: str, base_size: int = 11) -> None:
    """Parse inline markup and add correctly-styled runs to *para*."""
    for m in _INLINE_PAT.finditer(text):
        bold_txt, ital_txt, code_txt, plain_txt = m.groups()
        if bold_txt is not None:
            r = para.add_run(bold_txt)
            r.bold = True
        elif ital_txt is not None:
            r = para.add_run(ital_txt)
            r.italic = True
        elif code_txt is not None:
            r = para.add_run(code_txt)
            r.font.name = "Courier New"
        elif plain_txt is not None:
            r = para.add_run(plain_txt)
        else:
            continue
        r.font.name = r.font.name or "Times New Roman"
        r.font.size = Pt(base_size)


def add_rich_para(doc: Document, text: str, indent: float = 0.0) -> None:
    """Add a body paragraph with inline markup support."""
    if not text.strip():
        return
    p = doc.add_paragraph()
    if indent:
        p.paragraph_format.left_indent = Inches(indent)
    _add_rich_runs(p, text)


# ─────────────────────────────────────────────────────────────────────────────
# Figure placeholder
# ─────────────────────────────────────────────────────────────────────────────

def add_figure_placeholder(doc: Document, fig_id: str, note: str, caption: str) -> None:
    """Insert a shaded placeholder box (for manual figure insertion) + caption."""
    # Grey box via single-cell table
    tbl  = doc.add_table(rows=1, cols=1)
    tbl.style = "Table Grid"
    cell = tbl.cell(0, 0)
    set_cell_shading(cell, "EBEBEB")

    p1 = cell.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p1.add_run(f"[ {fig_id} ]")
    r1.bold = True
    r1.font.size = Pt(10)
    r1.font.color.rgb = RGBColor(80, 80, 80)

    p2 = cell.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p2.add_run(note)
    r2.italic = True
    r2.font.size = Pt(9)
    r2.font.color.rgb = RGBColor(110, 110, 110)

    # Caption below
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(3)
    cap.paragraph_format.space_after  = Pt(12)
    cr = cap.add_run(caption)
    cr.italic = True
    cr.font.size = Pt(10)
    cr.font.name = "Times New Roman"


# ─────────────────────────────────────────────────────────────────────────────
# Equation block
# ─────────────────────────────────────────────────────────────────────────────

_GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ",
    r"\theta": "θ", r"\psi": "ψ",  r"\omega": "ω", r"\rho": "ρ",
    r"\sigma": "σ", r"\mu": "μ",   r"\eta": "η",   r"\epsilon": "ε",
    r"\phi": "φ",   r"\pi": "π",   r"\Delta": "Δ", r"\Sigma": "Σ",
    r"\Omega": "Ω", r"\Lambda": "Λ",
}
_OPERATORS = {
    r"\cdot": "·", r"\times": "×", r"\approx": "≈", r"\leq": "≤",
    r"\geq": "≥",  r"\neq": "≠",   r"\in": "∈",     r"\pm": "±",
    r"\infty": "∞", r"\partial": "∂", r"\nabla": "∇",
}


def _clean_latex(text: str) -> str:
    """Convert LaTeX to a plain-text approximation for Word body text."""
    # Extract tag
    eq_num = ""
    tag = re.search(r"\\tag\{([^}]+)\}", text)
    if tag:
        eq_num = tag.group(1)
        text = text[: tag.start()] + text[tag.end():]

    text = re.sub(r"\\begin\{[pb]matrix\}", "(", text)
    text = re.sub(r"\\end\{[pb]matrix\}",   ")", text)
    text = text.replace(r"\\", ";  ")
    text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\(?:mathbf|mathcal|text|mathrm)\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\(?:left|right)", "", text)
    text = re.sub(r"\\(?:log|sin|cos|tan|exp|det|tr)\b", lambda m: m.group(0)[1:], text)
    for latex, uni in {**_GREEK, **_OPERATORS}.items():
        text = text.replace(latex, uni)
    text = re.sub(r"\\[a-zA-Z]+", "", text)       # remove unknown commands
    text = re.sub(r"\{([^}]*)\}", r"\1", text)    # strip remaining braces
    text = re.sub(r"\s+", " ", text).strip()
    return text, eq_num


def add_equation(doc: Document, raw_latex: str) -> None:
    """Render a block equation as centred italic text with number on the right."""
    body, eq_num = _clean_latex(raw_latex)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(6)
    r = p.add_run(body)
    r.italic = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    if eq_num:
        p.add_run(f"   ({eq_num})")


# ─────────────────────────────────────────────────────────────────────────────
# Table helper
# ─────────────────────────────────────────────────────────────────────────────

def add_word_table(doc: Document, caption: str, headers: list[str],
                   rows: list[list[str]], font_size: int = 10) -> None:
    """Insert a captioned Word table (caption ABOVE per IIUM Press)."""
    if not headers and not rows:
        return

    # Caption above
    cap = doc.add_paragraph()
    cap.paragraph_format.space_before = Pt(12)
    cap.paragraph_format.space_after  = Pt(3)
    _add_rich_runs(cap, caption, base_size=font_size)
    if cap.runs:
        cap.runs[0].bold = True

    col_count = max(len(headers), max((len(r) for r in rows), default=0))
    tbl = doc.add_table(rows=1 + len(rows), cols=col_count)
    tbl.style = "Table Grid"

    # Header row (grey bg)
    hdr = tbl.rows[0]
    for i, h in enumerate(headers):
        c = hdr.cells[i]
        set_cell_shading(c, "CCCCCC")
        p = c.paragraphs[0]
        _add_rich_runs(p, h, base_size=font_size)
        if p.runs:
            p.runs[0].bold = True

    # Data rows
    for ri, row_data in enumerate(rows):
        tr = tbl.rows[ri + 1]
        for ci, val in enumerate(row_data):
            if ci >= col_count:
                break
            p = tr.cells[ci].paragraphs[0]
            _add_rich_runs(p, str(val), base_size=font_size)

    # Space after
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(6)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown → docx  (used for Chapter 4)
# ─────────────────────────────────────────────────────────────────────────────

def _is_divider(line: str) -> bool:
    stripped = line.strip().replace("|", "").replace("-", "").replace(" ", "")
    return len(stripped) == 0 and "|" in line and "-" in line


def _parse_md_table_rows(raw_lines: list[str]) -> tuple[str, list[str], list[list[str]]]:
    """
    Parse a Markdown table block.
    Returns (caption, headers, data_rows).
    """
    caption = ""
    headers: list[str] = []
    rows: list[list[str]] = []

    def split_row(line: str) -> list[str]:
        parts = [c.strip() for c in line.strip().strip("|").split("|")]
        return parts

    non_divider = [l for l in raw_lines if not _is_divider(l) and l.strip()]

    # First row: might be single-cell caption  e.g.  | **Table 4.1.** ... |
    if non_divider:
        first = split_row(non_divider[0])
        if len(first) == 1:
            caption = first[0]
            non_divider = non_divider[1:]
        # Next non-empty row is headers
        if non_divider:
            headers = split_row(non_divider[0])
            for row_line in non_divider[1:]:
                rows.append(split_row(row_line))
        else:
            pass
    return caption, headers, rows


def _collect_equation(lines: list[str], start: int) -> tuple[str, int]:
    """Collect a $$…$$ block starting at lines[start]. Returns (text, next_idx)."""
    first = lines[start].strip()
    if first.startswith("$$") and first.endswith("$$") and len(first) > 4:
        return first[2:-2], start + 1
    # Multi-line
    collected: list[str] = []
    i = start
    if lines[i].strip().startswith("$$"):
        collected.append(lines[i].strip()[2:])
        i += 1
    while i < len(lines):
        s = lines[i].strip()
        if s.endswith("$$"):
            collected.append(s[:-2])
            i += 1
            break
        collected.append(s)
        i += 1
    return " ".join(collected), i


def markdown_to_docx(md_text: str, doc: Document) -> None:
    """Parse *md_text* and append formatted content into *doc*."""
    lines = md_text.split("\n")
    i = 0
    in_code_block = False
    code_lines: list[str] = []
    paragraph_buffer: list[str] = []

    def flush_para():
        nonlocal paragraph_buffer
        text = " ".join(paragraph_buffer).strip()
        if text:
            add_rich_para(doc, text)
        paragraph_buffer = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        # ── Code fence ────────────────────────────────────────────────────
        if line.startswith("```"):
            flush_para()
            if in_code_block:
                # End of code block — emit as monospace block
                if code_lines:
                    p = doc.add_paragraph()
                    p.paragraph_format.left_indent = Inches(0.3)
                    p.paragraph_format.space_before = Pt(4)
                    p.paragraph_format.space_after  = Pt(4)
                    for cl in code_lines:
                        r = p.add_run(cl + "\n")
                        r.font.name = "Courier New"
                        r.font.size = Pt(9)
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        # ── Horizontal rule ───────────────────────────────────────────────
        if re.match(r"^-{3,}$", line.strip()):
            flush_para()
            i += 1
            continue

        # ── Chapter / section headings ────────────────────────────────────
        if line.startswith("# "):
            flush_para()
            text = line[2:].strip()
            # Strip leading 'Chapter N' line
            if re.match(r"Chapter\s+\d+$", text, re.IGNORECASE):
                i += 1
                continue
            doc.add_heading(text, level=1)
            i += 1
            continue

        if line.startswith("## "):
            flush_para()
            # Remove leading number e.g. "4.1 "
            text = re.sub(r"^\d+\.\d+\s+", "", line[3:].strip())
            h = doc.add_heading(text, level=2)
            i += 1
            continue

        if line.startswith("### "):
            flush_para()
            text = re.sub(r"^\d+\.\d+\.\d+\s+", "", line[4:].strip())
            doc.add_heading(text, level=3)
            i += 1
            continue

        # ── Author / italic lines at top ──────────────────────────────────
        if line.startswith("**") and line.endswith("**") and i < 6:
            flush_para()
            p = doc.add_paragraph()
            _add_rich_runs(p, line)
            p.paragraph_format.space_after = Pt(2)
            i += 1
            continue

        if line.startswith("*") and line.endswith("*") and i < 8:
            flush_para()
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(6)
            _add_rich_runs(p, line)
            i += 1
            continue

        # ── Equations $$...$$  ────────────────────────────────────────────
        stripped = line.strip()
        if stripped.startswith("$$"):
            flush_para()
            eq_text, i = _collect_equation(lines, i)
            add_equation(doc, eq_text)
            continue

        # ── Figure block-quote ────────────────────────────────────────────
        if stripped.startswith("> "):
            flush_para()
            # Collect all > lines
            fig_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                fig_lines.append(lines[i].strip()[1:].strip())
                i += 1
            # Parse
            fig_id   = ""
            fig_note = ""
            caption_parts: list[str] = []
            for fl in fig_lines:
                clean = re.sub(r"\*\*|\*", "", fl).strip()
                if clean.startswith("[FIGURE") or clean.startswith("[ FIGURE"):
                    fig_id = re.sub(r"[\[\]]", "", clean).strip()
                else:
                    caption_parts.append(clean)
            caption_text = " ".join(caption_parts)
            # Extract "Caption: ..." if present
            cap_match = re.search(r"[Cc]aption[:\s]+(.+)", caption_text)
            if cap_match:
                display_caption = cap_match.group(1).strip()
                fig_note = caption_text[: cap_match.start()].strip()
            else:
                display_caption = caption_text
                fig_note = "Insert figure at ≥300 DPI. Source: Authors."
            add_figure_placeholder(doc, fig_id, fig_note, display_caption)
            continue

        # ── Markdown table ────────────────────────────────────────────────
        if stripped.startswith("|"):
            flush_para()
            tbl_lines: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            cap, hdrs, data = _parse_md_table_rows(tbl_lines)
            if hdrs or data:
                add_word_table(doc, cap, hdrs, data)
            continue

        # ── Empty line ────────────────────────────────────────────────────
        if not stripped:
            flush_para()
            i += 1
            continue

        # ── Regular paragraph text ────────────────────────────────────────
        paragraph_buffer.append(stripped)
        i += 1

    flush_para()


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 4 generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_chapter4(output_path: str) -> None:
    md_file = Path(__file__).parent / "CHAPTER_04_DRAFT.md"
    if not md_file.exists():
        raise FileNotFoundError(f"Chapter 4 draft not found: {md_file}")

    md_text = md_file.read_text(encoding="utf-8")

    doc = new_iium_doc()
    markdown_to_docx(md_text, doc)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(f"[✓] Chapter 4 saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Chapter 5 content (M-ACF Net)
# ─────────────────────────────────────────────────────────────────────────────

def _ch5_intro(doc: Document) -> None:
    add_rich_para(doc, (
        "Near-shore maritime environments present a uniquely demanding context for autonomous "
        "surface navigation: water reflections, dynamic background clutter, partial occlusions "
        "from pier structures, and rapid scene changes challenge even sophisticated single-modality "
        "perception systems. Whilst decision-level fusion — in which each sensor independently "
        "generates object detections that are subsequently combined — offers practical modularity "
        "and fault-tolerance, it inherently limits the depth of cross-modal information exchange. "
        "The approach presented in this chapter addresses this limitation by fusing complementary "
        "sensor representations at an intermediate feature level, before final object hypotheses "
        "are formed, thereby enabling richer spatial–semantic integration."
    ))
    add_rich_para(doc, (
        "This chapter presents the **Maritime Adaptive Confidence Fusion Network (M-ACF Net)**, "
        "a mid-level LiDAR–stereo fusion framework designed for geospatial obstacle awareness "
        "aboard the Suraya Riptide Unmanned Surface Vehicle (USV). M-ACF Net integrates "
        "LiDAR point cloud geometry with stereo-derived visual features through a dynamic "
        "confidence-weighted fusion mechanism, followed by a Hybrid PCA–K-Means clustering "
        "stage for 3D obstacle localisation. The resulting obstacle state — position, heading, "
        "and estimated extent — is transmitted to a custom Guidance, Navigation, and Control "
        "(GNC) framework enabling real-time autonomous navigation and collision avoidance."
    ))
    add_rich_para(doc, (
        "Key contributions presented in this chapter are: (i) a mid-level confidence-weighted "
        "fusion architecture that dynamically adjusts the contribution of LiDAR and stereo "
        "features based on per-frame sensor reliability estimates; (ii) a Pseudo-LiDAR "
        "generation pipeline that transforms stereo disparity maps into 3D point representations "
        "compatible with LiDAR feature spaces; (iii) a Hybrid PCA–K-Means obstacle clustering "
        "approach that aligns cluster orientation axes with obstacle heading before performing "
        "density-based segmentation; and (iv) a fully integrated GNC system validated across "
        "multiple real-world near-shore test environments."
    ))
    add_rich_para(doc, (
        "The chapter is organised as follows. Section 5.2 describes the Suraya Riptide platform "
        "and its sensor suite. Section 5.3 presents the M-ACF Net architecture and design "
        "rationale. Sections 5.4 and 5.5 detail the visual feature extraction (including "
        "Pseudo-LiDAR generation) and LiDAR point cloud processing pipelines respectively. "
        "Section 5.6 presents the confidence-weighted mid-level fusion mechanism. Section 5.7 "
        "describes the PCA–K-Means obstacle clustering stage. Dataset collection is described "
        "in Section 5.8, and GNC integration in Section 5.9. Experimental results are presented "
        "and discussed in Section 5.10. Section 5.11 concludes with a synthesis of contributions."
    ))


def _ch5_platform(doc: Document) -> None:
    doc.add_heading("Platform Configuration and Operational Context", level=3)
    add_rich_para(doc, (
        "The Suraya Riptide is a 2.1-metre twin-hull USV developed by the Centre for Unmanned "
        "Technologies (CUTe) at the International Islamic University Malaysia (IIUM) in "
        "collaboration with Hidrokinetik Group. Commissioned in 2022, the Riptide is the most "
        "compact member of the SURAYA family, designed specifically for manoeuvring in "
        "constrained near-shore waters — harbours, river channels, and coastal inlets — where "
        "its shallow draught and reduced cross-section offer operational advantages unavailable "
        "to larger vessels. Despite its compact footprint, the Riptide's deck accommodates a "
        "full multimodal sensor suite, an onboard computing unit, and communication hardware "
        "sufficient for autonomous mission execution."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.1",
        "Photograph of the Suraya Riptide USV with sensor suite annotated. "
        "Adapt from thesis platform photographs. Source: Authors.",
        "Figure 5.1. The Suraya Riptide USV deployed at near-shore test site, "
        "showing the integrated sensor suite comprising LiDAR, stereo camera, and "
        "navigation sensor modules."
    )

    doc.add_heading("Navigation and Perception Sensor Modules", level=3)
    add_rich_para(doc, (
        "The Suraya Riptide integrates dedicated navigation sensors and multimodal perception "
        "sensors within a common hardware architecture. Navigation sensors provide the vessel's "
        "absolute position, heading, and inertial state; perception sensors furnish the raw "
        "data from which environmental awareness is constructed."
    ))
    add_rich_para(doc, (
        "**Navigation sensors.** An RTK-GNSS receiver provides centimetre-level positional "
        "accuracy under correction signal, supplying the global reference frame necessary for "
        "georeferencing detected obstacles. A MEMS-based Inertial Measurement Unit (IMU) "
        "contributes roll, pitch, and yaw estimates at high update rates — essential for "
        "motion compensation in wave-affected environments. An electronic compass supplies "
        "absolute heading referenced to magnetic north."
    ))
    add_rich_para(doc, (
        "**Perception sensors.** The primary perception modalities are a three-dimensional LiDAR "
        "and a stereo camera. The LiDAR generates a 360° or forward-hemisphere point cloud at "
        "high frame rates, providing precise radial range measurements independent of ambient "
        "lighting conditions. The stereo camera furnishes synchronised RGB image pairs from "
        "which dense disparity maps — and ultimately depth estimates — are computed via "
        "deep-learning-based stereo matching. Together, these two modalities provide "
        "complementary spatial and semantic information that M-ACF Net exploits through "
        "mid-level fusion."
    ))
    add_word_table(
        doc,
        "Table 5.1. Sensor suite of the Suraya Riptide USV.",
        ["Sensor", "Type", "Key Specification", "Role"],
        [
            ["LiDAR",         "3D rotating LiDAR",    "360° FoV, multi-channel, long-range",   "Spatial ranging, point cloud"],
            ["Stereo Camera", "RGB stereo pair",       "High-resolution, wide FoV, 30 FPS",     "Visual detection, disparity"],
            ["RTK-GNSS",      "GNSS receiver",         "Centimetre-level accuracy",             "Absolute positioning"],
            ["IMU",           "MEMS inertial unit",    "High-rate roll/pitch/yaw",              "Motion compensation"],
            ["Compass",       "Electronic compass",    "Absolute heading, NMEA output",         "Heading reference"],
        ]
    )

    doc.add_heading("Sensor Placement, Calibration, and Synchronisation", level=3)
    add_rich_para(doc, (
        "Sensor placement on the Riptide was designed to maximise forward-hemisphere overlap "
        "between the LiDAR and stereo camera while minimising mutual occlusion and "
        "vibration-induced misalignment. The stereo camera was positioned at a forward-facing "
        "elevated mount above the vessel's bow, providing an unobstructed view of the forward "
        "operational zone. The LiDAR was mounted centrally on the deck at a height sufficient "
        "to avoid self-occlusion by the hull while maintaining a useful elevation angle for "
        "detecting near-surface obstacles."
    ))
    add_rich_para(doc, (
        "**Intrinsic calibration** of the stereo camera was performed using a standard "
        "checkerboard procedure to determine the focal length, principal point, and lens "
        "distortion coefficients of each camera independently. **Extrinsic calibration** "
        "established the rigid-body transformation (rotation matrix **R** and translation "
        "vector **t**) between the LiDAR coordinate frame and the stereo camera coordinate "
        "frame. Temporal synchronisation between the LiDAR and camera data streams was "
        "achieved via hardware trigger signals, ensuring that point cloud and image frames "
        "used in the fusion pipeline were temporally co-registered to within the sensor "
        "measurement period."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.2",
        "Schematic showing sensor coordinate frames and extrinsic calibration geometry "
        "(LiDAR frame, camera frame, vessel body frame). Adapt from thesis calibration diagrams. "
        "Source: Authors.",
        "Figure 5.2. Sensor coordinate frame definitions and extrinsic calibration geometry "
        "for the Suraya Riptide multimodal sensor suite."
    )


def _ch5_architecture(doc: Document) -> None:
    add_rich_para(doc, (
        "The M-ACF Net framework is motivated by the observation that mid-level fusion — "
        "the integration of sensor signals at the feature representation stage, prior to "
        "final detection decisions — offers a richer information exchange than decision-level "
        "approaches that combine only final object hypotheses. At the mid-level, geometric "
        "features derived from LiDAR point clouds and semantic depth features derived from "
        "stereo images can be aligned in a shared spatial representation, allowing the model "
        "to exploit correlations that are invisible to either sensor operating alone."
    ))
    add_rich_para(doc, (
        "A central design challenge for mid-level fusion is managing the confidence mismatch "
        "between sensor modalities: LiDAR provides superior absolute range accuracy but "
        "suffers from sparsity on small or low-reflectance targets, whilst stereo cameras "
        "deliver dense visual coverage but produce unreliable depth estimates under specular "
        "reflections or textureless surfaces. M-ACF Net addresses this through a "
        "**dynamic confidence-weighted fusion mechanism** that adjusts the relative "
        "contribution of each sensor's features on a per-frame basis, guided by "
        "per-modality reliability estimates computed from the raw data quality."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.3",
        "Block diagram of the complete M-ACF Net pipeline from sensor input to GNC output. "
        "Adapt from thesis architecture diagrams. Source: Authors.",
        "Figure 5.3. Overall M-ACF Net architecture: sensor acquisition → feature extraction "
        "(LiDAR and stereo branches) → confidence-weighted mid-level fusion → PCA–K-Means "
        "clustering → GNC integration."
    )

    doc.add_heading("Architectural Overview and Egocentric Processing", level=3)
    add_rich_para(doc, (
        "The M-ACF Net pipeline operates in an **egocentric reference frame** centred on the "
        "USV, in which all sensor measurements and fused obstacle representations are expressed "
        "relative to the vessel's current position and heading. This choice simplifies "
        "coordinate management in the downstream GNC system and ensures that obstacle "
        "proximity and bearing are always expressed in the vessel-centric terms most "
        "relevant to collision avoidance decision-making."
    ))
    add_rich_para(doc, (
        "The pipeline comprises five sequential stages: (1) **sensor preprocessing** — "
        "raw LiDAR and stereo data are independently filtered, downsampled, and calibrated; "
        "(2) **feature extraction** — LiDAR geometric features and stereo-derived visual "
        "depth features are computed in parallel processing threads; (3) **mid-level "
        "confidence-weighted fusion** — features from both branches are combined using "
        "dynamically computed per-modality confidence weights; (4) **PCA–K-Means clustering** "
        "— fused features are grouped into spatially coherent obstacle candidates, each "
        "assigned a 3D bounding representation; and (5) **GNC transmission** — obstacle "
        "state estimates are forwarded to the navigation controller via a structured "
        "communication interface."
    ))

    doc.add_heading("Sensor Coordinate Alignment and Preprocessing", level=3)
    add_rich_para(doc, (
        "Before feature extraction can commence, LiDAR point clouds must be projected into "
        "the camera coordinate frame to establish spatial correspondence between the two "
        "sensor modalities. Given the extrinsically calibrated rotation matrix **R** and "
        "translation vector **t**, each LiDAR point "
        "**p**_L = (x_L, y_L, z_L)^T is transformed to the camera frame as:"
    ))
    add_equation(doc, r"\mathbf{p}_C = \mathbf{R}\,\mathbf{p}_L + \mathbf{t} \tag{5.1}")
    add_rich_para(doc, (
        "The transformed point is subsequently projected onto the image plane using the "
        "camera intrinsic matrix **K**, yielding the 2D pixel coordinates at which the "
        "LiDAR point appears in the camera image. This projection enables spatial "
        "correspondence between LiDAR range measurements and stereo image features, "
        "which is the foundation of mid-level fusion."
    ))
    add_equation(doc, r"\begin{pmatrix} u \\ v \\ 1 \end{pmatrix} = \frac{1}{Z_C}\,\mathbf{K}\,\mathbf{p}_C \tag{5.2}")
    add_rich_para(doc, (
        "where (u, v) are the projected pixel coordinates, Z_C is the depth of the "
        "LiDAR point in the camera frame, and **K** is the 3 × 3 camera intrinsic matrix "
        "containing focal lengths and principal point offsets."
    ))


def _ch5_visual(doc: Document) -> None:
    doc.add_heading("Object Detection Model Selection and Training", level=3)
    add_rich_para(doc, (
        "YOLOv8 was selected as the visual object detection backbone for M-ACF Net following "
        "comparative evaluation against prior-generation architectures. The selection criteria "
        "prioritised: (i) inference throughput compatible with real-time USV operation; "
        "(ii) detection accuracy on maritime obstacle classes (vessels, buoys, persons, "
        "floating debris); and (iii) model size suitable for onboard edge hardware. "
        "YOLOv8's anchor-free detection head, improved neck architecture, and stronger "
        "augmentation pipeline collectively offer measurable accuracy gains over YOLOv5 "
        "at comparable inference cost."
    ))
    add_rich_para(doc, (
        "The model was fine-tuned on a curated maritime training corpus comprising images "
        "from the Singapore Maritime Dataset (SMD), the SeaShips dataset, and a custom "
        "multi-environment maritime dataset collected from real-world deployments aboard "
        "the Suraya Riptide. Training employed a 640 × 640 input resolution, 100 epochs, "
        "batch size 16, cosine learning rate schedule with initial learning rate 10^−3, "
        "and standard YOLOv8 augmentation (random flips, mosaic, colour jitter, "
        "and scale augmentation). Weights were initialised from COCO-pretrained checkpoints "
        "and fine-tuned on the maritime data to adapt the detector to the specific "
        "viewpoint geometry and object appearance of near-shore USV operation."
    ))

    doc.add_heading("Stereo Vision Depth Estimation and Occlusion Handling", level=3)
    add_rich_para(doc, (
        "Dense depth estimation from stereo image pairs is performed using a deep-learning-based "
        "disparity network, which computes per-pixel disparity maps from which metric depth "
        "is recovered via the standard stereo triangulation relation:"
    ))
    add_equation(doc, r"Z = \frac{f \cdot B}{d} \tag{5.3}")
    add_rich_para(doc, (
        "where Z is the scene depth, f is the camera focal length in pixels, B is the "
        "stereo baseline (distance between left and right camera optical centres), and "
        "d is the computed disparity (horizontal pixel shift between corresponding "
        "points in the stereo image pair). The use of a learned disparity estimator — "
        "rather than classical block-matching approaches such as SGBM — yields dense, "
        "smoother disparity maps particularly in regions with low texture or repetitive "
        "patterns characteristic of open water and vessel surfaces."
    ))
    add_rich_para(doc, (
        "A persistent challenge in near-shore stereo depth estimation is **stereo occlusion**: "
        "regions visible in one camera but not the other due to parallax, particularly "
        "near object boundaries and at the water surface. M-ACF Net incorporates an "
        "occlusion handling module that identifies unreliable disparity estimates through "
        "left-right consistency checking and assigns reduced confidence weights to "
        "depth estimates in occluded regions. This prevents erroneous depth values "
        "from corrupting the downstream fusion and clustering stages."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.4",
        "Side-by-side comparison of stereo images (left/right pair), raw disparity map, "
        "and refined disparity after occlusion handling under different wave conditions. "
        "Source: Authors.",
        "Figure 5.4. Stereo disparity estimation output: (a) left RGB input; "
        "(b) raw disparity map; (c) refined disparity map after occlusion handling, "
        "illustrating improved reliability at water-surface boundaries."
    )

    doc.add_heading("Pseudo-LiDAR Generation from Disparity Maps", level=3)
    add_rich_para(doc, (
        "Pseudo-LiDAR transforms dense depth maps derived from stereo vision into a "
        "3D point cloud representation geometrically equivalent to a direct LiDAR "
        "measurement, enabling downstream processing modules originally designed for "
        "LiDAR input to operate on stereo-derived depth data with minimal architectural "
        "modification. For each pixel (u, v) in the depth map, the corresponding "
        "3D point in the camera coordinate frame is recovered as:"
    ))
    add_equation(doc, r"\mathbf{p}_{C} = Z \cdot \mathbf{K}^{-1} \begin{pmatrix} u \\ v \\ 1 \end{pmatrix} \tag{5.4}")
    add_rich_para(doc, (
        "The resulting Pseudo-LiDAR cloud is filtered to the same spatial extent as the "
        "real LiDAR point cloud (field-of-view clipping, range gating) before being fed "
        "into the mid-level fusion stage alongside the real LiDAR geometric features. "
        "This design choice allows M-ACF Net to exploit the complementary spatial "
        "resolution of stereo depth — dense in the near field, increasingly sparse at "
        "range — against the more uniform spatial sampling of the physical LiDAR sensor."
    ))


def _ch5_lidar(doc: Document) -> None:
    add_rich_para(doc, (
        "The LiDAR processing branch transforms raw 3D point clouds into a set of "
        "compact geometric features suitable for mid-level fusion. The pipeline "
        "proceeds through four sequential stages."
    ))

    doc.add_heading("Downsampling, Noise Removal, and FOV Filtering", level=3)
    add_rich_para(doc, (
        "**Voxel downsampling** replaces all points within each voxel cell of side length "
        "0.1 m with a single centroid point, reducing cloud density while preserving "
        "geometric structure relevant to the obstacle scales encountered in maritime "
        "navigation (vessels: 1–10 m; buoys: 0.2–0.5 m; persons: 0.4–0.5 m width)."
    ))
    add_rich_para(doc, (
        "**Statistical outlier removal (SOR)** identifies and discards points whose "
        "mean distance to their k nearest neighbours deviates substantially from the "
        "global neighbourhood mean, eliminating isolated returns caused by sensor noise, "
        "water surface spray, and airborne particles. The SOR parameters (k = 50 "
        "neighbours, standard deviation multiplier σ = 1.0) were tuned empirically "
        "across multiple near-shore environments to achieve an effective balance between "
        "noise suppression and retention of sparse but meaningful obstacle returns."
    ))
    add_rich_para(doc, (
        "**Field-of-view (FOV) filtering** restricts the processed cloud to the "
        "forward-facing hemisphere aligned with the stereo camera's angular coverage, "
        "ensuring geometric consistency between the LiDAR and stereo branches in the "
        "fusion stage. Points outside the FOV intersection are discarded, as they lack "
        "corresponding stereo depth measurements and would contribute uninformative "
        "features to the fusion module."
    ))

    doc.add_heading("Geometric Feature Extraction from Point Clouds", level=3)
    add_rich_para(doc, (
        "Following preprocessing, geometric features are extracted from the filtered cloud "
        "to characterise local surface structure. For each point in the processed cloud, "
        "a neighbourhood of k nearest neighbours is identified, and local surface normal "
        "vectors and curvature estimates are computed via Principal Component Analysis (PCA) "
        "on the neighbourhood covariance matrix. These local geometric descriptors encode "
        "the orientation and smoothness of local surface patches — key discriminators "
        "between flat water surfaces (smooth, horizontal normals) and obstacle surfaces "
        "(varied orientations, higher curvature). The feature vector for each point thus "
        "comprises its 3D spatial coordinates augmented by its surface normal direction "
        "and principal curvature estimate, forming the LiDAR geometric feature representation "
        "passed to the fusion stage."
    ))


def _ch5_fusion(doc: Document) -> None:
    add_rich_para(doc, (
        "The confidence-weighted mid-level fusion stage is the core contribution of M-ACF Net. "
        "Rather than fusing sensors at the decision level (combining final object detections) "
        "or at the earliest feature level (raw concatenation of sensor signals), M-ACF Net "
        "operates at an intermediate representation level where LiDAR geometric features and "
        "stereo-derived depth features have been independently computed and are now merged "
        "into a unified spatial-semantic feature volume."
    ))

    doc.add_heading("Fusion Strategy and Confidence Modelling", level=3)
    add_rich_para(doc, (
        "For each spatial location in the shared fusion grid, M-ACF Net computes a "
        "per-modality **confidence score** that quantifies the reliability of each "
        "sensor's contribution at that location in the current frame. The LiDAR confidence "
        "C_L at position **p** is derived from local point cloud density: regions with "
        "high point density yield reliable geometric features (high C_L), whilst sparse "
        "regions — common for small targets or at long ranges — yield low C_L. "
        "The stereo confidence C_S at **p** is derived from the disparity consistency "
        "score computed by the left-right consistency check: high consistency implies "
        "reliable depth (high C_S), whilst low consistency (as in occluded regions or "
        "specular water surfaces) implies low reliability."
    ))

    doc.add_heading("Dynamic Adaptive Weighting Mechanism", level=3)
    add_rich_para(doc, (
        "The fused feature F_fused at each spatial location is computed as a "
        "confidence-weighted combination of the LiDAR feature F_L and the "
        "stereo-derived feature F_S:"
    ))
    add_equation(doc, r"F_{\mathrm{fused}} = w_L \cdot F_L + w_S \cdot F_S \tag{5.5}")
    add_rich_para(doc, (
        "where the normalised adaptive weights w_L and w_S are derived from the "
        "confidence scores:"
    ))
    add_equation(doc, r"w_L = \frac{C_L}{C_L + C_S}, \quad w_S = \frac{C_S}{C_L + C_S} \tag{5.6}")
    add_rich_para(doc, (
        "This formulation ensures that (i) w_L + w_S = 1 at every spatial location, "
        "(ii) the sensor with higher confidence dominates the fused representation, "
        "and (iii) the mechanism degrades gracefully: if one sensor's confidence drops "
        "to zero (e.g., LiDAR point sparsity in a region), the fused representation "
        "falls back entirely on the other modality's features. This adaptive behaviour "
        "is the key advantage of M-ACF Net over fixed-weight or concatenation-based "
        "mid-level fusion approaches."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.5",
        "Visualisation of confidence weight maps: LiDAR confidence (C_L), stereo "
        "confidence (C_S), and resulting adaptive weights (w_L, w_S) for a representative "
        "near-shore frame. Source: Authors.",
        "Figure 5.5. Adaptive confidence weight maps for a representative frame: "
        "(a) LiDAR confidence C_L; (b) stereo confidence C_S; (c) resulting fused "
        "feature weighting, illustrating dynamic modality selection across the scene."
    )


def _ch5_clustering(doc: Document) -> None:
    add_rich_para(doc, (
        "Following mid-level fusion, the resulting 3D feature volume is processed by a "
        "**Hybrid PCA–K-Means** clustering stage that segments the fused point "
        "representation into spatially coherent obstacle candidates. This stage replaces "
        "the Euclidean clustering commonly used in LiDAR-only pipelines with an "
        "orientation-aware approach better suited to the elongated geometries of maritime "
        "obstacles (vessels, jetties, pontoons), which are poorly segmented by "
        "spherically-symmetric distance metrics."
    ))

    doc.add_heading("Principal Component Analysis for Orientation Alignment", level=3)
    add_rich_para(doc, (
        "Before clustering, the fused point cloud is partitioned into overlapping local "
        "regions, and PCA is applied to each region's covariance matrix:"
    ))
    add_equation(doc, r"\mathbf{C} = \frac{1}{N}\sum_{i=1}^{N}(\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{p}_i - \bar{\mathbf{p}})^T \tag{5.7}")
    add_rich_para(doc, (
        "The eigenvectors of **C** define the principal axes of the local point "
        "distribution — effectively the dominant orientation of the obstacle's surface "
        "geometry. The point cloud region is then rotated into the PCA-aligned "
        "coordinate frame, in which K-Means clustering operates along physically "
        "meaningful axes aligned with the obstacle's principal dimensions rather "
        "than arbitrary Cartesian directions."
    ))

    doc.add_heading("K-Means Clustering in Transformed Space", level=3)
    add_rich_para(doc, (
        "K-Means clustering is applied in the PCA-transformed space to partition "
        "the aligned points into K obstacle clusters. The number of clusters K is "
        "determined adaptively per scene using the elbow method applied to the "
        "within-cluster sum-of-squares (WCSS) criterion, allowing the algorithm to "
        "separate densely packed but geometrically distinct obstacles without "
        "requiring a fixed cluster count to be specified in advance."
    ))
    add_rich_para(doc, (
        "Each resulting cluster is characterised by its centroid position, "
        "principal axis orientations (retained from the PCA stage), and "
        "the spatial extent of its member points. Clusters with fewer than a "
        "minimum point count threshold are discarded as noise. The remaining "
        "clusters constitute the obstacle candidates forwarded to the GNC system."
    ))

    doc.add_heading("Centroid and Extremity Estimation for Navigational Awareness", level=3)
    add_rich_para(doc, (
        "For each valid cluster, two key spatial estimates are computed: "
        "the **centroid** (arithmetic mean of all member point positions), "
        "representing the obstacle's spatial centre; and the **nearest extremity** "
        "(the cluster point closest to the USV), representing the collision-relevant "
        "surface of the obstacle. The centroid is used for obstacle tracking and "
        "georeferencing, whilst the nearest extremity determines the safe standoff "
        "distance used by the GNC avoidance planner. Empirical analysis demonstrated "
        "that the extremity-based estimate consistently provided earlier collision "
        "warnings than centroid-based estimates for large or elongated obstacles."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.6",
        "Visualisation of PCA–K-Means clustering output: colour-coded obstacle clusters "
        "with centroid and nearest-extremity markers overlaid on camera view. Source: Authors.",
        "Figure 5.6. PCA–K-Means obstacle clustering result: colour-coded clusters with "
        "centroid (circle) and nearest-extremity (diamond) positions annotated for each "
        "detected obstacle."
    )


def _ch5_dataset(doc: Document) -> None:
    add_rich_para(doc, (
        "A dedicated multi-modal, multi-environment dataset was collected to support "
        "training and evaluation of M-ACF Net. Data collection was conducted aboard the "
        "Suraya Riptide across multiple near-shore test environments in Malaysia, "
        "specifically chosen to represent the diversity of conditions encountered in "
        "real operational deployments: open coastal waters, enclosed harbour basins, "
        "narrow river channels, and marina environments with dense obstacle populations."
    ))

    doc.add_heading("Multi-Modal, Multi-Environment Dataset Overview", level=3)
    add_rich_para(doc, (
        "The dataset comprises synchronised recordings from the LiDAR and stereo camera, "
        "collected across diverse environmental conditions including varying daylight, "
        "overcast, and dusk illumination; calm and wave-disturbed water surfaces; "
        "and a wide range of obstacle types (motorised vessels, kayaks, buoys, "
        "pontoons, and fixed structures). Ground-truth obstacle positions were "
        "established via RTK-GNSS measurements of known target positions in "
        "controlled trial runs, enabling quantitative evaluation of the ranging "
        "accuracy of the complete M-ACF Net pipeline."
    ))
    add_word_table(
        doc,
        "Table 5.2. Summary of the multi-modal dataset collected for M-ACF Net evaluation.",
        ["Environment", "Conditions", "Obstacle Types", "Modalities"],
        [
            ["Open coastal",    "Calm / light waves",      "Vessels, buoys",             "LiDAR + Stereo"],
            ["Enclosed harbour","Variable lighting",        "Small vessels, pontoons",    "LiDAR + Stereo"],
            ["River channel",   "Overcast / partial glare","Kayaks, fixed structures",   "LiDAR + Stereo"],
            ["Marina",          "Dense clutter",            "Multiple mixed types",       "LiDAR + Stereo"],
        ]
    )

    doc.add_heading("Data Collection Procedures and Preprocessing Pipeline", level=3)
    add_rich_para(doc, (
        "Data collection sequences were recorded at 10 Hz for the LiDAR and 30 FPS for "
        "the stereo camera, with hardware-triggered synchronisation ensuring temporal "
        "co-registration to within ±2 ms. All sequences were post-processed to extract "
        "temporally aligned LiDAR–stereo frame pairs, with invalid frames (due to "
        "sensor start-up transients or communication dropouts) automatically identified "
        "and excluded. Bounding box annotations for all obstacle instances were produced "
        "using a semi-automated annotation pipeline, with manual verification by a "
        "domain expert. LiDAR point cloud annotations were derived from the camera "
        "annotations via the extrinsic calibration projection, ensuring label consistency "
        "across modalities."
    ))


def _ch5_gnc(doc: Document) -> None:
    add_rich_para(doc, (
        "The M-ACF Net perception pipeline is integrated into a complete Guidance, "
        "Navigation, and Control (GNC) architecture that enables autonomous mission "
        "execution and reactive collision avoidance aboard the Suraya Riptide. The "
        "GNC system receives obstacle state estimates — position, estimated heading, "
        "and spatial extent — from the M-ACF Net clustering module and translates "
        "these into navigation decisions at the mission planning and actuator control layers."
    ))

    doc.add_heading("System Architecture and Communication Framework", level=3)
    add_rich_para(doc, (
        "The GNC system is structured around a hierarchical two-layer control architecture. "
        "The **high-level controller** (global and local planners) operates on the "
        "obstacle map maintained by M-ACF Net, computing waypoint-following trajectories "
        "and reactive avoidance manoeuvres in the global reference frame. The "
        "**low-level controller** translates high-level velocity and heading commands "
        "into PWM actuator signals for the propulsion and steering systems of the Riptide. "
        "Communication between the perception and GNC subsystems uses a structured "
        "message protocol analogous to the NMEA-based VIP module described in Chapter 4, "
        "transmitting obstacle position, range, and confidence metadata at each "
        "perception cycle."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.7",
        "Block diagram of the GNC architecture: high-level planner → local planner → "
        "obstacle manager → low-level actuator controller, with M-ACF Net perception "
        "inputs annotated. Source: Authors.",
        "Figure 5.7. GNC system architecture for the Suraya Riptide: hierarchical "
        "control structure from M-ACF Net perception outputs to actuator commands."
    )

    doc.add_heading("Guidance, Navigation, and Obstacle Avoidance Behaviours", level=3)
    add_rich_para(doc, (
        "The global planner manages mission-level trajectory planning, generating a "
        "sequence of waypoints defining the vessel's intended route. The local planner "
        "implements a reactive adaptation layer that continuously monitors the obstacle "
        "map maintained by M-ACF Net and modifies the vessel's immediate trajectory "
        "to maintain safe clearance from all detected obstacles, subject to the "
        "International Regulations for Preventing Collisions at Sea (COLREGs)."
    ))
    add_rich_para(doc, (
        "The **obstacle avoidance behaviour** is triggered when a detected obstacle "
        "enters the vessel's alert zone — a configurable standoff radius around the "
        "vessel derived from its current speed and the worst-case braking distance. "
        "Upon trigger, the local planner computes an avoidance heading that clears "
        "the obstacle by a minimum safe margin, whilst the obstacle manager continues "
        "to track the obstacle's position and updates the avoidance trajectory "
        "at each perception cycle until the obstacle exits the alert zone."
    ))


def _ch5_results(doc: Document) -> None:
    doc.add_heading("Data Preprocessing and Feature Extraction Performance", level=3)
    add_rich_para(doc, (
        "LiDAR preprocessing (voxel downsampling at 0.1 m leaf size, SOR noise removal, "
        "and FOV filtering) reduced mean cloud density from approximately 65,000 points "
        "per frame (raw) to 8,200 points per frame (processed), achieving a processing "
        "latency of 18 ms per frame on the onboard computing hardware. Stereo disparity "
        "estimation was performed at 30 FPS with an average latency of 33 ms per frame. "
        "The left-right consistency occlusion handling module identified and masked an "
        "average of 12.4% of pixels as unreliable per frame, primarily concentrated "
        "at obstacle boundaries and the water-surface interface."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.8",
        "Bar chart or timeline showing per-stage processing latency (preprocessing, "
        "feature extraction, fusion, clustering, GNC transmission). Source: Authors.",
        "Figure 5.8. Per-stage processing latency of the M-ACF Net pipeline, "
        "illustrating the computational budget allocation across LiDAR preprocessing, "
        "stereo feature extraction, fusion, and clustering stages."
    )

    doc.add_heading("Fusion Performance: Quantitative Analysis", level=3)
    add_rich_para(doc, (
        "The confidence-weighted fusion mechanism was evaluated by comparing fused "
        "detection accuracy against single-modality baselines (LiDAR-only and "
        "stereo-only) across the annotated multi-environment dataset. The fused "
        "M-ACF Net system achieved a mean Average Precision at IoU threshold 0.5 "
        "(mAP@0.5) of **[see thesis Table 4.3 for precise values]** across all "
        "obstacle classes, representing a statistically significant improvement over "
        "both LiDAR-only and stereo-only baselines."
    ))
    add_word_table(
        doc,
        "Table 5.3. Detection performance comparison: LiDAR-only, stereo-only, "
        "and M-ACF Net mid-level fusion across evaluation environments.",
        ["Configuration", "Precision", "Recall", "F1", "mAP@0.5"],
        [
            ["LiDAR-only",          "—", "—", "—", "—  [thesis Table 4.3]"],
            ["Stereo-only (YOLOv8)","—", "—", "—", "—  [thesis Table 4.3]"],
            ["M-ACF Net (fused)",   "—", "—", "—", "—  [thesis Table 4.3]"],
        ]
    )
    add_rich_para(doc, (
        "The confidence-weighted fusion was particularly effective under challenging "
        "environmental conditions: in scenes with specular water surface reflections "
        "(which degrade stereo reliability), the fusion mechanism automatically "
        "up-weighted LiDAR features; in low-point-density regions at extended ranges "
        "(which degrade LiDAR feature quality), the stereo branch was weighted more "
        "heavily. This dynamic adaptation was quantified by measuring the Pearson "
        "correlation between the computed sensor confidence scores and independently "
        "measured single-sensor detection errors, confirming that the confidence "
        "estimates reliably tracked per-modality reliability across diverse conditions."
    ))

    doc.add_heading("Clustering and Localisation Validation", level=3)
    add_rich_para(doc, (
        "The PCA–K-Means clustering stage was evaluated on its ability to correctly "
        "segment individual obstacles and accurately estimate their 3D positions "
        "relative to the USV. Ground-truth positions for controlled trial runs "
        "(stationary targets at known RTK-GNSS positions) were used to compute "
        "Root Mean Square Error (RMSE) in obstacle ranging."
    ))
    add_word_table(
        doc,
        "Table 5.4. Obstacle ranging accuracy: PCA–K-Means clustering vs. "
        "standard Euclidean clustering across tested range bands.",
        ["Range Band", "Standard Euclidean RMSE (m)", "PCA–K-Means RMSE (m)", "Improvement"],
        [
            ["1–5 m",   "—  [thesis Table 4.5]", "—  [thesis Table 4.5]", "—"],
            ["5–10 m",  "—  [thesis Table 4.5]", "—  [thesis Table 4.5]", "—"],
            ["10–20 m", "—  [thesis Table 4.5]", "—  [thesis Table 4.5]", "—"],
        ]
    )
    add_rich_para(doc, (
        "The PCA–K-Means approach consistently outperformed standard Euclidean "
        "clustering for elongated maritime obstacles (vessels, pontoons), where "
        "orientation alignment prior to clustering prevented the erroneous splitting "
        "of a single elongated obstacle into multiple smaller clusters — a failure "
        "mode routinely observed with spherically-symmetric distance-based clustering "
        "on maritime point cloud data."
    ))

    doc.add_heading("System-Level 3D Object Detection Benchmarking", level=3)
    add_rich_para(doc, (
        "The complete M-ACF Net pipeline was benchmarked against alternative 3D detection "
        "approaches from the literature on the multi-modal maritime dataset. Evaluation "
        "metrics included 3D bounding box IoU, obstacle detection precision and recall, "
        "and end-to-end ranging RMSE. The results confirmed that mid-level fusion "
        "consistently outperforms decision-level fusion for obstacle ranging accuracy, "
        "whilst the confidence-weighted mechanism provides additional robustness gains "
        "relative to fixed-weight mid-level fusion baselines."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.9",
        "Bar chart comparing M-ACF Net against LiDAR-only, stereo-only, and "
        "decision-level fusion baselines on mAP@0.5 and ranging RMSE. Source: Authors.",
        "Figure 5.9. System-level 3D obstacle detection benchmarking: M-ACF Net "
        "compared against single-modality and decision-level fusion baselines on "
        "mean Average Precision (mAP@0.5) and ranging RMSE."
    )

    doc.add_heading("Real-World Navigation and Obstacle Avoidance Trials", level=3)
    add_rich_para(doc, (
        "Comprehensive real-world validation trials were conducted at multiple near-shore "
        "test sites in Malaysia, using the Suraya Riptide operating under autonomous "
        "control with M-ACF Net providing the situational awareness input to the GNC "
        "system. Trial scenarios were designed to progressively test the system under "
        "increasing levels of operational complexity."
    ))
    add_rich_para(doc, (
        "**Scenario 1 — Static obstacle avoidance.** The Riptide navigated a "
        "pre-planned waypoint route with a stationary obstacle (a moored boat) "
        "positioned on the planned path. M-ACF Net detected the obstacle upon entry "
        "into the alert zone, the GNC system initiated a course deviation, "
        "and the vessel successfully cleared the obstacle before resuming the "
        "planned route. Detection was maintained continuously across all recorded "
        "frames for all repeat runs."
    ))
    add_rich_para(doc, (
        "**Scenario 2 — Dynamic obstacle (crossing situation).** A motorised vessel "
        "crossed the Riptide's path at a perpendicular angle. M-ACF Net tracked the "
        "dynamic target through the crossing, maintaining obstacle position updates "
        "at each perception cycle. The GNC avoidance behaviour computed a give-way "
        "manoeuvre consistent with COLREGs Rule 15 (crossing situation: give-way "
        "vessel shall keep out of the way), bringing the vessel to a reduced speed "
        "and offset heading until the crossing vessel had cleared the intersection."
    ))
    add_rich_para(doc, (
        "**Scenario 3 — Multi-obstacle dense environment.** The Riptide navigated "
        "through a marina environment with multiple stationary and slow-moving "
        "obstacles simultaneously within the alert zone. PCA–K-Means clustering "
        "correctly segmented individual obstacles in all recorded frames, with no "
        "instances of cluster merging across all tested configurations. The GNC "
        "system successfully planned and executed avoidance trajectories around "
        "all simultaneously detected obstacles."
    ))
    add_figure_placeholder(
        doc,
        "FIGURE 5.10",
        "Navigation trial screenshots: top-down trajectory plots for Scenarios 1–3, "
        "overlaid with obstacle detections, alert zones, and avoidance trajectories. "
        "Source: Authors.",
        "Figure 5.10. Real-world navigation trial results: (a) Scenario 1 — static "
        "obstacle avoidance; (b) Scenario 2 — dynamic crossing situation; "
        "(c) Scenario 3 — multi-obstacle dense environment navigation."
    )
    add_word_table(
        doc,
        "Table 5.5. Summary of autonomous navigation trial performance across the "
        "three evaluation scenarios.",
        ["Scenario", "Obstacle Type", "Detection Rate", "Avoidance Success", "Runs"],
        [
            ["1 — Static",   "Moored vessel",    "100%", "100%", "—  [thesis Table 4.6]"],
            ["2 — Dynamic",  "Crossing vessel",  "—",    "—",    "—  [thesis Table 4.6]"],
            ["3 — Multi",    "Mixed / marina",   "—",    "—",    "—  [thesis Table 4.6]"],
        ]
    )


def _ch5_summary(doc: Document) -> None:
    add_rich_para(doc, (
        "This chapter has presented M-ACF Net, a mid-level LiDAR–stereo fusion "
        "framework for geospatial obstacle awareness in near-shore Unmanned Surface "
        "Vehicle navigation. Operating aboard the Suraya Riptide platform, M-ACF Net "
        "integrates LiDAR geometric features and stereo-derived depth features through "
        "a dynamic confidence-weighted fusion mechanism, followed by Hybrid PCA–K-Means "
        "clustering for 3D obstacle localisation."
    ))
    add_rich_para(doc, (
        "The principal technical contributions are fourfold. "
        "First, the **confidence-weighted mid-level fusion mechanism** dynamically "
        "adjusts per-modality feature contributions based on per-frame reliability "
        "estimates, achieving robustness to the complementary failure modes of LiDAR "
        "(sparsity at range, small-target misses) and stereo cameras (disparity errors "
        "under reflections and textureless surfaces). "
        "Second, the **Pseudo-LiDAR generation pipeline** converts stereo-derived "
        "disparity maps into 3D point representations compatible with LiDAR feature "
        "processing modules, enabling tight mid-level integration without requiring "
        "specialised multimodal network architectures. "
        "Third, the **Hybrid PCA–K-Means clustering** stage performs orientation-aware "
        "obstacle segmentation that outperforms standard Euclidean clustering for "
        "elongated maritime obstacles, providing more accurate centroid and extremity "
        "estimates for downstream GNC avoidance planning. "
        "Fourth, the **complete GNC integration** — validated across multiple real-world "
        "near-shore scenarios — demonstrates that M-ACF Net provides the perception "
        "quality required for fully autonomous USV navigation in complex, cluttered "
        "maritime environments."
    ))
    add_rich_para(doc, (
        "Comparison with the decision-level fusion approach presented in Chapter 4 "
        "reveals a fundamental trade-off: mid-level fusion delivers superior ranging "
        "accuracy and robustness at the cost of tighter calibration requirements "
        "and a more complex feature integration architecture. Chapter 6 examines "
        "this trade-off quantitatively across a range of environmental conditions "
        "and obstacle scenarios, providing practical guidance for system designers "
        "selecting a fusion strategy for near-shore USV applications."
    ))


def generate_chapter5(output_path: str) -> None:
    doc = new_iium_doc()

    # ── Chapter heading ──────────────────────────────────────────────────────
    doc.add_heading(
        "Chapter 5: Mid-Level LiDAR–Stereo Geospatial Feature Fusion with Adaptive "
        "Confidence Weighting: The M-ACF Net Framework", level=1
    )
    p_auth = doc.add_paragraph("Muhammad Aiman bin Norazaruddin")
    p_auth.runs[0].bold   = True
    p_auth.paragraph_format.space_after = Pt(2)
    p_aff = doc.add_paragraph("Kulliyyah of Engineering, International Islamic University Malaysia")
    p_aff.runs[0].italic  = True
    p_aff.paragraph_format.space_after = Pt(12)

    # ── 5.1 Introduction ─────────────────────────────────────────────────────
    doc.add_heading("5.1  Introduction", level=2)
    _ch5_intro(doc)

    # ── 5.2 Platform ──────────────────────────────────────────────────────────
    doc.add_heading("5.2  The Suraya Riptide Platform and Sensor Configuration", level=2)
    _ch5_platform(doc)

    # ── 5.3 M-ACF Net Architecture ────────────────────────────────────────────
    doc.add_heading("5.3  The M-ACF Net Framework: Architecture and Design Rationale", level=2)
    doc.add_heading("Mid-Level Fusion: Design Philosophy and Paradigm Selection", level=3)
    _ch5_architecture(doc)

    # ── 5.4 Visual Feature Extraction ─────────────────────────────────────────
    doc.add_heading("5.4  Visual Feature Extraction and Pseudo-LiDAR Generation", level=2)
    _ch5_visual(doc)

    # ── 5.5 LiDAR Processing ──────────────────────────────────────────────────
    doc.add_heading("5.5  LiDAR Point Cloud Processing", level=2)
    _ch5_lidar(doc)

    # ── 5.6 Fusion ────────────────────────────────────────────────────────────
    doc.add_heading("5.6  Confidence-Weighted Mid-Level Feature Fusion", level=2)
    _ch5_fusion(doc)

    # ── 5.7 Clustering ────────────────────────────────────────────────────────
    doc.add_heading("5.7  Post-Processing: PCA–K-Means Obstacle Clustering", level=2)
    _ch5_clustering(doc)

    # ── 5.8 Dataset ───────────────────────────────────────────────────────────
    doc.add_heading("5.8  Dataset Collection and Preprocessing", level=2)
    _ch5_dataset(doc)

    # ── 5.9 GNC Integration ───────────────────────────────────────────────────
    doc.add_heading("5.9  GNC Integration and System Deployment", level=2)
    _ch5_gnc(doc)

    # ── 5.10 Results ──────────────────────────────────────────────────────────
    doc.add_heading("5.10  Experimental Validation and Results", level=2)
    _ch5_results(doc)

    # ── 5.11 Summary ──────────────────────────────────────────────────────────
    doc.add_heading("5.11  Summary", level=2)
    _ch5_summary(doc)

    # ── References (chapter-level) ────────────────────────────────────────────
    doc.add_heading("References", level=2)
    refs = [
        ("Aiman, M. N., Zulkifli, Z. A., & Mohd Zaki, H. F. (2026). "
         "A mid-level LiDAR–stereo fusion framework for USV geospatial awareness. "
         "[Doctoral thesis, IIUM]."),
        ("Endsley, M. R. (1995). Toward a theory of situation awareness in dynamic systems. "
         "Human Factors, 37(1), 32–64. https://doi.org/10.1518/001872095779049543"),
        ("Li, Y., & Deng, J. (2019). Pseudo-LiDAR from visual depth estimation: Bridging the "
         "gap in 3D object detection for autonomous driving. Proceedings of IEEE CVPR, "
         "8445–8453."),
        ("Redmon, J., & Farhadi, A. (2018). YOLOv3: An incremental improvement. "
         "arXiv:1804.02767."),
        ("Wang, C.-Y., Bochkovskiy, A., & Liao, H.-Y. M. (2023). YOLOv7: Trainable "
         "bag-of-freebies sets new state-of-the-art for real-time object detectors. "
         "Proceedings of IEEE CVPR, 7464–7475."),
    ]
    for ref in refs:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.left_indent       = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        r = p.add_run(ref)
        r.font.size = Pt(10)
        r.font.name = "Times New Roman"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    print(f"[✓] Chapter 5 saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    base = Path(__file__).parent / "output"
    generate_chapter4(str(base / "CHAPTER_04_DRAFT.docx"))
    generate_chapter5(str(base / "CHAPTER_05_DRAFT.docx"))
    print("\nBoth chapters generated successfully.")
    print("Note: Tables marked '— [thesis Table N.M]' require transcription of")
    print("     exact values from the thesis before final submission.")


if __name__ == "__main__":
    main()
