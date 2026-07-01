#!/usr/bin/env python3
"""Create a visually checkable PDF from the generated manuscript markdown."""

from __future__ import annotations

import html
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
PAPER_DIR = ROOT / "paper"
INPUT = PAPER_DIR / "manuscript_results_filled.md"
OUTPUT = PAPER_DIR / "manuscript_results_filled.pdf"


def clean_inline(text: str) -> str:
    text = html.escape(text.strip())
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    return text


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        parts = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if parts and not all(set(cell) <= {"-", ":"} for cell in parts):
            rows.append(parts)
    return rows


def table_widths(rows: list[list[str]], total_width: float) -> list[float]:
    cols = max(len(row) for row in rows)
    weights = []
    for idx in range(cols):
        max_len = max((len(row[idx]) if idx < len(row) else 0) for row in rows)
        weights.append(max(8, min(max_len, 34)))
    scale = total_width / sum(weights)
    return [w * scale for w in weights]


def make_table(rows: list[list[str]], doc_width: float):
    widths = table_widths(rows, doc_width)
    cols = len(widths)
    font_size = 5.2 if cols >= 8 else 6.5 if cols >= 6 else 8
    leading = font_size + 1.4
    style = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=font_size,
        leading=leading,
        spaceAfter=0,
    )
    data = []
    for row in rows:
        data.append([Paragraph(clean_inline(row[idx] if idx < len(row) else ""), style) for idx in range(cols)])
    table = LongTable(data, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0B2545")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D7DE")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def add_image(story, alt: str, path_text: str, doc_width: float, styles) -> None:
    path = Path(path_text)
    if not path.is_absolute():
        path = (PAPER_DIR / path).resolve()
    if not path.exists():
        return
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(doc_width / width, 4.1 * inch / height)
    story.append(
        KeepTogether(
            [
                Image(str(path), width=width * scale, height=height * scale),
                Paragraph(clean_inline(alt), styles["Caption"]),
                Spacer(1, 0.12 * inch),
            ]
        )
    )


def build_pdf() -> None:
    if not INPUT.exists():
        raise SystemExit(f"Missing {INPUT}; run build_paper_from_results.py first.")

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.72 * inch,
    )
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "PaperTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=19,
            textColor=colors.HexColor("#0B2545"),
            spaceAfter=14,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#2E74B5"),
            spaceBefore=14,
            spaceAfter=7,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=13,
            textColor=colors.HexColor("#1F4D78"),
            spaceBefore=10,
            spaceAfter=5,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            spaceAfter=6,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12,
            leftIndent=14,
            firstLineIndent=-8,
            spaceAfter=4,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            alignment=1,
            textColor=colors.HexColor("#555555"),
            spaceAfter=6,
        ),
    }

    story = []
    lines = INPUT.read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(make_table(parse_table(table_lines), doc.width))
            story.append(Spacer(1, 0.12 * inch))
            continue
        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            add_image(story, image_match.group(1), image_match.group(2), doc.width, styles)
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(clean_inline(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(clean_inline(line[3:]), styles["H1"]))
        elif line.startswith("### "):
            story.append(Paragraph(clean_inline(line[4:]), styles["H2"]))
        elif line.startswith("- "):
            story.append(Paragraph("&#8226; " + clean_inline(line[2:]), styles["Bullet"]))
        elif re.match(r"^\d+\.\s+", line):
            story.append(Paragraph(clean_inline(line), styles["Body"]))
        else:
            story.append(Paragraph(clean_inline(line), styles["Body"]))
        i += 1

    doc.build(story)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build_pdf()
