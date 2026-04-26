"""PDF report generation for Agora research runs."""
from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

CREAM   = colors.HexColor("#f5f1eb")
AMBER   = colors.HexColor("#d97706")
TEXT_1  = colors.HexColor("#2a2118")
TEXT_2  = colors.HexColor("#6b5c45")
TEXT_3  = colors.HexColor("#a89278")
GREEN   = colors.HexColor("#059669")
RED     = colors.HexColor("#dc2626")
BLUE    = colors.HexColor("#2563eb")
SURFACE = colors.HexColor("#faf8f5")
BORDER  = colors.HexColor("#e8dfd4")


def _styles():
    return {
        "title": ParagraphStyle(
            "title", fontName="Helvetica-Bold", fontSize=22,
            textColor=TEXT_1, spaceAfter=4, leading=28,
        ),
        "meta": ParagraphStyle(
            "meta", fontName="Helvetica", fontSize=9,
            textColor=TEXT_3, spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section", fontName="Helvetica-Bold", fontSize=10,
            textColor=AMBER, spaceBefore=16, spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "body", fontName="Helvetica", fontSize=10,
            textColor=TEXT_1, leading=16, spaceAfter=6,
        ),
        "body_italic": ParagraphStyle(
            "body_italic", fontName="Helvetica-Oblique", fontSize=10,
            textColor=TEXT_2, leading=16, spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small", fontName="Helvetica", fontSize=9,
            textColor=TEXT_3, leading=13, spaceAfter=2,
        ),
        "small_bold": ParagraphStyle(
            "small_bold", fontName="Helvetica-Bold", fontSize=9,
            textColor=TEXT_2, leading=13, spaceAfter=2,
        ),
        "subq_label": ParagraphStyle(
            "subq_label", fontName="Helvetica-Bold", fontSize=9,
            textColor=TEXT_3, spaceAfter=3,
        ),
        "subq_text": ParagraphStyle(
            "subq_text", fontName="Helvetica", fontSize=10,
            textColor=TEXT_1, leading=15, spaceAfter=4,
        ),
        "answer": ParagraphStyle(
            "answer", fontName="Helvetica", fontSize=11,
            textColor=TEXT_1, leading=18, spaceAfter=10,
        ),
        "cite_url": ParagraphStyle(
            "cite_url", fontName="Helvetica", fontSize=9,
            textColor=BLUE, leading=13, spaceAfter=2,
        ),
        "cite_quote": ParagraphStyle(
            "cite_quote", fontName="Helvetica-Oblique", fontSize=9,
            textColor=TEXT_2, leading=13, spaceAfter=4,
        ),
    }


def _divider(color=BORDER):
    return HRFlowable(width="100%", thickness=0.5, color=color, spaceAfter=8, spaceBefore=4)


def _coverage_badge(coverage: str) -> str:
    if coverage == "well-supported":
        return "OK"
    if coverage == "failed":
        return "X"
    return "~"


def _coverage_color(coverage: str):
    if coverage == "well-supported":
        return GREEN
    if coverage == "failed":
        return RED
    return AMBER


def generate_pdf(run_data: dict) -> bytes:
    buf = BytesIO()
    S = _styles()
    margin = 20 * mm

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin, bottomMargin=margin,
        title="Agora Research Report",
        author="Agora",
    )

    story = []

    # Header
    story.append(Paragraph("AGORA RESEARCH REPORT", S["section"]))
    story.append(Paragraph(run_data.get("question", ""), S["title"]))
    run_id = str(run_data.get("id", ""))[:8]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(f"Run {run_id} - Generated {ts}", S["meta"]))
    story.append(Spacer(1, 4))
    story.append(_divider(AMBER))
    story.append(Spacer(1, 8))

    tasks = run_data.get("tasks", [])
    planner_task = next((t for t in tasks if t["kind"] == "planner"), None)
    researchers  = [t for t in tasks if t["kind"] == "researcher"]
    synth_task   = next((t for t in tasks if t["kind"] == "synthesizer"), None)

    # Research Plan
    if planner_task and planner_task.get("output"):
        plan = planner_task["output"]
        story.append(Paragraph("RESEARCH PLAN", S["section"]))
        interp = plan.get("interpretation", "")
        if interp:
            story.append(Paragraph(interp, S["body_italic"]))
            story.append(Spacer(1, 4))
        for i, sq in enumerate(plan.get("sub_questions", []), 1):
            q = sq.get("question", "") if isinstance(sq, dict) else str(sq)
            story.append(Paragraph(f"{i}. {q}", S["subq_text"]))
        story.append(Spacer(1, 8))
        story.append(_divider())

    # Researcher Findings
    if researchers:
        story.append(Paragraph("RESEARCHER FINDINGS", S["section"]))
        for i, r in enumerate(researchers, 1):
            inp    = r.get("input", {})
            out    = r.get("output", {}) or {}
            sub_q  = inp.get("sub_question", f"Sub-question {i}")
            status = r.get("status", "unknown")
            iterations = out.get("iterations", "-")
            terminated = out.get("terminated_reason", "-")
            summary    = out.get("summary", "")
            citations  = out.get("citations", [])

            status_color = GREEN if "completed" in status else RED
            header_data = [[
                Paragraph(f"{i}. {sub_q}", S["subq_text"]),
                Paragraph(status.replace("_", " ").upper(), ParagraphStyle(
                    "st", fontName="Helvetica-Bold", fontSize=8, textColor=status_color, alignment=2,
                )),
            ]]
            header_table = Table(header_data, colWidths=[None, 28*mm])
            header_table.setStyle(TableStyle([
                ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(header_table)

            if summary:
                story.append(Paragraph(summary, S["body"]))

            for c in citations:
                url   = c.get("url", "")   if isinstance(c, dict) else ""
                quote = c.get("quote", "") if isinstance(c, dict) else ""
                if url:
                    story.append(Paragraph(f"-> {url}", S["cite_url"]))
                if quote:
                    story.append(Paragraph(f'"{quote}"', S["cite_quote"]))

            story.append(Paragraph(
                f"Iterations: {iterations} - Terminated: {terminated}", S["small"],
            ))
            story.append(Spacer(1, 6))
            story.append(_divider())

    # Synthesized Answer
    if run_data.get("final_answer"):
        story.append(Paragraph("SYNTHESIZED ANSWER", S["section"]))
        answer_paragraphs = [
            p.strip() for p in run_data["final_answer"].split("\n\n") if p.strip()
        ]
        for p in answer_paragraphs:
            story.append(Paragraph(p, S["answer"]))
        story.append(Spacer(1, 8))
        story.append(_divider())

    # Citations
    if synth_task and synth_task.get("output"):
        citations = synth_task["output"].get("citations", [])
        if citations:
            story.append(Paragraph("CITATIONS", S["section"]))
            for i, c in enumerate(citations, 1):
                url      = c.get("url", "")      if isinstance(c, dict) else ""
                quote    = c.get("quote", "")    if isinstance(c, dict) else ""
                supports = c.get("supports", "") if isinstance(c, dict) else ""
                story.append(Paragraph(f"[{i}] {url}", S["cite_url"]))
                if quote:
                    story.append(Paragraph(f'"{quote}"', S["cite_quote"]))
                if supports:
                    story.append(Paragraph(supports, S["small"]))
                story.append(Spacer(1, 4))
            story.append(_divider())

    # Coverage Notes
    if synth_task and synth_task.get("output"):
        coverage_notes = synth_task["output"].get("coverage", [])
        if coverage_notes:
            story.append(Paragraph("COVERAGE NOTES", S["section"]))
            for cn in coverage_notes:
                sub_q    = cn.get("sub_question", "") if isinstance(cn, dict) else str(cn)
                coverage = cn.get("coverage", "unknown") if isinstance(cn, dict) else "unknown"
                note     = cn.get("note", "") if isinstance(cn, dict) else ""
                badge    = _coverage_badge(coverage)
                c_color  = _coverage_color(coverage)
                row = [[
                    Paragraph(badge, ParagraphStyle(
                        "badge", fontName="Helvetica-Bold", fontSize=10, textColor=c_color,
                    )),
                    [
                        Paragraph(sub_q, S["small_bold"]),
                        Paragraph(note, S["small"]) if note else Spacer(1, 0),
                    ],
                ]]
                t = Table(row, colWidths=[10*mm, None])
                t.setStyle(TableStyle([
                    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING",   (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(t)

    # Footer
    story.append(Spacer(1, 16))
    story.append(_divider(AMBER))
    story.append(Paragraph(
        f"Generated by Agora - Distributed Multi-Agent Research - {ts}", S["meta"],
    ))

    doc.build(story)
    return buf.getvalue()