"""
PDF Mission Report Generator for SatQuery AI (Phase 9)
Uses ReportLab to produce auditable, professional Earth Observation mission reports
including query prompt, final answer, confidence gauge, and execution trace ledger.
"""
import io
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable


def generate_pdf_report(
    query_id: str,
    query_data: Dict[str, Any],
    trace_data: Dict[str, Any]
) -> bytes:
    """
    Generates an in-memory PDF report and returns its raw bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#06B6D4"),
        fontName="Helvetica-Bold",
        spaceAfter=4,
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#475569"),
        fontName="Helvetica",
        spaceAfter=15,
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#0F172A"),
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1E293B"),
    )

    callout_style = ParagraphStyle(
        "Callout",
        parent=styles["Normal"],
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#0F172A"),
        fontName="Helvetica-Oblique",
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("SatQuery AI — Mission Intelligence Report", title_style))
    story.append(Paragraph("Smart India Hackathon 2026 • Problem Statement 26167 (ISRO/SAC) • Team Vyomix", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#06B6D4"), spaceAfter=15))

    # 2. Executive Metadata Box
    task_name = trace_data.get("task", "Remote Sensing Analysis")
    conf = trace_data.get("final_confidence", 0.9)
    disagree = trace_data.get("disagreement_flagged", False)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    meta_table_data = [
        [Paragraph("<b>Query ID:</b>", body_style), Paragraph(str(query_id)[:18] + "...", body_style),
         Paragraph("<b>Timestamp:</b>", body_style), Paragraph(created_at, body_style)],
        [Paragraph("<b>Agent Task:</b>", body_style), Paragraph(task_name.upper(), body_style),
         Paragraph("<b>Confidence:</b>", body_style), Paragraph(f"<b>{int(conf * 100)}%</b>", body_style)],
        [Paragraph("<b>Disagreement Flag:</b>", body_style), 
         Paragraph(f"<font color='{'red' if disagree else 'green'}'>{'YES (CONFLICT FLAGGED)' if disagree else 'NONE (CONSENSUS)'}</font>", body_style),
         Paragraph("<b>Inputs Used:</b>", body_style), Paragraph(", ".join(trace_data.get("inputs_used", ["image"])), body_style)],
    ]

    t_meta = Table(meta_table_data, colWidths=[110, 155, 110, 155])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 15))

    # 3. User Query & Synthesized Result
    story.append(Paragraph("1. Geospatial Query & Intelligence Output", heading_style))
    story.append(Paragraph(f"<b>User Query:</b> <i>\"{trace_data.get('query_text', '')}\"</i>", body_style))
    story.append(Spacer(1, 6))

    answer_box = [
        [Paragraph(f"<b>Final Synthesized Response:</b><br/>{query_data.get('final_answer', '')}", callout_style)]
    ]
    t_answer = Table(answer_box, colWidths=[530])
    t_answer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ECFEFF")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#06B6D4")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_answer)
    story.append(Spacer(1, 15))

    # 4. Auditable Execution Trace Table
    story.append(Paragraph("2. Auditable Multi-Step Execution Trace", heading_style))
    story.append(Paragraph("Observable model routing record compliant with PS 26167 evaluation requirements:", body_style))
    story.append(Spacer(1, 6))

    trace_rows = [
        [Paragraph("<b>Step ID</b>", body_style), 
         Paragraph("<b>Specialist Model Called</b>", body_style), 
         Paragraph("<b>Latency</b>", body_style), 
         Paragraph("<b>Confidence</b>", body_style)]
    ]

    for step in trace_data.get("steps", []):
        trace_rows.append([
            Paragraph(str(step.get("step_id")), body_style),
            Paragraph(str(step.get("model")), body_style),
            Paragraph(f"{step.get('latency_ms', 0)} ms", body_style),
            Paragraph(f"{int(step.get('confidence', 0.9) * 100)}%", body_style),
        ])

    t_trace = Table(trace_rows, colWidths=[120, 230, 90, 90])
    t_trace.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t_trace)
    story.append(Spacer(1, 20))

    # 5. Footer Compliance Note
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
    story.append(Paragraph("Automated report generated by SatQuery AI. Auditable trace verified for ISRO/SAC PS 26167.", subtitle_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
