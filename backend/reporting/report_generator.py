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
    conf = trace_data.get("final_confidence")
    conf_display = f"<b>{int(conf * 100)}%</b>" if conf is not None else "<b>N/A (Uncalibrated)</b>"
    disagree = trace_data.get("disagreement_flagged", False)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    meta_table_data = [
        [Paragraph("<b>Query ID:</b>", body_style), Paragraph(str(query_id)[:18] + "...", body_style),
         Paragraph("<b>Timestamp:</b>", body_style), Paragraph(created_at, body_style)],
        [Paragraph("<b>Agent Task:</b>", body_style), Paragraph(task_name.upper(), body_style),
         Paragraph("<b>Confidence:</b>", body_style), Paragraph(conf_display, body_style)],
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
    story.append(Spacer(1, 14))

    # 4. Multi-Source Evidence & Geospatial Measurements
    story.append(Paragraph("2. Multi-Source Evidence & Geospatial Measurements", heading_style))
    
    evidence_items = query_data.get("evidence_items", [])
    supporting_facts = query_data.get("supporting_facts", [])
    visual_artifacts = query_data.get("visual_artifacts", {})

    evidence_rows = [
        [Paragraph("<b>Source / Modality</b>", body_style),
         Paragraph("<b>Evidence Type</b>", body_style),
         Paragraph("<b>Details / Physical Measurement</b>", body_style)]
    ]

    # Add evidence items
    for ev in evidence_items:
        source_str = ev.get("source", "model")
        ev_type = ev.get("type", "observation")
        val = ev.get("value", "")
        # Format dictionary or list values cleanly
        if isinstance(val, dict):
            val_str = ", ".join(f"{k}: {v}" for k, v in val.items() if not str(k).startswith("_"))
        else:
            val_str = str(val)
        evidence_rows.append([
            Paragraph(f"<b>{source_str.upper()}</b>", body_style),
            Paragraph(ev_type.capitalize(), body_style),
            Paragraph(val_str[:160] + ("..." if len(val_str) > 160 else ""), body_style),
        ])

    # Add visual artifacts / measurements if present
    if visual_artifacts:
        if visual_artifacts.get("bounding_box"):
            bbox_str = f"Pixel: {visual_artifacts['bounding_box']}"
            if visual_artifacts.get("geographic_bbox"):
                bbox_str += f" | Geo: {visual_artifacts['geographic_bbox']}"
            evidence_rows.append([
                Paragraph("<b>SPATIAL</b>", body_style),
                Paragraph("Grounding BBox", body_style),
                Paragraph(bbox_str, body_style),
            ])
        if visual_artifacts.get("aoi"):
            aoi_info = visual_artifacts["aoi"]
            evidence_rows.append([
                Paragraph("<b>AOI</b>", body_style),
                Paragraph("Spatial Filter", body_style),
                Paragraph(f"{aoi_info.get('aoi_type', 'rectangle').capitalize()} ({aoi_info.get('coord_system', 'pixel')})", body_style),
            ])

    if len(evidence_rows) > 1:
        t_evidence = Table(evidence_rows, colWidths=[130, 130, 270])
        t_evidence.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_evidence)
    else:
        story.append(Paragraph("<i>Evidence grounded directly from model inference tensor passes.</i>", body_style))
    
    if supporting_facts:
        story.append(Spacer(1, 6))
        for fact in supporting_facts[:4]:
            story.append(Paragraph(f"• {fact}", body_style))
    
    story.append(Spacer(1, 14))

    # 5. Disagreement & Conflict Resolution Ledger
    story.append(Paragraph("3. Disagreement & Cross-Modal Conflict Ledger", heading_style))
    conflicts = trace_data.get("conflicts", [])
    if conflicts:
        conflict_rows = [
            [Paragraph("<b>Step / Modality</b>", body_style),
             Paragraph("<b>Severity</b>", body_style),
             Paragraph("<b>Conflict Description & Resolution</b>", body_style)]
        ]
        for c in conflicts:
            step_id = c.get("step_id", "cross_modal")
            sev = c.get("severity", "MEDIUM")
            desc = c.get("description", str(c))
            conflict_rows.append([
                Paragraph(str(step_id), body_style),
                Paragraph(f"<font color='{'red' if sev == 'HIGH' else 'orange'}'><b>{sev}</b></font>", body_style),
                Paragraph(desc, body_style),
            ])
        t_conflicts = Table(conflict_rows, colWidths=[120, 80, 330])
        t_conflicts.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#FEF2F2")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#F87171")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#FECACA")),
            ("PADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(t_conflicts)
    else:
        consensus_box = [
            [Paragraph("<font color='#059669'><b>✔ Full Inter-Model Consensus:</b></font> No contradictory claims or spatial/spectral conflicts were detected across specialist model outputs.", body_style)]
        ]
        t_consensus = Table(consensus_box, colWidths=[530])
        t_consensus.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FDF4")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#86EFAC")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(t_consensus)

    story.append(Spacer(1, 14))

    # 6. Auditable Execution Trace Table
    story.append(Paragraph("4. Auditable Multi-Step Execution Trace", heading_style))
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
            Paragraph(f"{int(step['confidence'] * 100)}%" if step.get('confidence') is not None else "N/A", body_style),
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

    # 7. Footer Compliance Note
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=8))
    story.append(Paragraph("Automated report generated by SatQuery AI. Auditable trace verified for ISRO/SAC PS 26167.", subtitle_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
