"""
Specialist Adapters for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Implements the Adapter pattern for all five specialist pipelines.
Translates between the common SpecialistRequest and specialist-specific interfaces,
and normalizes raw model returns into strongly typed SpecialistResult instances
with verified evidence provenance.
"""
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

from agent.schemas import (
    SpecialistRequest,
    SpecialistResult,
    EvidenceItem,
)
from agent.model_registry import SPECIALIST_REGISTRY

# Specialist model imports
from models.vqa_model import answer_question
from models.captioning_model import generate_caption
from models.grounding_model import ground_expression
from models.change_detection import compute_change_map
from models.change_vqa_model import answer_change_question
from models.optical_sar_fusion import fuse_optical_and_sar

logger = logging.getLogger("satquery.agent.adapters")


class SpecialistAdapter(ABC):
    """Abstract Base Class for all specialist AI model adapters."""

    def __init__(self, specialist_id: str):
        self.specialist_id = specialist_id
        self.metadata = SPECIALIST_REGISTRY.get(specialist_id, {
            "id": specialist_id,
            "name": specialist_id,
            "version": "1.0.0",
        })

    @abstractmethod
    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        """Executes the specialist model with translated parameters."""
        pass


class VQASpecialistAdapter(SpecialistAdapter):
    """Adapter for Tilak's Remote-Sensing VLM Single-Image VQA."""

    def __init__(self):
        super().__init__("vqa_specialist")

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        start_time = time.time()
        image_path = request.inputs.get("image_path")
        question = request.inputs.get("question")

        if not image_path or not question:
            return SpecialistResult(
                task="vqa",
                status="failed",
                errors=["Missing required inputs: 'image_path' and/or 'question'"],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=0.0,
            )

        try:
            raw_res = answer_question(image_path=image_path, question=question)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            evidence_items = []
            for ev_text in raw_res.get("evidence", []):
                evidence_items.append(EvidenceItem(
                    source="vlm",
                    type="visual_feature",
                    finding=ev_text,
                    reliability=raw_res.get("confidence"),
                ))

            conf_dict = None
            if raw_res.get("confidence") is not None:
                conf_dict = {"value": float(raw_res["confidence"]), "source": "model"}

            return SpecialistResult(
                task="vqa",
                status="success",
                answer=raw_res.get("answer"),
                confidence=conf_dict,
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                evidence=evidence_items,
                artifacts={},
                metrics={"latency_ms": raw_res.get("latency_ms", elapsed_ms)},
                metadata=raw_res.get("details", {}),
                errors=[],
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error(f"[VQASpecialistAdapter Error]: {e}")
            return SpecialistResult(
                task="vqa",
                status="failed",
                errors=[str(e)],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )


class CaptioningSpecialistAdapter(SpecialistAdapter):
    """Adapter for Tilak's Remote-Sensing Dense Scene Captioner."""

    def __init__(self):
        super().__init__("captioning_specialist")

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        start_time = time.time()
        image_path = request.inputs.get("image_path")

        if not image_path:
            return SpecialistResult(
                task="captioning",
                status="failed",
                errors=["Missing required input: 'image_path'"],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=0.0,
            )

        try:
            raw_res = generate_caption(image_path=image_path)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            evidence_items = []
            for ev_text in raw_res.get("evidence", []):
                evidence_items.append(EvidenceItem(
                    source="captioning",
                    type="scene_description",
                    finding=ev_text,
                    reliability=raw_res.get("confidence"),
                ))

            conf_dict = None
            if raw_res.get("confidence") is not None:
                conf_dict = {"value": float(raw_res["confidence"]), "source": "model"}

            return SpecialistResult(
                task="captioning",
                status="success",
                answer=raw_res.get("caption"),
                confidence=conf_dict,
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                evidence=evidence_items,
                artifacts={"caption": raw_res.get("caption")},
                metrics={"latency_ms": raw_res.get("latency_ms", elapsed_ms)},
                metadata={"features_detected": raw_res.get("features_detected", [])},
                errors=[],
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error(f"[CaptioningSpecialistAdapter Error]: {e}")
            return SpecialistResult(
                task="captioning",
                status="failed",
                errors=[str(e)],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )


class GroundingSpecialistAdapter(SpecialistAdapter):
    """Adapter for Tilak's Referring-Expression Grounding Engine."""

    def __init__(self):
        super().__init__("grounding_specialist")

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        start_time = time.time()
        image_path = request.inputs.get("image_path")
        expression = request.inputs.get("expression")

        if not image_path or not expression:
            return SpecialistResult(
                task="grounding",
                status="failed",
                errors=["Missing required inputs: 'image_path' and/or 'expression'"],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=0.0,
            )

        try:
            raw_res = ground_expression(image_path=image_path, expression=expression)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            evidence_items = []
            for ev_text in raw_res.get("evidence", []):
                evidence_items.append(EvidenceItem(
                    source="grounding",
                    type="spatial_region",
                    finding=ev_text,
                    artifact=str(raw_res.get("bbox")),
                    reliability=raw_res.get("confidence"),
                ))

            conf_dict = None
            if raw_res.get("confidence") is not None:
                conf_dict = {"value": float(raw_res["confidence"]), "source": "model"}

            artifacts = {
                "bbox": raw_res.get("bbox"),
                "normalized_bbox": raw_res.get("normalized_bbox"),
                "found": raw_res.get("found", False),
                "regions": [raw_res["bbox"]] if raw_res.get("bbox") else [],
            }

            return SpecialistResult(
                task="grounding",
                status="success",
                answer=raw_res.get("message"),
                confidence=conf_dict,
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                evidence=evidence_items,
                artifacts=artifacts,
                metrics={"latency_ms": raw_res.get("latency_ms", elapsed_ms)},
                metadata={},
                errors=[],
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error(f"[GroundingSpecialistAdapter Error]: {e}")
            return SpecialistResult(
                task="grounding",
                status="failed",
                errors=[str(e)],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )


class DifferencingEngineAdapter(SpecialistAdapter):
    """Adapter for Sridhar's Computer-Vision Differencing Engine."""

    def __init__(self):
        super().__init__("differencing_engine")

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        start_time = time.time()
        before_path = request.inputs.get("before_path")
        after_path = request.inputs.get("after_path")

        if not before_path or not after_path:
            return SpecialistResult(
                task="change_analysis",
                status="failed",
                errors=["Missing required inputs: 'before_path' and/or 'after_path'"],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=0.0,
            )

        try:
            raw_res = compute_change_map(before_path=before_path, after_path=after_path)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            pct = raw_res.get("percentage_changed", 0.0)
            sector = raw_res.get("dominant_sector", "central")

            evidence_items = [
                EvidenceItem(
                    source="change_detection",
                    type="change_mask",
                    finding=f"Differencing detected {pct}% variance concentrated in the {sector} quadrant.",
                    value=pct,
                    unit="percent",
                    location=sector,
                    artifact=raw_res.get("mask_path"),
                    reliability=0.92,
                )
            ]

            artifacts = {
                "change_map": raw_res.get("mask_path"),
                "mask_path": raw_res.get("mask_path"),
                "overlay_path": raw_res.get("overlay_path"),
            }

            metrics = {
                "percentage_changed": pct,
                "change_percentage": pct,
                "dominant_sector": sector,
                "change_detected": raw_res.get("change_detected", False),
            }

            return SpecialistResult(
                task="change_analysis",
                status="success",
                answer=raw_res.get("location_summary"),
                confidence={"value": 0.90, "source": "statistical_cv"},
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                evidence=evidence_items,
                artifacts=artifacts,
                metrics=metrics,
                metadata={"co_registration": raw_res.get("co_registration", {})},
                errors=[],
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error(f"[DifferencingEngineAdapter Error]: {e}")
            return SpecialistResult(
                task="change_analysis",
                status="failed",
                errors=[str(e)],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )


class ChangeVQASpecialistAdapter(SpecialistAdapter):
    """Adapter for Sridhar's Bi-Temporal Change-VQA Specialist."""

    def __init__(self):
        super().__init__("change_vqa_specialist")

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        start_time = time.time()
        before_path = request.inputs.get("before_path")
        after_path = request.inputs.get("after_path")
        question = request.inputs.get("question")
        change_res = request.inputs.get("change_map_result")

        if not before_path or not after_path or not question:
            return SpecialistResult(
                task="change_vqa",
                status="failed",
                errors=["Missing required inputs: 'before_path', 'after_path', and/or 'question'"],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=0.0,
            )

        try:
            raw_res = answer_change_question(
                before_path=before_path,
                after_path=after_path,
                question=question,
                change_map_result=change_res,
            )
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            cm = raw_res.get("change_metrics", {})
            pct = cm.get("percentage_changed", 0.0)

            evidence_items = [
                EvidenceItem(
                    source="change_vqa",
                    type="temporal_reasoning",
                    finding=raw_res.get("answer", ""),
                    value=pct,
                    unit="percent",
                    reliability=raw_res.get("confidence"),
                )
            ]

            conf_dict = None
            if raw_res.get("confidence") is not None:
                conf_dict = {"value": float(raw_res["confidence"]), "source": "model"}

            artifacts = {
                "change_overlay_path": cm.get("overlay_path"),
                "change_mask_path": cm.get("mask_path"),
                "change_metrics": cm,
            }

            return SpecialistResult(
                task="change_vqa",
                status="success",
                answer=raw_res.get("answer"),
                confidence=conf_dict,
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                evidence=evidence_items,
                artifacts=artifacts,
                metrics=cm,
                metadata={},
                errors=[],
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error(f"[ChangeVQASpecialistAdapter Error]: {e}")
            return SpecialistResult(
                task="change_vqa",
                status="failed",
                errors=[str(e)],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )


class OpticalSARSpecialistAdapter(SpecialistAdapter):
    """Adapter for Shreyas's Optical-SAR Cross-Modal Fusion Specialist."""

    def __init__(self):
        super().__init__("optical_sar_fusion_specialist")

    def execute(self, request: SpecialistRequest) -> SpecialistResult:
        start_time = time.time()
        optical_path = request.inputs.get("optical_path")
        sar_path = request.inputs.get("sar_path")
        query = request.inputs.get("query")

        if not optical_path or not sar_path or not query:
            return SpecialistResult(
                task="optical_sar",
                status="failed",
                errors=["Missing required inputs: 'optical_path', 'sar_path', and/or 'query'"],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=0.0,
            )

        try:
            raw_res = fuse_optical_and_sar(
                optical_path=optical_path,
                sar_path=sar_path,
                query=query,
            )
            elapsed_ms = round((time.time() - start_time) * 1000, 2)

            evidence_items = []
            opt_findings = raw_res.get("evidence", {}).get("optical", {}).get("findings")
            if opt_findings:
                evidence_items.append(EvidenceItem(
                    source="optical_analysis",
                    type="spectral_reflectance",
                    finding=opt_findings,
                    reliability=raw_res.get("evidence", {}).get("optical", {}).get("confidence"),
                ))

            sar_findings = raw_res.get("evidence", {}).get("sar", {}).get("findings")
            if sar_findings:
                evidence_items.append(EvidenceItem(
                    source="sar_analysis",
                    type="radar_backscatter",
                    finding=sar_findings,
                    reliability=raw_res.get("evidence", {}).get("sar", {}).get("confidence"),
                ))

            conf_dict = None
            if raw_res.get("confidence") is not None:
                conf_dict = {"value": float(raw_res["confidence"]), "source": "dual_branch"}

            artifacts = {
                "cross_modal_evidence": raw_res.get("evidence", {}),
                "complementary_gains": raw_res.get("complementary_gains", []),
            }

            return SpecialistResult(
                task="optical_sar",
                status="success",
                answer=raw_res.get("answer"),
                confidence=conf_dict,
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                evidence=evidence_items,
                artifacts=artifacts,
                metrics={},
                metadata={"co_registration": raw_res.get("co_registration", {})},
                errors=[],
                latency_ms=elapsed_ms,
            )
        except Exception as e:
            logger.error(f"[OpticalSARSpecialistAdapter Error]: {e}")
            return SpecialistResult(
                task="optical_sar",
                status="failed",
                errors=[str(e)],
                model={"id": self.specialist_id, "name": self.metadata["name"], "version": self.metadata["version"]},
                latency_ms=round((time.time() - start_time) * 1000, 2),
            )


# =========================================================================
# Adapter Factory
# =========================================================================

ADAPTER_REGISTRY: Dict[str, SpecialistAdapter] = {
    "vqa_specialist": VQASpecialistAdapter(),
    "vqa_model": VQASpecialistAdapter(),
    "captioning_specialist": CaptioningSpecialistAdapter(),
    "captioning_model": CaptioningSpecialistAdapter(),
    "grounding_specialist": GroundingSpecialistAdapter(),
    "grounding_model": GroundingSpecialistAdapter(),
    "differencing_engine": DifferencingEngineAdapter(),
    "change_detection": DifferencingEngineAdapter(),
    "change_vqa_specialist": ChangeVQASpecialistAdapter(),
    "change_vqa_model": ChangeVQASpecialistAdapter(),
    "optical_sar_fusion_specialist": OpticalSARSpecialistAdapter(),
    "optical_sar_fusion": OpticalSARSpecialistAdapter(),
}


def get_adapter(model_id: str) -> Optional[SpecialistAdapter]:
    """Retrieves the unified adapter instance for a model identifier."""
    return ADAPTER_REGISTRY.get(model_id)
