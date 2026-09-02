"""
Specialist Adapters Interface for SatQuery AI (Phase 13)
Provides standardized, unified adapter interfaces for all five specialist capabilities:
  1. Single-Image VQA Specialist
  2. Dense Scene Captioning Specialist
  3. Referring-Expression Grounding Specialist
  4. Bi-Temporal Change Detection & Change-VQA Specialist
  5. Optical + SAR Cross-Modal Fusion Specialist
"""
from typing import Dict, Any, Optional
from models.vqa_model import answer_question
from models.captioning_model import generate_caption
from models.grounding_model import ground_expression
from models.change_detection import compute_change_map
from models.change_vqa_model import answer_change_question
from models.optical_sar_fusion import fuse_optical_and_sar


class SpecialistAdapters:
    """Unified specialist invocation facade."""

    @staticmethod
    def run_vqa(image_path: str, question: str) -> Dict[str, Any]:
        """Specialist 1: Single-Image VQA"""
        return answer_question(image_path=image_path, question=question)

    @staticmethod
    def run_captioning(image_path: str) -> Dict[str, Any]:
        """Specialist 2: Dense Scene Captioning"""
        return generate_caption(image_path=image_path)

    @staticmethod
    def run_grounding(image_path: str, expression: str) -> Dict[str, Any]:
        """Specialist 3: Text-Guided Referring Expression Grounding"""
        return ground_expression(image_path=image_path, expression=expression)

    @staticmethod
    def run_change_analysis(
        before_path: str, 
        after_path: str, 
        question: str, 
        change_map: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Specialist 4: Bi-Temporal Differencing & Change-VQA"""
        if change_map is None:
            change_map = compute_change_map(before_path=before_path, after_path=after_path)
        
        vqa_res = answer_change_question(
            before_path=before_path, 
            after_path=after_path, 
            question=question, 
            change_map_result=change_map
        )
        return {
            "differencing": change_map,
            "change_vqa": vqa_res,
            "answer": vqa_res["answer"],
            "confidence": vqa_res["confidence"],
            "change_metrics": vqa_res["change_metrics"],
        }

    @staticmethod
    def run_optical_sar_fusion(optical_path: str, sar_path: str, query: str) -> Dict[str, Any]:
        """Specialist 5: Optical + SAR Dual-Branch Cross-Modal Fusion"""
        return fuse_optical_and_sar(optical_path=optical_path, sar_path=sar_path, query=query)
