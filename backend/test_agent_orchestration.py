"""
Comprehensive Master Orchestration & Contract Verification Test Suite
SIH 2026 Problem Statement 26167 | Team Vyomix | Vinayaka (Agentic AI Engineer)

Tests all mandatory evaluation criteria:
- Test 1: Single-Image VQA Pipeline
- Test 2: Dense Scene Captioning Pipeline
- Test 3: Referring Expression Grounding Pipeline
- Test 4: Bi-Temporal Change Detection & Differencing Pipeline
- Test 5: Optical + SAR Dual-Branch Cross-Modal Fusion Pipeline
- Test 6: Precondition Rejection: Single Image provided for Change Analysis
- Test 7: Precondition Rejection: Non-complementary modalities for Optical-SAR
- Test 8: Input File Validation: Unsupported file extension rejection
- Test 9: Specialist Failure Handling: Graceful degradation without API crash
- Test 10: Dependency-Aware Execution: Cascading BLOCKED step propagation
- Test 11: Cross-Specialist Evidence Conflict & Disagreement Detection
- Test 12: Defensible Confidence Engine: Genuine null handling without fake fallbacks
- Test 13: Natural-Language Query Paraphrase Invariance & Auto-Routing
- Test 14: Specialist Adapter Contract Verification (SpecialistRequest -> SpecialistResult)
"""
import os
import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.schemas import (
    QueryIntent,
    TaskType,
    ValidationResult,
    SpecialistRequest,
    SpecialistResult,
    ExecutionStatus,
)
from agent.query_interpreter import interpret_query, parse_query_intent
from agent.task_classifier import classify_and_validate_task, validate_task_requirements
from agent.geospatial_validator import validate_geospatial_compatibility
from agent.model_registry import SPECIALIST_REGISTRY, get_specialist, is_specialist_available
from agent.planner import create_execution_plan
from agent.executor import execute_plan
from agent.adapters import get_adapter, ADAPTER_REGISTRY
from agent.evidence_fusion import fuse_execution_evidence
from agent.confidence import evaluate_confidence_and_conflicts, compute_defensible_confidence
from agent.execution_trace import build_execution_trace
from validation.file_validator import validate_file_format

TEST_SCRATCH = Path(__file__).resolve().parent / "data" / "test_scratch_orchestration"
TEST_SCRATCH.mkdir(parents=True, exist_ok=True)


def _create_dummy_image(name: str, mode: str = "RGB", size=(64, 64)) -> str:
    img_path = TEST_SCRATCH / name
    img = Image.new(mode, size, color=(128, 128, 128) if mode == "RGB" else 128)
    img.save(img_path)
    return str(img_path)


class TestAgentOrchestration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.img_optical = _create_dummy_image("test_opt.png", "RGB")
        cls.img_sar = _create_dummy_image("test_sar.png", "L")
        cls.img_before = _create_dummy_image("test_t0.png", "RGB")
        cls.img_after = _create_dummy_image("test_t1.png", "RGB")

    # ---------------------------------------------------------------------
    # Test 1: VQA Pipeline
    # ---------------------------------------------------------------------
    def test_01_vqa_pipeline(self):
        query = "What is present in this image?"
        intent = interpret_query(query)
        self.assertEqual(intent["task"], "single_image_vqa")

        manifest = {"optical": {"saved_path": self.img_optical, "modality": {"modality": "OPTICAL"}}}
        ok, msg, cfg = classify_and_validate_task(intent, manifest)
        self.assertTrue(ok)

        plan = create_execution_plan(cfg, manifest, query)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["model_id"], "vqa_specialist")

        results = execute_plan(plan)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "SUCCESS")
        self.assertIsNotNone(results[0]["output"].get("answer"))

        fused = fuse_execution_evidence(results, "single_image_vqa", raw_query=query)
        self.assertTrue(len(fused["final_answer"]) > 0)

    # ---------------------------------------------------------------------
    # Test 2: Captioning Pipeline
    # ---------------------------------------------------------------------
    def test_02_captioning_pipeline(self):
        query = "Describe this satellite image and summarize the scene."
        intent = interpret_query(query)
        self.assertEqual(intent["task"], "captioning")

        manifest = {"optical": {"saved_path": self.img_optical, "modality": {"modality": "OPTICAL"}}}
        ok, msg, cfg = classify_and_validate_task(intent, manifest)
        self.assertTrue(ok)

        plan = create_execution_plan(cfg, manifest, query)
        self.assertEqual(plan[0]["model_id"], "captioning_specialist")

        results = execute_plan(plan)
        self.assertEqual(results[0]["status"], "SUCCESS")
        self.assertIn("caption", results[0]["output"])

    # ---------------------------------------------------------------------
    # Test 3: Grounding Pipeline
    # ---------------------------------------------------------------------
    def test_03_grounding_pipeline(self):
        query = "Where are the buildings in this scene?"
        intent = interpret_query(query)
        self.assertEqual(intent["task"], "grounding")

        manifest = {"optical": {"saved_path": self.img_optical, "modality": {"modality": "OPTICAL"}}}
        ok, msg, cfg = classify_and_validate_task(intent, manifest)
        self.assertTrue(ok)

        plan = create_execution_plan(cfg, manifest, query)
        self.assertEqual(plan[0]["model_id"], "grounding_specialist")

        results = execute_plan(plan)
        self.assertEqual(results[0]["status"], "SUCCESS")
        self.assertIn("found", results[0]["output"])
        self.assertIn("bbox", results[0]["output"])

    # ---------------------------------------------------------------------
    # Test 4: Change Detection Pipeline
    # ---------------------------------------------------------------------
    def test_04_change_pipeline(self):
        query = "What changed between 2024 and 2026?"
        intent = interpret_query(query)
        self.assertEqual(intent["task"], "change_vqa")

        manifest = {
            "before": {"saved_path": self.img_before, "modality": {"modality": "OPTICAL"}},
            "after": {"saved_path": self.img_after, "modality": {"modality": "OPTICAL"}},
        }
        ok, msg, cfg = classify_and_validate_task(intent, manifest)
        self.assertTrue(ok)

        plan = create_execution_plan(cfg, manifest, query)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["model_id"], "differencing_engine")
        self.assertEqual(plan[1]["model_id"], "change_vqa_specialist")
        self.assertEqual(plan[1]["depends_on"], ["step_1_cv_differencing"])

        results = execute_plan(plan)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["status"], "SUCCESS")
        self.assertEqual(results[1]["status"], "SUCCESS")

        fused = fuse_execution_evidence(results, "change_vqa", raw_query=query)
        self.assertIn("percentage_changed", results[0]["output"])
        self.assertIn("change_metrics", results[1]["output"])

    # ---------------------------------------------------------------------
    # Test 5: Optical + SAR Cross-Modal Pipeline
    # ---------------------------------------------------------------------
    def test_05_optical_sar_pipeline(self):
        query = "Compare optical and SAR imagery to identify water bodies."
        intent = interpret_query(query)
        self.assertEqual(intent["task"], "optical_sar_fusion")

        manifest = {
            "optical": {"saved_path": self.img_optical, "modality": {"modality": "OPTICAL"}},
            "sar": {"saved_path": self.img_sar, "modality": {"modality": "SAR"}},
        }
        ok, msg, cfg = classify_and_validate_task(intent, manifest)
        self.assertTrue(ok)

        plan = create_execution_plan(cfg, manifest, query)
        self.assertEqual(plan[0]["model_id"], "optical_sar_fusion_specialist")

        results = execute_plan(plan)
        self.assertEqual(results[0]["status"], "SUCCESS")
        self.assertIn("evidence", results[0]["output"])

    # ---------------------------------------------------------------------
    # Test 6: Precondition Rejection: Single Image for Change
    # ---------------------------------------------------------------------
    def test_06_insufficient_temporal_inputs(self):
        query = "What changed between these observation dates?"
        intent = interpret_query(query)
        manifest_single = {"optical": {"saved_path": self.img_optical, "modality": {"modality": "OPTICAL"}}}

        ok, msg, cfg = classify_and_validate_task(intent, manifest_single)
        self.assertFalse(ok)
        self.assertIn("E-TEMP-01", msg)

    # ---------------------------------------------------------------------
    # Test 7: Precondition Rejection: Non-complementary modalities for Optical-SAR
    # ---------------------------------------------------------------------
    def test_07_invalid_modalities_for_fusion(self):
        query = "Compare optical and SAR sensors."
        intent = interpret_query(query)
        manifest_dual_optical = {
            "image_1": {"saved_path": self.img_optical, "modality": {"modality": "OPTICAL"}},
            "image_2": {"saved_path": self.img_after, "modality": {"modality": "OPTICAL"}},
        }
        ok, msg, cfg = classify_and_validate_task(intent, manifest_dual_optical)
        self.assertFalse(ok)
        self.assertIn("E-FUSE-02", msg)

    # ---------------------------------------------------------------------
    # Test 8: Input File Format Validation
    # ---------------------------------------------------------------------
    def test_08_unsupported_file_format(self):
        bad_file = TEST_SCRATCH / "corrupted.xyz"
        bad_file.write_text("not a geotiff or image")
        is_valid, msg, _ = validate_file_format(str(bad_file), is_benchmark_input=False)
        self.assertFalse(is_valid)
        self.assertIn("Unsupported file format", msg)

    # ---------------------------------------------------------------------
    # Test 9: Specialist Failure Handling
    # ---------------------------------------------------------------------
    def test_09_specialist_failure_handling(self):
        plan = [{
            "step_id": "step_fail",
            "model_id": "vqa_specialist",
            "model_name": "Test Failure Specialist",
            "inputs": {"image_path": "C:/non_existent_image.tif", "question": "Any question?"},
            "depends_on": [],
        }]
        results = execute_plan(plan)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "FAILED")
        self.assertIsNotNone(results[0]["error"])
        # Verify no fake confidence was injected
        self.assertIsNone(results[0]["confidence"])

    # ---------------------------------------------------------------------
    # Test 10: Dependency-Aware Execution (BLOCKED Cascade)
    # ---------------------------------------------------------------------
    def test_10_dependency_cascade_blocked(self):
        plan = [
            {
                "step_id": "step_1_diff",
                "model_id": "differencing_engine",
                "inputs": {"before_path": "C:/missing_before.tif", "after_path": "C:/missing_after.tif"},
                "depends_on": [],
            },
            {
                "step_id": "step_2_vqa",
                "model_id": "change_vqa_specialist",
                "inputs": {"before_path": "a", "after_path": "b", "question": "q"},
                "depends_on": ["step_1_diff"],
            }
        ]
        results = execute_plan(plan)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["status"], "FAILED")
        self.assertEqual(results[1]["status"], "BLOCKED")
        self.assertIn("BLOCKED", results[1]["error"])

    # ---------------------------------------------------------------------
    # Test 11: Cross-Specialist Evidence Conflict & Disagreement Detection
    # ---------------------------------------------------------------------
    def test_11_evidence_conflict_detection(self):
        mock_steps = [
            {
                "step_id": "step_1_differencing",
                "model_called": "Differencing Engine",
                "confidence": 0.90,
                "output": {"percentage_changed": 0.4}  # 0.4% change
            },
            {
                "step_id": "step_2_change_vqa",
                "model_called": "Change-VQA Specialist",
                "confidence": 0.92,
                "output": {"answer": "Substantial expansion has occurred with massive increase in built-up area."}
            }
        ]
        conf, flagged, conflicts = evaluate_confidence_and_conflicts(mock_steps, "change_vqa")
        self.assertTrue(flagged)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["conflict_id"], "CONFLICT_TEMP_01")
        self.assertLess(conf, 0.70)  # Penalized

    # ---------------------------------------------------------------------
    # Test 12: Missing Confidence Handling (Null Confidence)
    # ---------------------------------------------------------------------
    def test_12_missing_confidence_handling(self):
        uncalibrated_steps = [{
            "step_id": "step_uncal",
            "model_called": "Experimental Specialist",
            "confidence": None,
            "output": {"answer": "Some result."}
        }]
        conf, flagged, conflicts = evaluate_confidence_and_conflicts(uncalibrated_steps, "custom_task")
        # Must return None or uncalibrated, never fabricate 0.95 or 0.85
        self.assertIsNone(conf)

    # ---------------------------------------------------------------------
    # Test 13: Query Paraphrase Routing Invariance
    # ---------------------------------------------------------------------
    def test_13_query_paraphrase_routing(self):
        vqa_queries = [
            "Are there buildings in this satellite scene?",
            "Can you identify buildings?",
            "Does this scene contain buildings?",
            "What is the predominant land cover?",
        ]
        for q in vqa_queries:
            intent = interpret_query(q)
            self.assertEqual(intent["task"], "single_image_vqa", f"Failed on: '{q}'")

        grounding_queries = [
            "Where are the buildings?",
            "Locate all roads in this image.",
            "Highlight the water bodies.",
            "Delineate the runway.",
        ]
        for q in grounding_queries:
            intent = interpret_query(q)
            self.assertEqual(intent["task"], "grounding", f"Failed on: '{q}'")

        change_queries = [
            "What changed between 2024 and 2026?",
            "Has urbanization increased between the two dates?",
            "Describe the surface alterations before and after.",
        ]
        for q in change_queries:
            intent = interpret_query(q)
            self.assertEqual(intent["task"], "change_vqa", f"Failed on: '{q}'")

    # ---------------------------------------------------------------------
    # Test 14: Specialist Adapter Contract Verification
    # ---------------------------------------------------------------------
    def test_14_adapter_contracts(self):
        for model_id, adapter in ADAPTER_REGISTRY.items():
            self.assertIsNotNone(adapter.specialist_id)
            # Test empty request handling
            res = adapter.execute(SpecialistRequest(specialist_id=model_id, task="test", inputs={}))
            self.assertIsInstance(res, SpecialistResult)
            self.assertIn(res.status, ("success", "failed", "degraded"))
            self.assertIsInstance(res.evidence, list)
            self.assertIsInstance(res.errors, list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
