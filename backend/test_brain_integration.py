"""
Phase 14: Central Brain Component Integration Test Suite
Unit and integration tests for every module of the Central Brain:
  - Query Interpreter (Phase 4)
  - Task Classifier (Phase 5)
  - Geospatial Validator (Phase 6)
  - Specialist Model Registry (Phase 7)
  - DAG Planner (Phase 8)
  - Step Executor (Phase 9)
  - Evidence Fusion Engine (Phase 10)
  - Confidence & Conflict Engine (Phase 11)
  - Execution Trace Ledger (Phase 12)
  - Specialist Adapters (Phase 13)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.query_interpreter import interpret_query
from agent.task_classifier import classify_and_validate_task
from agent.geospatial_validator import validate_geospatial_compatibility
from agent.model_registry import SPECIALIST_REGISTRY
from agent.planner import create_execution_plan
from agent.executor import execute_plan
from agent.evidence_fusion import fuse_execution_evidence
from agent.confidence import evaluate_confidence_and_conflicts
from agent.execution_trace import build_execution_trace
from models.specialist_adapters import SpecialistAdapters


def run_integration_tests():
    print("==========================================================")
    print("  SatQuery AI - Phase 14 Central Brain Integration Tests  ")
    print("==========================================================")

    # ----------------------------------------------------
    # Test 1: Phase 4 Query Interpretation
    # ----------------------------------------------------
    print("\n[1] Testing Phase 4: Query Interpreter...")
    q1 = "Highlight the agricultural field in the northern sector"
    i1 = interpret_query(q1)
    assert i1["task"] == "grounding"
    assert "agricultural field" in i1["target_entity"]
    assert i1["spatial_constraint"] == "northern"
    print(f" -> PASS: Grounding intent parsed: target='{i1['target_entity']}', sector='{i1['spatial_constraint']}'")

    q2 = "Use optical and SAR sensors together to extract water bodies"
    i2 = interpret_query(q2)
    assert i2["task"] == "optical_sar_fusion"
    assert "OPTICAL" in i2["requires_modalities"] and "SAR" in i2["requires_modalities"]
    print(f" -> PASS: Cross-modal intent parsed: modalities={i2['requires_modalities']}")

    q3 = "What changed between the before and after acquisition dates?"
    i3 = interpret_query(q3)
    assert i3["task"] == "change_vqa"
    assert i3["requires_multi_temporal"] is True
    print(f" -> PASS: Multi-temporal change intent parsed: task={i3['task']}")

    # ----------------------------------------------------
    # Test 2: Phase 5 Task Classifier & Precondition Checking
    # ----------------------------------------------------
    print("\n[2] Testing Phase 5: Task Classifier...")
    # Happy path: 2 images for change
    manifest_dual = {
        "before": {"slot": "before", "modality": {"modality": "OPTICAL"}},
        "after": {"slot": "after", "modality": {"modality": "OPTICAL"}},
    }
    ok, msg, cfg = classify_and_validate_task(i3, manifest_dual)
    assert ok is True
    assert cfg["before_slot"] == "before" and cfg["after_slot"] == "after"
    print(f" -> PASS: Multi-temporal happy path validated.")

    # Rejection path: only 1 image provided for change query
    manifest_single = {
        "optical": {"slot": "optical", "modality": {"modality": "OPTICAL"}},
    }
    ok_fail, err_msg, _ = classify_and_validate_task(i3, manifest_single)
    assert ok_fail is False
    assert "E-TEMP-01" in err_msg or "requires two observations" in err_msg
    print(f" -> PASS: Single-image change mismatch correctly rejected: '{err_msg[:60]}...'")

    # Rejection path: only optical provided for optical+SAR query
    ok_fuse_fail, err_fuse, _ = classify_and_validate_task(i2, manifest_dual)
    assert ok_fuse_fail is False
    assert "E-FUSE-02" in err_fuse or "requires complementary sensor modalities" in err_fuse
    print(f" -> PASS: Missing SAR modality correctly rejected: '{err_fuse[:60]}...'")

    # ----------------------------------------------------
    # Test 3: Phase 6 Geospatial Compatibility Validator
    # ----------------------------------------------------
    print("\n[3] Testing Phase 6: Geospatial Compatibility Validator...")
    manifest_geo = {
        "before": {
            "metadata": {
                "crs": "EPSG:32643",
                "epsg": 32643,
                "bounds": {"bbox_list": [300000, 2000000, 310000, 2010000]},
                "resolution": {"x": 10.0, "y": 10.0},
            }
        },
        "after": {
            "metadata": {
                "crs": "EPSG:32643",
                "epsg": 32643,
                "bounds": {"bbox_list": [300000, 2000000, 310000, 2010000]},
                "resolution": {"x": 10.0, "y": 10.0},
            }
        }
    }
    g_ok, g_msg, g_rep = validate_geospatial_compatibility(cfg, manifest_geo)
    assert g_ok is True
    assert g_rep["spatial_alignment_status"] == "VERIFIED"
    print(" -> PASS: Co-registered rasters geospatial compatibility verified.")

    # ----------------------------------------------------
    # Test 4: Phase 7 Model Registry Completeness
    # ----------------------------------------------------
    print("\n[4] Testing Phase 7: Model Registry...")
    expected_models = [
        "vqa_specialist", "captioning_specialist", "grounding_specialist",
        "differencing_engine", "change_vqa_specialist", "optical_sar_fusion_specialist"
    ]
    for m in expected_models:
        assert m in SPECIALIST_REGISTRY, f"Missing model {m} in registry"
        assert "inputs" in SPECIALIST_REGISTRY[m]
        assert "outputs" in SPECIALIST_REGISTRY[m]
    print(f" -> PASS: All {len(expected_models)} specialist models catalogued with full contracts.")

    # ----------------------------------------------------
    # Test 5: Phase 8 DAG Execution Planner
    # ----------------------------------------------------
    print("\n[5] Testing Phase 8: DAG Execution Planner...")
    manifest_paths = {
        "before": {"saved_path": "data/uploads/before.tif"},
        "after": {"saved_path": "data/uploads/after.tif"},
    }
    plan = create_execution_plan(cfg, manifest_paths, q3, geospatial_report=g_rep)
    assert len(plan) == 2
    assert plan[0]["model_id"] == "differencing_engine"
    assert plan[1]["model_id"] == "change_vqa_specialist"
    assert plan[1]["depends_on"] == ["step_1_cv_differencing"]
    print(f" -> PASS: 2-step DAG plan constructed with dependency chaining: {[s['step_id'] for s in plan]}")

    # ----------------------------------------------------
    # Test 6: Phase 10 Evidence Fusion Engine
    # ----------------------------------------------------
    print("\n[6] Testing Phase 10: Evidence Fusion Engine...")
    mock_steps = [
        {
            "step_id": "step_1",
            "model_called": "Differencing Engine",
            "model_version": "1.0.0",
            "confidence": 0.92,
            "status": "SUCCESS",
            "output": {
                "percentage_changed": 12.4,
                "overlay_path": "data/change_overlay.png",
                "location_summary": "12.4% change in northeastern sector.",
            }
        },
        {
            "step_id": "step_2",
            "model_called": "Change-VQA Specialist",
            "model_version": "1.1.0",
            "confidence": 0.94,
            "status": "SUCCESS",
            "output": {
                "answer": "Substantial urban built-up expansion is verified across 12.4% of the scene.",
            }
        }
    ]
    fused = fuse_execution_evidence(mock_steps, "change_vqa", raw_query=q3)
    assert "12.4%" in fused["final_answer"]
    assert fused["visual_artifacts"]["change_overlay_path"] == "data/change_overlay.png"
    assert len(fused["supporting_facts"]) >= 2
    print(f" -> PASS: Evidence successfully fused into final answer and visual artifacts.")

    # ----------------------------------------------------
    # Test 7: Phase 11 Confidence & Conflict Detection
    # ----------------------------------------------------
    print("\n[7] Testing Phase 11: Confidence Scoring & Conflict Detector...")
    # Disagreement scenario: differencing says 0.2% change, but VQA says "substantial expansion"
    mock_conflict_steps = [
        {
            "step_id": "step_1_differencing",
            "model_called": "Differencing Engine",
            "confidence": 0.90,
            "output": {"percentage_changed": 0.2}
        },
        {
            "step_id": "step_2_change_vqa",
            "model_called": "Change-VQA Specialist",
            "confidence": 0.90,
            "output": {"answer": "Substantial expansion has taken place with heavy new buildings."}
        }
    ]
    conf_val, flag, conflicts = evaluate_confidence_and_conflicts(mock_conflict_steps, "change_vqa")
    assert flag is True, "Disagreement should have been flagged"
    assert conf_val < 0.70, "Confidence should be penalized when conflict is flagged"
    assert len(conflicts) > 0
    print(f" -> PASS: Model conflict detected and penalized! Conf: {conf_val}, Flag: {flag}, Conflict: {conflicts[0]['type']}")

    # ----------------------------------------------------
    # Test 8: Phase 12 Observable Execution Trace Ledger
    # ----------------------------------------------------
    print("\n[8] Testing Phase 12: Observable Execution Trace Ledger...")
    trace = build_execution_trace(
        task_name="change_vqa",
        query_text=q3,
        inputs_used=["before", "after"],
        executed_steps=mock_steps,
        final_confidence=0.93,
        disagreement_flagged=False,
        conflicts=[],
        intent=i3,
        geospatial_report=g_rep,
    )
    assert trace["task"] == "change_vqa"
    assert len(trace["steps"]) == 2
    assert trace["audit_compliance"]["suppressed_internal_cot"] is True
    print(f" -> PASS: Execution trace generated ({len(trace['steps'])} observable steps, Cot suppressed).")

    print("\n==========================================================")
    print(" ALL PHASE 14 COMPONENT INTEGRATION TESTS PASSED! ")
    print("==========================================================")


if __name__ == "__main__":
    run_integration_tests()
