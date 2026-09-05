"""
Central Contract and Schema Definitions for SatQuery AI
SIH 2026 Problem Statement 26167 | Team Vyomix

Provides strongly typed Pydantic models for the entire Agentic AI orchestration lifecycle:
- QueryIntent & InputDescriptor
- ValidationResult & TaskRequirement
- SpecialistRequest & SpecialistResult
- EvidenceItem & Conflict
- ExecutionPlan & ExecutionStep
- ConfidenceResult & ExecutionTrace
- FinalResponse
"""
from enum import Enum
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field


# =========================================================================
# Enumerations
# =========================================================================

class TaskType(str, Enum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CHANGE_ANALYSIS = "change_analysis"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR = "optical_sar"


class ModalityType(str, Enum):
    OPTICAL = "OPTICAL"
    SAR = "SAR"
    MULTISPECTRAL = "MULTISPECTRAL"
    UNKNOWN = "UNKNOWN"


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    PENDING_INTEGRATION = "pending_integration"
    UNAVAILABLE = "unavailable"


# =========================================================================
# Query Interpretation & Task Requirements
# =========================================================================

class QueryIntent(BaseModel):
    task: TaskType
    target_entity: Optional[str] = None
    spatial_constraint: Optional[str] = None
    temporal_markers: List[str] = Field(default_factory=list)
    requires_modalities: List[str] = Field(default_factory=lambda: ["OPTICAL"])
    requires_multi_temporal: bool = False
    required_inputs: int = 1
    signals: List[str] = Field(default_factory=list)
    raw_query: str
    confidence: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class InputDescriptor(BaseModel):
    slot: str
    filename: str
    saved_path: str
    modality: str = "UNKNOWN"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dimensions: Optional[Dict[str, int]] = None
    is_georeferenced: bool = False
    crs: Optional[str] = None


class ValidationResult(BaseModel):
    is_valid: bool
    code: Optional[str] = None
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class TaskRequirement(BaseModel):
    task: TaskType
    pipeline_type: str
    min_inputs: int
    max_inputs: int
    required_modalities: List[str]
    temporal_pair_required: bool = False
    slots_assigned: Dict[str, str] = Field(default_factory=dict)
    target_entity: Optional[str] = None
    spatial_constraint: Optional[str] = None
    detected_modalities: Dict[str, str] = Field(default_factory=dict)


# =========================================================================
# Model Registry & Specialist Contracts
# =========================================================================

class SpecialistMetadata(BaseModel):
    id: str
    name: str
    version: str
    status: ModelStatus = ModelStatus.AVAILABLE
    capabilities: List[str] = Field(default_factory=list)
    modalities_supported: List[str] = Field(default_factory=list)
    min_inputs: int = 1
    max_inputs: int = 1
    device_requirements: str = "cpu"
    dependencies: List[str] = Field(default_factory=list)
    estimated_latency_ms: float = 30.0
    fallback_strategy: Optional[str] = None


class SpecialistRequest(BaseModel):
    specialist_id: str
    task: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    source: str
    type: str
    finding: str
    value: Optional[Any] = None
    unit: Optional[str] = None
    location: Optional[str] = None
    artifact: Optional[str] = None
    reliability: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpecialistResult(BaseModel):
    task: str
    status: str = "success"  # "success", "failed", "degraded"
    answer: Optional[str] = None
    confidence: Optional[Dict[str, Any]] = None  # {"value": float, "source": str}
    model: Dict[str, Any] = Field(default_factory=dict)  # {"id": ..., "name": ..., "version": ...}
    evidence: List[EvidenceItem] = Field(default_factory=list)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Ensures compatibility with existing downstream consumers."""
        d = self.model_dump()
        conf_val = None
        if self.confidence and isinstance(self.confidence, dict):
            conf_val = self.confidence.get("value")
        elif isinstance(self.confidence, (float, int)):
            conf_val = float(self.confidence)
        d["confidence"] = conf_val
        return d


# =========================================================================
# Planning & Execution
# =========================================================================

class ExecutionStep(BaseModel):
    step_id: str
    specialist_id: str
    model_called: str
    model_version: str = "1.0.0"
    description: str = ""
    depends_on: List[str] = Field(default_factory=list)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    status: ExecutionStatus = ExecutionStatus.PENDING
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    confidence: Optional[float] = None
    output: Optional[Union[SpecialistResult, Dict[str, Any]]] = None


class ExecutionPlan(BaseModel):
    plan_id: str
    task: str
    steps: List[ExecutionStep] = Field(default_factory=list)
    estimated_latency_ms: float = 0.0


# =========================================================================
# Evidence Fusion, Conflicts & Confidence
# =========================================================================

class Conflict(BaseModel):
    conflict_id: str
    type: str
    description: str
    source_a: str
    source_b: str
    finding_a: Optional[str] = None
    finding_b: Optional[str] = None
    severity: str = "warning"  # "info", "warning", "critical"


class ConfidenceResult(BaseModel):
    value: Optional[float] = None  # None if genuinely unavailable
    source: str = "unavailable"  # "model", "weighted_evidence_aggregation", "unavailable"
    method: Optional[str] = None
    components: Dict[str, Any] = Field(default_factory=dict)


# =========================================================================
# Trace & Final Response
# =========================================================================

class ExecutionTrace(BaseModel):
    query: str
    task: str
    execution_timestamp: str
    inputs: List[str] = Field(default_factory=list)
    models_called: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    final_confidence: Optional[float] = None
    disagreement_flagged: bool = False
    conflicts_detected: List[Conflict] = Field(default_factory=list)
    audit_compliance: Dict[str, Any] = Field(default_factory=dict)


class FinalResponse(BaseModel):
    status: str = "success"  # "success", "failed", "degraded"
    query_id: str
    upload_id: str
    query_text: str
    timestamp: str
    task: str
    answer: str
    confidence: Optional[float] = None
    confidence_breakdown: Optional[ConfidenceResult] = None
    disagreement_flagged: bool = False
    conflicts: List[Conflict] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    visual_artifacts: Dict[str, Any] = Field(default_factory=dict)
    execution_summary: Dict[str, Any] = Field(default_factory=dict)
    execution_trace: Optional[Dict[str, Any]] = None
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
