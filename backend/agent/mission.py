"""
Mission / Investigation Context Module for SatQuery AI
SIH Problem Statement 26167 | Team Vyomix

Supports multi-turn analysis sessions where follow-up queries
reuse cached intermediate results from prior analysis steps.
"""
import uuid
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

logger = logging.getLogger("satquery.agent.mission")

# In-memory session store
_MISSION_STORE: Dict[str, "Mission"] = {}


class Mission:
    """
    A mission holds the objective, uploaded data references, analysis history,
    and cached results for multi-turn interaction.
    """

    def __init__(self, upload_id: str, objective: Optional[str] = None):
        self.mission_id = str(uuid.uuid4())
        self.upload_id = upload_id
        self.objective = objective or "Remote sensing analysis"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.queries: List[Dict[str, Any]] = []
        self.cached_results: Dict[str, Any] = {}
        self.analysis_history: List[Dict[str, Any]] = []

    def add_query_result(self, query_text: str, response: Dict[str, Any]) -> None:
        """Records a query and its response in the mission history."""
        entry = {
            "query_text": query_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": response.get("task"),
            "query_id": response.get("query_id"),
        }
        self.queries.append(entry)

        # Cache key results for follow-up reuse
        task = response.get("task", "")
        if "execution_trace" in response:
            self.cached_results["last_trace"] = response["execution_trace"]
        if response.get("visual_artifacts"):
            self.cached_results["last_visual_artifacts"] = response["visual_artifacts"]
        if response.get("evidence"):
            self.cached_results["last_evidence"] = response["evidence"]
        if response.get("confidence") is not None:
            self.cached_results["last_confidence"] = response["confidence"]

        # Task-specific caching
        if task in ("change_vqa", "change_analysis"):
            self.cached_results["change_result"] = response
        elif task == "grounding":
            self.cached_results["grounding_result"] = response
        elif task in ("single_image_vqa", "vqa"):
            self.cached_results["vqa_result"] = response
        elif task == "captioning":
            self.cached_results["captioning_result"] = response
        elif task in ("optical_sar_fusion", "optical_sar"):
            self.cached_results["fusion_result"] = response

        self.analysis_history.append({
            "query": query_text,
            "task": task,
            "answer_preview": str(response.get("answer", ""))[:150],
            "confidence": response.get("confidence"),
            "timestamp": entry["timestamp"],
        })

    def get_context_summary(self) -> Dict[str, Any]:
        """Returns a summary of mission context for the agent planner."""
        return {
            "mission_id": self.mission_id,
            "upload_id": self.upload_id,
            "objective": self.objective,
            "total_queries": len(self.queries),
            "cached_tasks": list(self.cached_results.keys()),
            "history": self.analysis_history[-5:],  # Last 5 entries
        }

    def has_cached(self, key: str) -> bool:
        return key in self.cached_results

    def get_cached(self, key: str) -> Optional[Any]:
        return self.cached_results.get(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "upload_id": self.upload_id,
            "objective": self.objective,
            "created_at": self.created_at,
            "total_queries": len(self.queries),
            "analysis_history": self.analysis_history,
        }


def create_mission(upload_id: str, objective: Optional[str] = None) -> Mission:
    """Creates and stores a new mission."""
    mission = Mission(upload_id=upload_id, objective=objective)
    _MISSION_STORE[mission.mission_id] = mission
    logger.info(f"[Mission] Created mission {mission.mission_id} for upload {upload_id}")
    return mission


def get_mission(mission_id: str) -> Optional[Mission]:
    """Retrieves an existing mission by ID."""
    return _MISSION_STORE.get(mission_id)


def list_missions() -> List[Dict[str, Any]]:
    """Lists all active missions."""
    return [m.to_dict() for m in _MISSION_STORE.values()]
