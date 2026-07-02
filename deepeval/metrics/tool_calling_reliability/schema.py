from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class FailureAttribution(str, Enum):
    NO_TOOL_CALLED = "NO_TOOL_CALLED"
    WRONG_TOOL = "WRONG_TOOL"
    WRONG_ARGUMENTS = "WRONG_ARGUMENTS"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    FALLBACK_USED = "FALLBACK_USED"
    FALLBACK_FAILED = "FALLBACK_FAILED"
    FINAL_ANSWER_FAILED_AFTER_TOOL_SUCCESS = (
        "FINAL_ANSWER_FAILED_AFTER_TOOL_SUCCESS"
    )


class ToolCallingReliabilityResult(BaseModel):
    tool_selection_accuracy: float = 0.0
    argument_match_rate: float = 0.0
    execution_success_rate: float = 0.0
    timeout_rate: float = 0.0
    retry_rate: float = 0.0
    fallback_rate: float = 0.0
    final_task_success_rate: float = 0.0
    avg_latency_ms: float = 0.0
    failure_attributions: List[FailureAttribution] = Field(
        default_factory=list
    )
    reason: Optional[str] = None