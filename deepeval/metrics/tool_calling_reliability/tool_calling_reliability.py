from typing import Any, Dict, List, Optional

from deepeval.metrics import BaseMetric
from deepeval.metrics.indicator import metric_progress_indicator
from deepeval.metrics.utils import (
    check_llm_test_case_params,
    construct_verbose_logs,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall

from .schema import FailureAttribution, ToolCallingReliabilityResult


SUCCESS_STATUSES = {"success", "fallback_success"}
TIMEOUT_STATUSES = {"timeout"}
FAILED_STATUSES = {"failed", "error", "fallback_failed"}


class ToolCallingReliabilityMetric(BaseMetric):
    _required_params: List[SingleTurnParams] = [
        SingleTurnParams.INPUT,
        SingleTurnParams.TOOLS_CALLED,
        SingleTurnParams.EXPECTED_TOOLS,
    ]

    def __init__(
        self,
        threshold: float = 0.8,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        require_exact_final_answer: bool = True,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.strict_mode = strict_mode
        self.verbose_mode = verbose_mode
        self.require_exact_final_answer = require_exact_final_answer
        self.evaluation_cost = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
        _log_metric_to_confident: bool = True,
    ) -> float:
        check_llm_test_case_params(
            test_case,
            self._required_params,
            None,
            None,
            self,
            None,
            test_case.multimodal,
        )

        self.evaluation_cost = 0
        self.input_tokens = 0
        self.output_tokens = 0

        with metric_progress_indicator(
            self, _show_indicator=_show_indicator, _in_component=_in_component
        ):
            result = self.evaluate_case(test_case)
            self.result = result
            self.score_breakdown = self._result_to_dict(result)
            self.score = self._calculate_score(result)
            self.score = (
                0
                if self.strict_mode and self.score < self.threshold
                else self.score
            )
            self.success = self.score >= self.threshold
            self.reason = result.reason

            if self.verbose_mode:
                self.verbose_logs = construct_verbose_logs(
                    self,
                    steps=[
                        f"Tool Selection Accuracy: {result.tool_selection_accuracy:.2f}",
                        f"Argument Match Rate: {result.argument_match_rate:.2f}",
                        f"Execution Success Rate: {result.execution_success_rate:.2f}",
                        f"Timeout Rate: {result.timeout_rate:.2f}",
                        f"Retry Rate: {result.retry_rate:.2f}",
                        f"Fallback Rate: {result.fallback_rate:.2f}",
                        f"Final Task Success Rate: {result.final_task_success_rate:.2f}",
                        f"Failure Attributions: {[item.value for item in result.failure_attributions]}",
                        f"Final Score: {self.score:.2f}",
                    ],
                )

            return self.score

    async def a_measure(
        self,
        test_case: LLMTestCase,
        _show_indicator: bool = True,
        _in_component: bool = False,
    ) -> float:
        return self.measure(
            test_case,
            _show_indicator=_show_indicator,
            _in_component=_in_component,
        )

    def evaluate_case(
        self, test_case: LLMTestCase
    ) -> ToolCallingReliabilityResult:
        expected_tools = test_case.expected_tools or []
        tools_called = test_case.tools_called or []
        traces = self._extract_tool_traces(test_case)

        tool_selection_accuracy = self._tool_selection_accuracy(
            expected_tools, tools_called
        )
        argument_match_rate = self._argument_match_rate(
            expected_tools, tools_called
        )
        execution_success_rate = self._execution_success_rate(traces)
        timeout_rate = self._rate_matching(traces, self._is_timeout)
        retry_rate = self._rate_matching(traces, self._has_retry)
        fallback_rate = self._rate_matching(traces, self._uses_fallback)
        avg_latency_ms = self._avg_latency_ms(traces)
        final_task_success_rate = (
            1.0 if self._final_answer_matches(test_case) else 0.0
        )
        attributions = self._attribute_failures(
            expected_tools=expected_tools,
            tools_called=tools_called,
            traces=traces,
            tool_selection_accuracy=tool_selection_accuracy,
            argument_match_rate=argument_match_rate,
            execution_success_rate=execution_success_rate,
            final_task_success_rate=final_task_success_rate,
        )

        reason = self._build_reason(
            tool_selection_accuracy=tool_selection_accuracy,
            argument_match_rate=argument_match_rate,
            execution_success_rate=execution_success_rate,
            final_task_success_rate=final_task_success_rate,
            attributions=attributions,
        )

        return ToolCallingReliabilityResult(
            tool_selection_accuracy=tool_selection_accuracy,
            argument_match_rate=argument_match_rate,
            execution_success_rate=execution_success_rate,
            timeout_rate=timeout_rate,
            retry_rate=retry_rate,
            fallback_rate=fallback_rate,
            final_task_success_rate=final_task_success_rate,
            avg_latency_ms=avg_latency_ms,
            failure_attributions=attributions,
            reason=reason,
        )

    def _tool_selection_accuracy(
        self, expected_tools: List[ToolCall], tools_called: List[ToolCall]
    ) -> float:
        if not expected_tools and not tools_called:
            return 1.0
        if expected_tools and not tools_called:
            return 0.0
        expected_names = {tool.name for tool in expected_tools}
        called_names = {tool.name for tool in tools_called}
        if not expected_names:
            return 0.0
        return len(expected_names.intersection(called_names)) / len(
            expected_names
        )

    def _argument_match_rate(
        self, expected_tools: List[ToolCall], tools_called: List[ToolCall]
    ) -> float:
        if not expected_tools and not tools_called:
            return 1.0
        if not expected_tools:
            return 0.0
        matched_scores = []
        used_indexes = set()
        for expected in expected_tools:
            best_score = 0.0
            best_index = None
            for index, called in enumerate(tools_called):
                if index in used_indexes or called.name != expected.name:
                    continue
                score = self._compare_dicts(
                    expected.input_parameters or {},
                    called.input_parameters or {},
                )
                if score > best_score:
                    best_score = score
                    best_index = index
            if best_index is not None:
                used_indexes.add(best_index)
            matched_scores.append(best_score)
        return sum(matched_scores) / len(expected_tools)

    def _compare_dicts(self, expected: Dict, actual: Dict) -> float:
        if expected == actual:
            return 1.0
        all_keys = set(expected.keys()).union(set(actual.keys()))
        if not all_keys:
            return 1.0
        matched = 0.0
        for key in all_keys:
            if key not in expected or key not in actual:
                continue
            if expected[key] == actual[key]:
                matched += 1
            elif isinstance(expected[key], dict) and isinstance(
                actual[key], dict
            ):
                matched += self._compare_dicts(expected[key], actual[key])
        return matched / len(all_keys)

    def _execution_success_rate(self, traces: List[Dict[str, Any]]) -> float:
        if not traces:
            return 0.0
        successful = sum(1 for trace in traces if self._is_success(trace))
        return successful / len(traces)

    def _rate_matching(self, traces, predicate) -> float:
        if not traces:
            return 0.0
        return sum(1 for trace in traces if predicate(trace)) / len(traces)

    def _is_success(self, trace: Dict[str, Any]) -> bool:
        status = str(trace.get("status", "")).lower()
        return status in SUCCESS_STATUSES or trace.get("success") is True

    def _is_timeout(self, trace: Dict[str, Any]) -> bool:
        status = str(trace.get("status", "")).lower()
        root_status = str(trace.get("root_status", "")).lower()
        error = str(trace.get("error", "")).lower()
        return (
            status in TIMEOUT_STATUSES
            or root_status in TIMEOUT_STATUSES
            or "timed out" in error
            or "timeout" in error
        )

    def _is_execution_failure(self, trace: Dict[str, Any]) -> bool:
        status = str(trace.get("status", "")).lower()
        return status in FAILED_STATUSES and not self._is_success(trace)

    def _has_retry(self, trace: Dict[str, Any]) -> bool:
        return self._as_int(trace.get("retry_count")) > 0 or self._as_int(
            trace.get("attempts")
        ) > 1

    def _uses_fallback(self, trace: Dict[str, Any]) -> bool:
        return bool(trace.get("fallback_used")) or bool(
            trace.get("fallback_tool")
        )

    def _as_int(self, value: Optional[Any]) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    def _avg_latency_ms(self, traces: List[Dict[str, Any]]) -> float:
        latencies = [
            float(trace.get("latency_ms") or 0.0)
            for trace in traces
            if trace.get("latency_ms") is not None
        ]
        return sum(latencies) / len(latencies) if latencies else 0.0

    def _final_answer_matches(self, test_case: LLMTestCase) -> bool:
        if not self.require_exact_final_answer:
            return True
        if test_case.expected_output is None:
            return True
        return (test_case.actual_output or "").strip() == (
            test_case.expected_output or ""
        ).strip()

    def _attribute_failures(
        self,
        expected_tools: List[ToolCall],
        tools_called: List[ToolCall],
        traces: List[Dict[str, Any]],
        tool_selection_accuracy: float,
        argument_match_rate: float,
        execution_success_rate: float,
        final_task_success_rate: float,
    ) -> List[FailureAttribution]:
        attributions = []
        if expected_tools and not tools_called:
            attributions.append(FailureAttribution.NO_TOOL_CALLED)
        elif tool_selection_accuracy < 1:
            attributions.append(FailureAttribution.WRONG_TOOL)
        if self._has_argument_mismatch(expected_tools, tools_called):
            attributions.append(FailureAttribution.WRONG_ARGUMENTS)
        if any(self._is_timeout(trace) for trace in traces):
            attributions.append(FailureAttribution.TOOL_TIMEOUT)
        if any(self._is_execution_failure(trace) for trace in traces):
            attributions.append(FailureAttribution.TOOL_EXECUTION_FAILED)
        elif execution_success_rate < 1 and traces:
            attributions.append(FailureAttribution.TOOL_EXECUTION_FAILED)
        if any(self._uses_fallback(trace) for trace in traces):
            attributions.append(FailureAttribution.FALLBACK_USED)
        if any(
            str(trace.get("status", "")).lower() == "fallback_failed"
            for trace in traces
        ):
            attributions.append(FailureAttribution.FALLBACK_FAILED)
        if (
            final_task_success_rate < 1
            and execution_success_rate == 1
            and tool_selection_accuracy == 1
            and argument_match_rate == 1
        ):
            attributions.append(
                FailureAttribution.FINAL_ANSWER_FAILED_AFTER_TOOL_SUCCESS
            )
        return attributions

    def _has_argument_mismatch(
        self, expected_tools: List[ToolCall], tools_called: List[ToolCall]
    ) -> bool:
        for expected in expected_tools:
            for called in tools_called:
                if called.name != expected.name:
                    continue
                if self._compare_dicts(
                    expected.input_parameters or {},
                    called.input_parameters or {},
                ) < 1:
                    return True
        return False

    def _extract_tool_traces(
        self, test_case: LLMTestCase
    ) -> List[Dict[str, Any]]:
        metadata = test_case.metadata or {}
        traces = metadata.get("tool_traces") or []
        normalized = []
        for trace in traces:
            if hasattr(trace, "model_dump"):
                normalized.append(trace.model_dump())
            elif hasattr(trace, "dict"):
                normalized.append(trace.dict())
            elif isinstance(trace, dict):
                normalized.append(trace)
            else:
                normalized.append(
                    {
                        key: getattr(trace, key)
                        for key in dir(trace)
                        if not key.startswith("_")
                        and not callable(getattr(trace, key))
                    }
                )
        return normalized

    def _build_reason(
        self,
        tool_selection_accuracy: float,
        argument_match_rate: float,
        execution_success_rate: float,
        final_task_success_rate: float,
        attributions: List[FailureAttribution],
    ) -> str:
        attribution_text = (
            ", ".join(item.value for item in attributions)
            if attributions
            else "NONE"
        )
        return (
            "Deterministic reliability score computed from tool calls, "
            f"tool traces, and final answer match. Selection={tool_selection_accuracy:.2f}, "
            f"Arguments={argument_match_rate:.2f}, Execution={execution_success_rate:.2f}, "
            f"Final={final_task_success_rate:.2f}. Attributions={attribution_text}."
        )

    def _calculate_score(self, result: ToolCallingReliabilityResult) -> float:
        score = (
            result.tool_selection_accuracy
            + result.argument_match_rate
            + result.execution_success_rate
            + result.final_task_success_rate
        ) / 4
        return max(0.0, min(1.0, score))

    def _result_to_dict(self, result: ToolCallingReliabilityResult) -> Dict:
        return (
            result.model_dump()
            if hasattr(result, "model_dump")
            else result.dict()
        )

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except Exception:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Tool Calling Reliability"