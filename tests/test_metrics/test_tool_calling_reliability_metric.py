import os
import pytest

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from deepeval.metrics import ToolCallingReliabilityMetric
from deepeval.metrics.tool_calling_reliability import FailureAttribution
from deepeval.test_case import LLMTestCase, ToolCall


def _case(
    *,
    expected_tool=None,
    called_tool=None,
    expected_output="done",
    actual_output="done",
    traces=None,
):
    return LLMTestCase(
        input="Find the cheapest flight from Beijing to Shanghai",
        actual_output=actual_output,
        expected_output=expected_output,
        expected_tools=[expected_tool]
        if expected_tool is not None
        else [
            ToolCall(
                name="search_flights",
                input_parameters={"from": "Beijing", "to": "Shanghai"},
            )
        ],
        tools_called=[] if called_tool is None else [called_tool],
        metadata={"tool_traces": traces or []},
    )


def _measure(test_case):
    metric = ToolCallingReliabilityMetric(threshold=0.8)
    score = metric.measure(test_case, _show_indicator=False)
    return metric, score


def test_perfect_tool_calling_reliability_score():
    tool = ToolCall(
        name="search_flights",
        input_parameters={"from": "Beijing", "to": "Shanghai"},
    )
    metric, score = _measure(
        _case(
            called_tool=tool,
            traces=[
                {
                    "tool_name": "search_flights",
                    "status": "success",
                    "latency_ms": 120,
                }
            ],
        )
    )

    assert score == pytest.approx(1.0)
    assert metric.result.tool_selection_accuracy == pytest.approx(1.0)
    assert metric.result.argument_match_rate == pytest.approx(1.0)
    assert metric.result.execution_success_rate == pytest.approx(1.0)
    assert metric.result.final_task_success_rate == pytest.approx(1.0)
    assert metric.result.avg_latency_ms == pytest.approx(120)
    assert metric.result.failure_attributions == []
    assert metric.evaluation_cost == 0


def test_no_tool_called_attribution():
    metric, score = _measure(
        _case(actual_output="failed", traces=[])
    )

    assert score == pytest.approx(0.0)
    assert metric.result.failure_attributions == [
        FailureAttribution.NO_TOOL_CALLED
    ]


def test_wrong_tool_does_not_double_count_wrong_arguments():
    metric, score = _measure(
        _case(
            called_tool=ToolCall(
                name="search_hotels",
                input_parameters={"city": "Shanghai"},
            ),
            actual_output="failed",
            traces=[{"tool_name": "search_hotels", "status": "success"}],
        )
    )

    assert score == pytest.approx(0.25)
    assert FailureAttribution.WRONG_TOOL in metric.result.failure_attributions
    assert (
        FailureAttribution.WRONG_ARGUMENTS
        not in metric.result.failure_attributions
    )


def test_wrong_arguments_attribution():
    expected_tool = ToolCall(
        name="search_flights",
        input_parameters={"from": "Beijing", "to": "Shanghai"},
    )
    called_tool = ToolCall(
        name="search_flights",
        input_parameters={"from": "Beijing", "to": "Hangzhou"},
    )
    metric, _ = _measure(
        _case(
            expected_tool=expected_tool,
            called_tool=called_tool,
            actual_output="failed",
            traces=[{"tool_name": "search_flights", "status": "success"}],
        )
    )

    assert metric.result.argument_match_rate == pytest.approx(0.5)
    assert FailureAttribution.WRONG_ARGUMENTS in metric.result.failure_attributions


def test_timeout_and_fallback_attribution():
    tool = ToolCall(
        name="search_flights",
        input_parameters={"from": "Beijing", "to": "Shanghai"},
    )
    metric, score = _measure(
        _case(
            called_tool=tool,
            traces=[
                {
                    "tool_name": "search_flights",
                    "status": "timeout",
                    "error": "request timeout",
                    "retry_count": 1,
                    "latency_ms": 900,
                },
                {
                    "tool_name": "cached_flights",
                    "status": "fallback_success",
                    "fallback_used": True,
                    "fallback_tool": "cached_flights",
                    "latency_ms": 150,
                },
            ],
        )
    )

    assert score == pytest.approx(0.875)
    assert metric.result.timeout_rate == pytest.approx(0.5)
    assert metric.result.retry_rate == pytest.approx(0.5)
    assert metric.result.fallback_rate == pytest.approx(0.5)
    assert FailureAttribution.TOOL_TIMEOUT in metric.result.failure_attributions
    assert FailureAttribution.FALLBACK_USED in metric.result.failure_attributions


def test_fallback_failed_attribution():
    tool = ToolCall(
        name="search_flights",
        input_parameters={"from": "Beijing", "to": "Shanghai"},
    )
    metric, _ = _measure(
        _case(
            called_tool=tool,
            actual_output="failed",
            traces=[
                {
                    "tool_name": "cached_flights",
                    "status": "fallback_failed",
                    "fallback_used": True,
                    "fallback_tool": "cached_flights",
                    "latency_ms": 300,
                }
            ],
        )
    )

    assert FailureAttribution.TOOL_EXECUTION_FAILED in metric.result.failure_attributions
    assert FailureAttribution.FALLBACK_USED in metric.result.failure_attributions
    assert FailureAttribution.FALLBACK_FAILED in metric.result.failure_attributions


def test_final_answer_failed_after_tool_success_attribution():
    tool = ToolCall(
        name="search_flights",
        input_parameters={"from": "Beijing", "to": "Shanghai"},
    )
    metric, score = _measure(
        _case(
            called_tool=tool,
            actual_output="wrong answer",
            traces=[{"tool_name": "search_flights", "status": "success"}],
        )
    )

    assert score == pytest.approx(0.75)
    assert (
        FailureAttribution.FINAL_ANSWER_FAILED_AFTER_TOOL_SUCCESS
        in metric.result.failure_attributions
    )


def test_timeout_root_cause_can_be_recovered_by_fallback():
    tool = ToolCall(
        name="search_flights",
        input_parameters={"from": "Beijing", "to": "Shanghai"},
    )
    metric, score = _measure(
        _case(
            called_tool=tool,
            traces=[
                {
                    "tool_name": "search_flights",
                    "status": "fallback_success",
                    "root_status": "timeout",
                    "fallback_used": True,
                    "fallback_tool": "cached_flights",
                    "latency_ms": 220,
                }
            ],
        )
    )

    assert score == pytest.approx(1.0)
    assert metric.result.final_task_success_rate == pytest.approx(1.0)
    assert metric.result.timeout_rate == pytest.approx(1.0)
    assert FailureAttribution.TOOL_TIMEOUT in metric.result.failure_attributions
    assert FailureAttribution.FALLBACK_USED in metric.result.failure_attributions
    assert (
        FailureAttribution.FINAL_ANSWER_FAILED_AFTER_TOOL_SUCCESS
        not in metric.result.failure_attributions
    )
