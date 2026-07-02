from __future__ import annotations

from collections import Counter
from pathlib import Path
import os
import sys
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepeval.metrics import ToolCallingReliabilityMetric
from deepeval.metrics.tool_calling_reliability import FailureAttribution
from deepeval.test_case import LLMTestCase, ToolCall


TOOLS = [
    (
        "search_flights",
        {"from": "Beijing", "to": "Shanghai", "date": "2026-07-20"},
    ),
    ("get_weather", {"city": "Beijing", "date": "2026-07-20"}),
    ("lookup_order", {"order_id": "A-10086"}),
    ("convert_currency", {"amount": 100, "from": "USD", "to": "CNY"}),
]


def expected_tool_for(index: int) -> ToolCall:
    name, params = TOOLS[index % len(TOOLS)]
    params = dict(params)
    if name == "lookup_order":
        params["order_id"] = f"A-{10000 + index}"
    return ToolCall(name=name, input_parameters=params)


def expected_answer(index: int) -> str:
    return f"case-{index:02d}: task completed"


def trace(
    tool: ToolCall,
    *,
    status: str,
    latency_ms: int,
    root_status: str | None = None,
    retry_count: int = 0,
    attempts: int = 1,
    fallback_used: bool = False,
    fallback_tool: str | None = None,
    error: str | None = None,
) -> Dict:
    result = {
        "tool_name": tool.name,
        "status": status,
        "latency_ms": latency_ms,
        "retry_count": retry_count,
        "attempts": attempts,
        "fallback_used": fallback_used,
    }
    if root_status is not None:
        result["root_status"] = root_status
    if fallback_tool is not None:
        result["fallback_tool"] = fallback_tool
    if error is not None:
        result["error"] = error
    return result


def case_from_outcome(index: int, variant: str, outcome: str) -> LLMTestCase:
    expected_tool = expected_tool_for(index)
    expected_output = expected_answer(index)
    called_tool = expected_tool
    actual_output = expected_output
    traces: List[Dict] = []

    if outcome == "ok":
        traces = [trace(expected_tool, status="success", latency_ms=410)]
    elif outcome == "timeout_fail":
        actual_output = "I could not complete the request."
        traces = [
            trace(
                expected_tool,
                status="timeout",
                latency_ms=1000,
                error="request timeout",
            )
        ]
    elif outcome == "execution_failed":
        actual_output = "I could not complete the request."
        traces = [
            trace(
                expected_tool,
                status="failed",
                latency_ms=620,
                error="upstream 500",
            )
        ]
    elif outcome == "wrong_arguments":
        actual_output = "I searched the wrong route."
        wrong_params = dict(expected_tool.input_parameters or {})
        wrong_params[next(iter(wrong_params))] = "WRONG_VALUE"
        called_tool = ToolCall(
            name=expected_tool.name,
            input_parameters=wrong_params,
        )
        traces = [trace(called_tool, status="success", latency_ms=390)]
    elif outcome == "wrong_tool":
        actual_output = "I used the wrong tool."
        called_tool = ToolCall(
            name="search_hotels",
            input_parameters={"city": "Shanghai"},
        )
        traces = [trace(called_tool, status="success", latency_ms=380)]
    elif outcome == "no_tool":
        actual_output = "I answered without calling the required tool."
        called_tool = None
    elif outcome == "fallback_recovered":
        traces = [
            trace(
                expected_tool,
                status="fallback_success",
                root_status="timeout",
                latency_ms=720,
                fallback_used=True,
                fallback_tool=f"cached_{expected_tool.name}",
                error="primary request timeout",
            )
        ]
    elif outcome == "retry_recovered":
        traces = [
            trace(
                expected_tool,
                status="success",
                latency_ms=650,
                retry_count=1,
                attempts=2,
            )
        ]
    else:
        raise ValueError(f"Unknown outcome: {outcome}")

    return LLMTestCase(
        input=f"case-{index:02d}: run {expected_tool.name}",
        actual_output=actual_output,
        expected_output=expected_output,
        expected_tools=[expected_tool],
        tools_called=[] if called_tool is None else [called_tool],
        metadata={
            "variant": variant,
            "scenario": outcome,
            "tool_traces": traces,
        },
    )


def build_cases(variant: str) -> List[LLMTestCase]:
    if variant == "baseline":
        outcomes = (
            ["ok"] * 34
            + ["timeout_fail"] * 8
            + ["execution_failed"] * 3
            + ["wrong_arguments"] * 2
            + ["wrong_tool"] * 2
            + ["no_tool"]
        )
    elif variant == "enhanced":
        outcomes = (
            ["ok"] * 34
            + ["fallback_recovered"] * 5
            + ["timeout_fail"] * 3
            + ["retry_recovered"] * 3
            + ["ok"] * 4
            + ["no_tool"]
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return [case_from_outcome(i, variant, outcome) for i, outcome in enumerate(outcomes)]


def evaluate_cases(cases: Iterable[LLMTestCase]) -> List[Tuple[LLMTestCase, Dict]]:
    rows = []
    for test_case in cases:
        metric = ToolCallingReliabilityMetric(threshold=0.8)
        metric.measure(test_case, _show_indicator=False)
        rows.append((test_case, metric.score_breakdown))
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def summarize(rows: List[Tuple[LLMTestCase, Dict]]) -> Dict:
    total = len(rows)
    final_success = sum(row[1]["final_task_success_rate"] for row in rows)
    execution_success = sum(row[1]["execution_success_rate"] for row in rows)
    selection = sum(row[1]["tool_selection_accuracy"] for row in rows)
    arguments = sum(row[1]["argument_match_rate"] for row in rows)
    retry = sum(row[1]["retry_rate"] for row in rows)
    fallback = sum(row[1]["fallback_rate"] for row in rows)
    latency = sum(row[1]["avg_latency_ms"] for row in rows)

    attribution_counts = Counter()
    for _, result in rows:
        for attribution in result["failure_attributions"]:
            attribution_counts[str(attribution).split(".")[-1]] += 1

    timeout_failures = sum(
        1
        for _, result in rows
        if FailureAttribution.TOOL_TIMEOUT in result["failure_attributions"]
        and result["final_task_success_rate"] < 1
    )
    fallback_recovered = sum(
        1
        for _, result in rows
        if FailureAttribution.FALLBACK_USED in result["failure_attributions"]
        and result["final_task_success_rate"] == 1
    )

    return {
        "total": total,
        "task_success_rate": final_success / total,
        "tool_execution_success_rate": execution_success / total,
        "tool_selection_accuracy": selection / total,
        "argument_match_rate": arguments / total,
        "retry_rate": retry / total,
        "fallback_rate": fallback / total,
        "avg_latency_ms": latency / total,
        "timeout_failures": timeout_failures,
        "fallback_recovered_cases": fallback_recovered,
        "failure_attributions": attribution_counts,
    }


def delta(current, previous) -> str:
    if isinstance(current, int):
        change = current - previous
        return f"{change:+d}"
    change = current - previous
    return f"{change:+.1f}"


def report_markdown(baseline: Dict, enhanced: Dict) -> str:
    latency_delta = enhanced["avg_latency_ms"] - baseline["avg_latency_ms"]
    lines = [
        "# Agent Tool-Calling Reliability Report",
        "",
        "Deterministic mock benchmark: 50 paired cases, no LLM/API calls.",
        "",
        "| Metric | Baseline | Enhanced | Delta |",
        "| --- | ---: | ---: | ---: |",
        f"| Total cases | {baseline['total']} | {enhanced['total']} | 0 |",
        f"| Task success rate | {pct(baseline['task_success_rate'])} | {pct(enhanced['task_success_rate'])} | {pct(enhanced['task_success_rate'] - baseline['task_success_rate'])} |",
        f"| Tool execution success | {pct(baseline['tool_execution_success_rate'])} | {pct(enhanced['tool_execution_success_rate'])} | {pct(enhanced['tool_execution_success_rate'] - baseline['tool_execution_success_rate'])} |",
        f"| Tool selection accuracy | {pct(baseline['tool_selection_accuracy'])} | {pct(enhanced['tool_selection_accuracy'])} | {pct(enhanced['tool_selection_accuracy'] - baseline['tool_selection_accuracy'])} |",
        f"| Argument match rate | {pct(baseline['argument_match_rate'])} | {pct(enhanced['argument_match_rate'])} | {pct(enhanced['argument_match_rate'] - baseline['argument_match_rate'])} |",
        f"| Timeout failures | {baseline['timeout_failures']} | {enhanced['timeout_failures']} | {delta(enhanced['timeout_failures'], baseline['timeout_failures'])} |",
        f"| Fallback recovered cases | {baseline['fallback_recovered_cases']} | {enhanced['fallback_recovered_cases']} | {delta(enhanced['fallback_recovered_cases'], baseline['fallback_recovered_cases'])} |",
        f"| Avg latency | {baseline['avg_latency_ms']:.0f} ms | {enhanced['avg_latency_ms']:.0f} ms | {latency_delta:+.0f} ms |",
        "",
        "## Failure Attribution Counts",
        "",
        "| Attribution | Baseline | Enhanced |",
        "| --- | ---: | ---: |",
    ]

    all_keys = sorted(
        set(baseline["failure_attributions"]).union(
            enhanced["failure_attributions"]
        )
    )
    for key in all_keys:
        lines.append(
            f"| {key} | {baseline['failure_attributions'].get(key, 0)} | {enhanced['failure_attributions'].get(key, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- ToolCorrectnessMetric checks whether expected tools and arguments were used.",
            "- TOOL_TIMEOUT is a root-cause attribution, not necessarily a final task failure; fallback can recover a timed-out primary tool call.",
            "- Enhanced latency is lower because timeout cutoffs and fallback remove several long-tail baseline failures.",
            "- ToolCallingReliabilityMetric is deterministic and trace-based: it adds timeout, retry, fallback, latency, runtime success, and root-cause attribution signals.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    baseline_rows = evaluate_cases(build_cases("baseline"))
    enhanced_rows = evaluate_cases(build_cases("enhanced"))
    baseline = summarize(baseline_rows)
    enhanced = summarize(enhanced_rows)
    report = report_markdown(baseline, enhanced)
    output_path = Path(__file__).with_name("report.md")
    output_path.write_text(report, encoding="utf-8", newline="\n")
    print(report)
    print("Report written to: examples/agent_tool_reliability/report.md")


if __name__ == "__main__":
    main()
