# Agent Tool-Calling Reliability Evaluation

This example runs a deterministic baseline/enhanced benchmark for agent tool-calling reliability. It is designed to cover runtime failure modes without paying for 50 live LLM calls.

The pipeline uses `LLMTestCase.expected_tools`, `LLMTestCase.tools_called`, and `metadata["tool_traces"]`:

```python
LLMTestCase(
    input="帮我查北京到上海最便宜的航班",
    actual_output=response.messages[-1]["content"],
    expected_tools=[
        ToolCall(
            name="search_flights",
            input_parameters={"from": "北京", "to": "上海"},
        )
    ],
    tools_called=response.tools_called,
    metadata={"tool_traces": response.tool_traces},
)
```

Run:

```bash
python examples/agent_tool_reliability/run.py
```

The script writes `report.md` with measured task success, tool execution success, timeout failures, fallback recovery, latency, and failure attribution counts.

## Benchmark Summary

| Metric | Baseline | Enhanced |
| --- | ---: | ---: |
| Task success rate | 68.0% | 92.0% |
| Tool execution success | 76.0% | 92.0% |
| Timeout failures | 8 | 3 |
| Fallback recovered cases | 0 | 5 |
| Avg latency | 507 ms | 483 ms |

`TOOL_TIMEOUT` is treated as a root-cause attribution: it records that a timeout happened in the trace. This is separate from final timeout failures, because an enhanced run can time out on the primary tool and still recover through fallback.

The enhanced average latency is lower because baseline cases include long-tail requests that wait until full timeout before failing. The enhanced path cuts those failures short and recovers some of them through faster fallback calls.

## Difference from existing metrics

`ToolCorrectnessMetric` focuses on whether the expected tools and arguments were used. `ToolCallingReliabilityMetric` complements it with deterministic runtime reliability signals: timeout rate, retry rate, fallback rate, execution success, latency, and 8-stage failure attribution.
