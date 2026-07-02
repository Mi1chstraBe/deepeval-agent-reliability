# Agent Tool-Calling Reliability Evaluation

This example evaluates a paired baseline/enhanced agent run with deterministic mock cases. It is designed to validate reliability behavior without paying for 50 live LLM calls.

The pipeline uses `LLMTestCase.expected_tools`, `LLMTestCase.tools_called`, and `metadata["tool_traces"]`:

```python
LLMTestCase(
    input="Find the cheapest flight from Beijing to Shanghai",
    actual_output=response.messages[-1]["content"],
    expected_tools=[
        ToolCall(
            name="search_flights",
            input_parameters={"from": "Beijing", "to": "Shanghai"},
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

The script writes `report.md` with actual measured numbers for task success, tool execution success, timeout failures, fallback recovery, latency, and failure attribution counts.

## Difference from existing metrics

`ToolCorrectnessMetric` focuses on whether the expected tools and arguments were used. `ToolUseMetric` is conversational and can rely on LLM-as-judge scoring. `ToolCallingReliabilityMetric` is deterministic and trace-based: it complements those metrics with runtime reliability signals such as timeout rate, retry rate, fallback rate, execution success, latency, and root-cause attribution.