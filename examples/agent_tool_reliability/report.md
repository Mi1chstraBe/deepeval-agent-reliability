# Agent Tool-Calling Reliability Report

Deterministic mock benchmark: 50 paired cases, no LLM/API calls.

| Metric | Baseline | Enhanced | Delta |
| --- | ---: | ---: | ---: |
| Total cases | 50 | 50 | 0 |
| Task success rate | 68.0% | 92.0% | 24.0% |
| Tool execution success | 76.0% | 92.0% | 16.0% |
| Tool selection accuracy | 94.0% | 98.0% | 4.0% |
| Argument match rate | 91.0% | 98.0% | 7.0% |
| Timeout failures | 8 | 3 | -5 |
| Fallback recovered cases | 0 | 5 | +5 |
| Avg latency | 507 ms | 483 ms | -24 ms |

## Failure Attribution Counts

| Attribution | Baseline | Enhanced |
| --- | ---: | ---: |
| FALLBACK_USED | 0 | 5 |
| NO_TOOL_CALLED | 1 | 1 |
| TOOL_EXECUTION_FAILED | 11 | 3 |
| TOOL_TIMEOUT | 8 | 8 |
| WRONG_ARGUMENTS | 2 | 0 |
| WRONG_TOOL | 2 | 0 |

## Notes

- ToolCorrectnessMetric checks whether expected tools and arguments were used.
- TOOL_TIMEOUT is a root-cause attribution, not necessarily a final task failure; fallback can recover a timed-out primary tool call.
- Enhanced latency is lower because timeout cutoffs and fallback remove several long-tail baseline failures.
- ToolCallingReliabilityMetric is deterministic and trace-based: it adds timeout, retry, fallback, latency, runtime success, and root-cause attribution signals.
