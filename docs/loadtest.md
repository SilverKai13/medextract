| concurrency | requests | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) | throughput (req/s) | errors |
|---|---|---|---|---|---|---|---|---|
| 1 | 200 | 273.545 | 422.7 | 429.687 | 566.189 | 601.186 | 4.28 | 0 |
| 2 | 200 | 439.572 | 638.924 | 698.743 | 802.204 | 874.323 | 5.1 | 0 |
| 4 | 200 | 758.13 | 1083.12 | 1152.977 | 1298.373 | 1413.386 | 5.76 | 0 |
| 8 | 200 | 1681.483 | 2318.308 | 2458.576 | 2973.302 | 3145.786 | 5.34 | 0 |
| 16 | 200 | 3821.333 | 5310.078 | 5589.4 | 6173.133 | 6891.5 | 4.72 | 0 |

Client-observed, end to end over HTTP against `/extract` on the `onnx` backend, 64 distinct texts cycled, mean 1616.2 characters (max 4476). Closed loop: each concurrent caller waits for its response before sending the next.

Client on Windows-11-10.0.26200-SP0, 16 vCPU. Server and client share the machine, so these numbers are a ceiling on what this box can serve, not a measurement of the server alone.
