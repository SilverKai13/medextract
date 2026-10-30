| backend | batch | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | throughput (seq/s) |
|---|---|---|---|---|---|---|
| pytorch_eager | 1 | 50.086 | 54.795 | 55.816 | 56.91 | 19.97 |
| pytorch_eager | 16 | 187.124 | 198.573 | 201.143 | 217.542 | 85.5 |
| onnxruntime | 1 | 18.387 | 19.572 | 19.955 | 20.929 | 54.39 |
| onnxruntime | 16 | 142.19 | 152.621 | 155.797 | 158.776 | 112.53 |

ONNX Runtime speedup over PyTorch eager at p50 — batch 1: 2.72x, batch 16: 1.32x.

Measured on Windows-11-10.0.26200-SP0, 16 vCPU, torch 2.6.0+cu124, onnxruntime 1.26.0. CPU only, no GPU involved.
