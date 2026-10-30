Reliability of `score` on ncbi_disease validation (923 sentences, 787 gold entities, 817 predictions), `onnx` backend.

| score bin | predictions | mean score | precision | gap |
|---|---|---|---|---|
| 0.5-0.6 | 8 | 0.5631 | 0.1250 | -0.4381 |
| 0.6-0.7 | 15 | 0.6589 | 0.4000 | -0.2589 |
| 0.7-0.8 | 20 | 0.7468 | 0.2000 | -0.5468 |
| 0.8-0.9 | 32 | 0.8657 | 0.5625 | -0.3032 |
| 0.9-1.0 | 742 | 0.9928 | 0.9070 | -0.0858 |

Expected calibration error **0.1123**; overall precision 0.8592 against a mean score of 0.9715.

| min_score | entities kept | precision | recall |
|---|---|---|---|
| 0.5 | 817 | 0.8592 | 0.8920 |
| 0.7 | 794 | 0.8753 | 0.8831 |
| 0.8 | 774 | 0.8928 | 0.8780 |
| 0.9 | 742 | 0.9070 | 0.8551 |
| 0.95 | 716 | 0.9176 | 0.8348 |
| 0.99 | 632 | 0.9383 | 0.7535 |
