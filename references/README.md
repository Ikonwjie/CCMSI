# Reference Metrics

This directory contains predefined reference metrics for result comparisons. The JSON/CSV files include data for ten models. Run outputs are saved under `results/`.

## Long-Term Forecasting: Mean over Four Horizons

Forecasting horizons: 96, 192, 336, and 720.

| Dataset | MSE | MAE |
| --- | --- | --- |
| ETTh1 | 0.422 ± 0.004 | 0.427 ± 0.000 |
| ETTm1 | 0.370 ± 0.003 | 0.382 ± 0.002 |
| ETTh2 | 0.350 ± 0.002 | 0.390 ± 0.003 |
| ETTm2 | 0.270 ± 0.005 | 0.316 ± 0.002 |
| Weather | 0.242 ± 0.001 | 0.272 ± 0.005 |
| Electricity | 0.164 ± 0.001 | 0.254 ± 0.000 |
| Traffic | 0.429 ± 0.007 | 0.279 ± 0.002 |

## M4 Short-Term Forecasting

| Subset | sMAPE | MASE | OWA |
| --- | --- | --- | --- |
| Yearly | 13.283 ± 0.192 | 2.785 ± 0.130 | 0.757 ± 0.022 |
| Quarterly | 10.028 ± 0.085 | 1.164 ± 0.021 | 0.880 ± 0.011 |
| Monthly | 12.601 ± 0.107 | 0.921 ± 0.024 | 0.870 ± 0.015 |
| Others | 4.638 ± 0.078 | 3.079 ± 0.014 | 0.974 ± 0.010 |
| Average | 11.742 ± 0.087 | 1.516 ± 0.035 | 0.829 ± 0.012 |

`aggregate_results.py` aggregates run results and reports their differences from the reference means.
