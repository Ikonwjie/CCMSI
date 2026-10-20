# Parameter Reference

Override configuration values with `--set parameter=value`, for example, `--set seed=43` or `--set save_predictions=false`.

## Default Traffic Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| data / features / enc_in | Traffic / M / 862 | Dataset, forecasting mode, and number of variables |
| root_path / data_path | dataset/traffic / traffic.csv | Data directory and filename |
| seq_len / pred_len | 96 / 96 | Input length and forecasting horizon |
| patch_len / stride | 16 / 8 | Patch length and stride |
| gpt_layers / hidden_size | 6 / 768 | Number of GPT-2 layers and hidden dimension |
| batch_size / eval_batch_size | 8 / 1 | Training and evaluation batch sizes |
| epochs / patience | 50 / 10 | Maximum training epochs and early-stopping patience |
| learning_rate | 0.0005 | AdamW learning rate |
| seed | 42 | Random seed; batch runs use 42/43/44 |
| device | cuda:0 | Execution device; can be set to cpu |

## Common Parameters

- `task`: long_term / low_data / m4.
- `percent`: Percentage of the earliest training windows to use; low-data tasks use 10/25/50.
- `bridge_layers`: Bridge locations, using one-based indexing; defaults to [1,3,5].
- `frequency`: M4 frequency. `pred_len` is set to the corresponding forecasting horizon, with `seq_len=2*pred_len`, `patch_len=4`, and `stride=2`.
- `m4_refit`: Whether to refit on the full training set for the selected number of epochs; defaults to true.
- `naive2_path`: Path to the M4 Naive2 benchmark CSV file.
- `backbone_path`: GPT-2 weights directory; `output_dir`: Output directory.
- `save_predictions`: Whether to save prediction arrays; defaults to true.
- `num_workers`: Number of data-loading worker processes; defaults to 0. `deterministic`: Whether to enable deterministic algorithms; defaults to false.

See `configs/base.json` for the complete set of default parameters. The effective configuration for each run is saved as `config.json`.

The default set of trainable components is fixed: GPT-2 token embeddings, attention layers, and feed-forward networks (FFNs) are frozen, while positional embeddings, LayerNorm parameters, and the newly introduced modules are trained.

Current Linux defaults: the base batch_size=512 is inherited by ETT/M4; Weather uses 16, and Traffic/Electricity use 8. The default training settings are epochs=50 and patience=10. See the [README](../README.md) for hardware specifications and launch instructions.
