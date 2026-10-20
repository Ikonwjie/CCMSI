# CCMSI

## 1. Environment

### Environment Setup

Run the following commands from this directory:

```bash
conda env create -f environment.yml
conda activate ccmsi-reproduction
hf auth login
hf download EleutherAI/gpt-j-6b --local-dir weights/gpt-j-6b
hf download Qwen/Qwen2.5-7B --local-dir weights/Qwen2.5-7B
hf download meta-llama/Llama-2-7b-hf --local-dir weights/Llama-2-7b-hf
python scripts/prepare_gpt2.py
```

### Running on Linux

The environment and dependencies are specified in `environment.yml`, `requirements.txt`, and `requirements-verified.txt`.
The environment is named `ccmsi-reproduction` and uses Python 3.12.7.
The launch scripts use `python` from the active environment. You can also specify an interpreter through the `PYTHON` environment variable. The scripts do not depend on user-specific installation paths or automatically create environments or install dependencies.

### Reference Hardware

The following specifications describe the development machine; they are not hardware requirements for running the code.

| Component | Specification |
| --- | --- |
| CPU | 2 × Intel Xeon Platinum 8358 @ 2.60 GHz |
| CPU cores | 64 physical cores and 128 logical threads in total |
| System memory | Approximately 251 GiB reported by Linux |
| GPU | 4 × NVIDIA GeForce RTX 4090, indexed 0–3 |
| Memory per GPU | 24564 MiB (approximately 24 GiB) |
| NVIDIA driver | 570.211.01 |

## 2. Data

Run the following commands from this directory to download all datasets used in the paper. The ETT command downloads ETTh1, ETTh2, ETTm1, and ETTm2 together:

```bash
python scripts/prepare_data.py --dataset ETT
python scripts/prepare_data.py --dataset Weather
python scripts/prepare_data.py --dataset Electricity
python scripts/prepare_data.py --dataset Traffic
python scripts/prepare_data.py --dataset M4
```

Data are saved automatically to the corresponding subdirectories under `dataset/`. The script records the source revision and the SHA-256 hash of each file without overwriting existing files. Extract the Naive2 archive to `dataset/m4/submission-Naive2.csv`. See the [data documentation](docs/DATA.md) for data sources and preprocessing details.

## 3. Tasks and Commands

| Experiment | Command |
| --- | --- |
| Long-term forecasting, H=96 | `bash scripts/long_term_forecasting/Traffic_pl96.sh` |
| Long-term forecasting, mean over four horizons | `bash scripts/long_term_forecasting/run_all.sh` |
| Short-term forecasting | `bash scripts/short_term_forecasting/M4.sh` |
| Low-data forecasting, 10%/25%/50% | `bash scripts/low_data_forecasting/run_all.sh` |
| GPT-2 module ablation on Weather | `bash scripts/run_linux.sh run_batch.py --suite configs/suites/ablation.json` |

```bash
python run_experiment.py --config configs/long_term/Traffic_pl96.json
```

Run outputs are saved under `results/`. Aggregate them using the batch directory name reported by the run:

```bash
python aggregate_results.py --manifest results/batches/<batch_id>/manifest.json
```

### Linux Runtime Options

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/long_term_forecasting/Traffic_pl96.sh --dry-run
```

## 4. Results

### Long-Term Forecasting Reference Results

![Long-term forecasting results averaged over four horizons](assets/results/paper_table1_long_term.png)

### M4 Short-Term Forecasting Reference Results

![M4 short-term forecasting results](assets/results/paper_table2_m4.png)
