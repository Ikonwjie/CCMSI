import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.reproduction import file_sha256, resolve_path


class ForecastWindows(Dataset):
    def __init__(self, values, seq_len, pred_len, start, stop):
        self.values = values
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.start = start
        self.stop = stop
        self.count = stop - start - seq_len - pred_len + 1
        if self.count <= 0:
            raise ValueError(f"No complete forecasting windows in [{start}, {stop})")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        origin = self.start + index + self.seq_len
        return self.values[origin - self.seq_len:origin], self.values[origin:origin + self.pred_len], origin


def prepare_csv(config):
    path = resolve_path(config["root_path"]) / config["data_path"]
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"Expected a date column in {path}")
    columns = [column for column in frame.columns if column != "date"]
    if config["features"] in {"S", "MS"}:
        if config["target"] not in columns:
            raise ValueError(f"Unknown target {config['target']}")
        columns = [config["target"]] if config["features"] == "S" else [column for column in columns if column != config["target"]] + [config["target"]]
    values = frame[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains missing or non-finite values; preprocess explicitly before running")
    if len(columns) != config["enc_in"]:
        raise ValueError(f"enc_in={config['enc_in']} but data contains {len(columns)} selected variables")
    row_count = len(values)
    if config["data"] in {"ETTh1", "ETTh2", "ETTm1", "ETTm2"}:
        multiplier = 4 if config["data"].startswith("ETTm") else 1
        train_stop, validation_stop, test_stop = [months * 30 * 24 * multiplier for months in (12, 16, 20)]
        if row_count < test_stop:
            raise ValueError(f"ETT split requires at least {test_stop} rows, got {row_count}")
    else:
        train_stop = int(row_count * 0.7)
        validation_stop = row_count - int(row_count * 0.2)
        test_stop = row_count
    seq_len, pred_len = config["seq_len"], config["pred_len"]
    total_train_windows = train_stop - seq_len - pred_len + 1
    if total_train_windows <= 0:
        raise ValueError("Training split is shorter than seq_len + pred_len")
    selected_windows = max(1, math.ceil(total_train_windows * config["percent"] / 100))
    available_stop = seq_len + pred_len - 1 + selected_windows
    scaler = StandardScaler().fit(values[:available_stop])
    standardized = scaler.transform(values).astype(np.float32)
    datasets = {
        "train": ForecastWindows(standardized, seq_len, pred_len, 0, available_stop),
        "val": ForecastWindows(standardized, seq_len, pred_len, train_stop - seq_len, validation_stop),
        "test": ForecastWindows(standardized, seq_len, pred_len, validation_stop - seq_len, test_stop),
    }
    metadata = {"file": str(path), "file_sha256": file_sha256(path), "columns": columns, "row_count": row_count, "target_intervals": {"train": [seq_len, available_stop], "val": [train_stop, validation_stop], "test": [validation_stop, test_stop]}, "percent": config["percent"], "low_data_protocol": "first ceil(percent * eligible_training_windows / 100) windows", "full_training_windows": total_train_windows, "sample_counts": {name: len(dataset) for name, dataset in datasets.items()}, "scaler_fit_interval": [0, available_stop], "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "metric_scale": "training-standardized", "input_normalization": "training-only StandardScaler followed by model RevIN"}
    prototypes = [standardized[:available_stop, channel] for channel in range(len(columns))]
    return datasets, prototypes, metadata
