import csv

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.reproduction import file_sha256, resolve_path


M4_FREQUENCIES = {
    "Yearly": {"horizon": 6, "period": 1, "count": 23000},
    "Quarterly": {"horizon": 8, "period": 4, "count": 24000},
    "Monthly": {"horizon": 18, "period": 12, "count": 48000},
    "Weekly": {"horizon": 13, "period": 1, "count": 359},
    "Daily": {"horizon": 14, "period": 1, "count": 4227},
    "Hourly": {"horizon": 48, "period": 24, "count": 414},
}


def read_m4_csv(path):
    series = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.reader(stream)):
            if not row:
                continue
            identifier = row[0].strip()
            if row_number == 0 and (not identifier or identifier.lower() in {"v1", "id", "m4id", "series_id"}):
                continue
            cells = row[1:]
            while cells and cells[-1].strip().lower() in {"", "na", "nan"}:
                cells.pop()
            if identifier in series:
                raise ValueError(f"Duplicate series ID {identifier} in {path}")
            try:
                values = np.asarray(cells, dtype=np.float64)
            except ValueError as error:
                raise ValueError(f"Non-numeric or internal missing values for {identifier} in {path}") from error
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"Empty or non-finite series {identifier}")
            series[identifier] = values
    if not series:
        raise ValueError(f"No M4 series found in {path}")
    return series


def padded_context(history, seq_len):
    history = np.asarray(history, dtype=np.float32)
    if len(history) == 0:
        raise ValueError("Cannot pad an empty history")
    context = history[-seq_len:]
    if len(context) < seq_len:
        context = np.pad(context, (seq_len - len(context), 0), mode="edge")
    return context[:, None]


class M4TrainingWindows(Dataset):
    def __init__(self, histories, seq_len, pred_len, samples_per_series=1):
        self.histories = histories
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples_per_series = samples_per_series
        self.means = [float(history.mean()) for history in histories]
        self.scales = [max(float(history.std()), 1e-5) for history in histories]
        self.eligible_indices = [index for index, history in enumerate(histories) if len(history) > pred_len]
        if samples_per_series < 1 or not self.eligible_indices:
            raise ValueError("M4 training requires at least one series with a history point and a complete future horizon")

    def __len__(self):
        return len(self.eligible_indices) * self.samples_per_series

    def __getitem__(self, index):
        series_index = self.eligible_indices[index % len(self.eligible_indices)]
        history = (self.histories[series_index] - self.means[series_index]) / self.scales[series_index]
        origin = int(torch.randint(1, len(history) - self.pred_len + 1, ()).item())
        return padded_context(history[:origin], self.seq_len), history[origin:origin + self.pred_len].astype(np.float32)[:, None], series_index


class M4EvaluationWindows(Dataset):
    def __init__(self, histories, targets, seq_len, scaler_histories=None):
        self.histories = histories
        self.targets = targets
        self.seq_len = seq_len
        scaler_histories = histories if scaler_histories is None else scaler_histories
        self.means = np.asarray([float(history.mean()) for history in scaler_histories])
        self.scales = np.asarray([max(float(history.std()), 1e-5) for history in scaler_histories])

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, index):
        history = (self.histories[index] - self.means[index]) / self.scales[index]
        target = (self.targets[index] - self.means[index]) / self.scales[index]
        return padded_context(history, self.seq_len), target.astype(np.float32)[:, None], index

    def inverse_batch(self, values, indices):
        return values * self.scales[indices, None, None] + self.means[indices, None, None]


class M4Data:
    def __init__(self, config):
        self.config = config
        self.frequency = config["frequency"]
        self.horizon = M4_FREQUENCIES[self.frequency]["horizon"]
        self.root = resolve_path(config["root_path"])
        self.train_path = self.root / "Train" / f"{self.frequency}-train.csv"
        train = read_m4_csv(self.train_path)
        self.identifiers = list(train)
        self.histories = list(train.values())
        if config["strict_m4_counts"] and len(self.identifiers) != M4_FREQUENCIES[self.frequency]["count"]:
            raise ValueError(f"Unexpected M4 {self.frequency} series count: {len(self.identifiers)}")
        if any(len(history) <= self.horizon for history in self.histories):
            raise ValueError("M4 internal holdout requires a nonempty history before the last horizon")
        self.fit_histories = [history[:-self.horizon] for history in self.histories]
        holdouts = [history[-self.horizon:] for history in self.histories]
        self.train = M4TrainingWindows(self.fit_histories, config["seq_len"], self.horizon, config["m4_samples_per_series"])
        self.validation = M4EvaluationWindows(self.fit_histories, holdouts, config["seq_len"])

    def full_training(self):
        return M4TrainingWindows(self.histories, self.config["seq_len"], self.horizon, self.config["m4_samples_per_series"])

    def test_data(self):
        test_path = self.root / "Test" / f"{self.frequency}-test.csv"
        test = read_m4_csv(test_path)
        if set(test) != set(self.identifiers):
            raise ValueError("M4 train/test series IDs differ")
        targets = [test[identifier] for identifier in self.identifiers]
        if any(len(target) != self.horizon for target in targets):
            raise ValueError("M4 test series does not match the official horizon")
        scaler_histories = self.histories if self.config["m4_refit"] else self.fit_histories
        return M4EvaluationWindows(self.histories, targets, self.config["seq_len"], scaler_histories), file_sha256(test_path)

    def prototype_series(self, refit=False):
        histories = self.histories if refit else self.fit_histories
        return [((history - history.mean()) / max(float(history.std()), 1e-5)).astype(np.float32) for history in histories]

    def metadata(self, refit=False):
        return {"file": str(self.train_path), "file_sha256": file_sha256(self.train_path), "frequency": self.frequency, "series_count": len(self.identifiers), "horizon": self.horizon, "period": M4_FREQUENCIES[self.frequency]["period"], "validation": "last H observations of each official training series; official test unseen during selection", "phase": "full_history_refit" if refit else "internal_holdout_selection", "normalization": "per-series training-phase mean/std (population std, floor 1e-5), then model RevIN; training/validation MSE standardized; test M4 metrics inverse transformed", "prototype_normalization": "per-series mean/std using only the current training phase history", "short_context": "left edge padding; targets always complete and never padded", "selection_training_series_count": len(self.train.eligible_indices), "selection_training_excluded_ids": [identifier for identifier, history in zip(self.identifiers, self.fit_histories) if len(history) <= self.horizon], "training_sampling": "uniform origin from 1 through history_length-H; internal prefixes too short for a complete target excluded from training only, never from validation/test"}
