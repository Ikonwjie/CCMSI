import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data_provider.m4 import M4_FREQUENCIES
from utils.m4_metrics import summarize_errors
from utils.reproduction import resolve_path, write_json


def signature(result, m4=False):
    excluded = {"seed", "pred_len", "output_dir", "centroid_cache_dir"}
    if m4:
        excluded.add("frequency")
        excluded.add("seq_len")
    config = {key: value for key, value in result["config"].items() if key not in excluded}
    if m4:
        config["input_to_horizon_ratio"] = result["config"]["seq_len"] / result["config"]["pred_len"]
    return json.dumps({"config": config, "backbone": result["backbone_sha256"], "source": result["source_sha256"], "protocol": result["protocol_version"]}, sort_keys=True)


def repeated_summary(per_seed):
    metrics = next(iter(per_seed.values()))
    return {key: {"mean": float(np.mean([row[key] for row in per_seed.values()])), "std": float(np.std([row[key] for row in per_seed.values()], ddof=1)) if len(per_seed) > 1 else None} for key in metrics}


def aggregate_long_term(records, expected_seeds):
    groups = defaultdict(dict)
    for result, path in records:
        config = result["config"]
        group = (config["task"], config["data"], config["percent"])
        pair = (config["seed"], config["pred_len"])
        if pair in groups[group]:
            raise ValueError(f"Duplicate run {group} {pair}")
        groups[group][pair] = result
    output = []
    expected = {(seed, horizon) for seed in expected_seeds for horizon in (96, 192, 336, 720)}
    for group, runs in groups.items():
        if set(runs) != expected:
            raise ValueError(f"Incomplete or unexpected horizon/seed grid for {group}: missing {expected - set(runs)}, extra {set(runs) - expected}")
        if len({signature(result) for result in runs.values()}) != 1 or len({result["data_sha256"] for result in runs.values()}) != 1:
            raise ValueError(f"Cannot mix settings, source versions, weights or data in {group}")
        per_seed = {str(seed): {metric: float(np.mean([runs[seed, horizon]["metrics"][metric] for horizon in (96, 192, 336, 720)])) for metric in ("MSE", "MAE")} for seed in expected_seeds}
        output.append({"task": group[0], "dataset": group[1], "percent": group[2], "per_seed_four_horizon_mean": per_seed, "summary": repeated_summary(per_seed)})
    return output


def aggregate_m4(records, expected_seeds):
    runs = {}
    for result, path in records:
        config = result["config"]
        pair = (config["seed"], config["frequency"])
        if pair in runs:
            raise ValueError(f"Duplicate M4 run {pair}")
        runs[pair] = result, path
    expected = {(seed, frequency) for seed in expected_seeds for frequency in M4_FREQUENCIES}
    if set(runs) != expected:
        raise ValueError(f"M4 requires all six frequencies for every seed: missing {expected - set(runs)}, extra {set(runs) - expected}")
    if len({signature(result, m4=True) for result, _ in runs.values()}) != 1:
        raise ValueError("M4 runs use inconsistent configurations, weights or code")
    naive_hashes = set()
    errors = {}
    for frequency in M4_FREQUENCIES:
        hashes = {(runs[seed, frequency][0]["data_sha256"], runs[seed, frequency][0]["test_sha256"]) for seed in expected_seeds}
        if len(hashes) != 1:
            raise ValueError(f"M4 data differs between seeds for {frequency}")
        baseline_reference = None
        for seed in expected_seeds:
            result, path = runs[seed, frequency]
            naive_hashes.add(json.loads((path.parent / "naive2.json").read_text(encoding="utf-8"))["sha256"])
            with (path.parent / "per_series.csv").open(encoding="utf-8", newline="") as stream:
                rows = sorted(csv.DictReader(stream), key=lambda row: row["id"])
            identifiers = [row["id"] for row in rows]
            if len(identifiers) != len(set(identifiers)) or len(rows) != result["metrics"]["series_count"]:
                raise ValueError("M4 duplicate IDs or inconsistent series count")
            if result["config"]["strict_m4_counts"] and len(rows) != M4_FREQUENCIES[frequency]["count"]:
                raise ValueError(f"Incomplete official M4 subset: {frequency}")
            arrays = np.asarray([[float(row[key]) for row in rows] for key in ("sMAPE", "MASE", "Naive2_sMAPE", "Naive2_MASE")])
            baseline = (identifiers, arrays[2:].tolist())
            if baseline_reference is not None and baseline != baseline_reference:
                raise ValueError("M4 series IDs or Naive2 errors differ across seeds")
            baseline_reference = baseline
            recalculated = summarize_errors(*arrays)
            if any(not np.isclose(recalculated[key], result["metrics"][key], rtol=1e-10, atol=1e-10) for key in ("sMAPE", "MASE", "OWA")):
                raise ValueError("M4 metrics.json does not match per_series.csv")
            errors[seed, frequency] = arrays
    if len(naive_hashes) != 1:
        raise ValueError("M4 runs use different Naive2 reference files")
    groups = {"Yearly": ["Yearly"], "Quarterly": ["Quarterly"], "Monthly": ["Monthly"], "Others": ["Weekly", "Daily", "Hourly"], "Average": list(M4_FREQUENCIES)}
    output = []
    for name, frequencies in groups.items():
        per_seed = {}
        for seed in expected_seeds:
            metrics = summarize_errors(*np.concatenate([errors[seed, frequency] for frequency in frequencies], axis=1))
            per_seed[str(seed)] = {key: metrics[key] for key in ("sMAPE", "MASE", "OWA")}
        output.append({"subset": name, "per_seed": per_seed, "summary": repeated_summary(per_seed)})
    return output


def aggregate(manifest_path, expected_seeds, output_dir):
    if len(set(expected_seeds)) != len(expected_seeds) or not expected_seeds:
        raise ValueError("Expected seeds must be unique and nonempty")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or len(manifest["jobs"]) != manifest["expected_jobs"]:
        raise ValueError("Cannot aggregate an unfinished batch")
    records = []
    for job in manifest["jobs"]:
        if job["returncode"] or not job["metrics"]:
            raise ValueError("Batch contains a failed job")
        path = Path(job["metrics"])
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "measured":
            raise ValueError("Only measured run outputs may be aggregated")
        records.append((result, path))
    if not records:
        raise ValueError("No completed runs")
    m4_records = [record for record in records if record[0]["config"]["task"] == "m4"]
    long_records = [record for record in records if record[0]["config"]["task"] != "m4"]
    output_dir = Path(output_dir)
    for name, rows in (("long_term_four_horizon_mean", aggregate_long_term(long_records, expected_seeds) if long_records else []), ("m4", aggregate_m4(m4_records, expected_seeds) if m4_records else [])):
        if not rows:
            continue
        reference = json.loads(resolve_path(f"references/paper_{name}.json").read_text(encoding="utf-8"))
        for row in rows:
            eligible = name == "m4" or (row["task"] == "long_term" and row["percent"] == 100 and not any(row["disabled_modules"]))
            key = row.get("dataset", row.get("subset"))
            reference_metrics = reference["ccmsi"].get(key) if eligible else None
            row["reference_metrics"] = reference_metrics
            row["difference_from_reference_mean"] = {metric: values["mean"] - reference_metrics[metric]["mean"] for metric, values in row["summary"].items()} if reference_metrics else None
        write_json(output_dir / f"{name}.json", {"status": "aggregated_measured_runs", "seeds": expected_seeds, "std_ddof": 1, "manifest": str(manifest_path), "rows": rows})
        with (output_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset_or_subset", "task", "percent", "disabled_modules", "metric", "mean", "sample_std", "reference_mean", "difference_from_reference_mean"])
            for row in rows:
                for metric, values in row["summary"].items():
                    reference_metrics = row["reference_metrics"]
                    writer.writerow([row.get("dataset", row.get("subset")), row.get("task", "m4"), row.get("percent", 100), json.dumps(row.get("disabled_modules", [])), metric, values["mean"], values["std"], reference_metrics[metric]["mean"] if reference_metrics else None, row["difference_from_reference_mean"][metric] if reference_metrics else None])
        print(output_dir / f"{name}.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    aggregate(args.manifest, args.seeds, args.output_dir or Path(args.manifest).parent / "summary")


if __name__ == "__main__":
    main()
