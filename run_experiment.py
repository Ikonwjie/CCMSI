import argparse
import csv
import gc
import hashlib
import json
import uuid
from datetime import datime, timezone
from pathlib import Path
from types import SimpleNamespac

from utils.linux_runtime import configure_environment

configure_environment()

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.m4 import M4Data, M4_FREQUENCIES, read_m4_csv
from data_provider.reproduction_csv import prepare_csv
from models.model_dual import BALM_MedualTime
from utils.m4_metrics import series_errors, summarize_errors
from utils.reproduction import environment_record, file_sha256, load_config, resolve_path, seed_worker, set_random_seed, validate_config, write_json


def parse_overrides(items):
    overrides = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Expected key=value, got {item}")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    return overrides


def make_model(config, prototypes, metadata, initial_centroids=None):
    fingerprint_payload = {"data": metadata, "seed": config["seed"], "patch_len": config["patch_len"], "stride": config["stride"], "mode_clusters": config["mode_clusters"], "max_patches": config["max_centroid_patches"], "algorithm": "sample-before-minibatch-kmeans-v1"}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    model_config = dict(config)
    model_config.update(backbone_path=str(resolve_path(config["backbone_path"])), num_hidden_layers=config["gpt_layers"], ts_config=SimpleNamespace(context_points=config["seq_len"], patch_len=config["patch_len"], stride=config["stride"], vars=config["enc_in"]), data_name=config["data"], experiment_seed=config["seed"], prototype_train_series=prototypes, prototype_fingerprint=fingerprint, initial_mode_centroids=initial_centroids, strict_prototypes=True, centroid_cache_dir=str(resolve_path(config["centroid_cache_dir"])), embd_pdrop=config["dropout"], gumbel_tau=config["gumbel_tau_start"], syncbridge_layers=config["bridge_layers"], syncbridge_scale=config["bridge_scale"], anticollapse_agreement_center=config["agreement_center"], anticollapse_agreement_width=config["agreement_width"], use_straight_through_tokens=False)
    model = BALM_MedualTime(SimpleNamespace(**model_config)).to(config["device"])
    model.config.prototype_train_series = None
    return model


def make_loader(dataset, config, training=False):
    generator = torch.Generator().manual_seed(config["seed"])
    return DataLoader(dataset, batch_size=config["batch_size"] if training else config["eval_batch_size"], shuffle=training, drop_last=False, num_workers=config["num_workers"], worker_init_fn=seed_worker, generator=generator, pin_memory=str(config["device"]).startswith("cuda"))


def select_channels(prediction, target, config):
    if config["features"] == "MS":
        return prediction[..., -1:], target[..., -1:]
    return prediction, target


@torch.no_grad()
def evaluate(model, dataset, config, output_dir=None):
    model.eval()
    square_sum = absolute_sum = 0.0
    count = offset = 0
    saved = {}
    collect = [] if config["task"] == "m4" else None
    for inputs, targets, origins in make_loader(dataset, config):
        prediction = model(inputs.to(config["device"]), mode="test")
        prediction, targets = select_channels(prediction, targets.to(config["device"]), config)
        difference = (prediction.double() - targets.double()).cpu().numpy()
        if not np.isfinite(difference).all():
            raise ValueError("Non-finite evaluation output")
        square_sum += float(np.square(difference).sum())
        absolute_sum += float(np.abs(difference).sum())
        count += difference.size
        pred_array, true_array = prediction.cpu().numpy(), targets.cpu().numpy()
        if hasattr(dataset, "inverse_batch"):
            pred_array = dataset.inverse_batch(pred_array, origins.numpy())
            true_array = dataset.inverse_batch(true_array, origins.numpy())
        if collect is not None:
            collect.append(pred_array[..., 0])
        if output_dir is not None and config["save_predictions"]:
            if not saved:
                for name in ("predictions", "targets"):
                    saved[name] = np.lib.format.open_memmap(output_dir / f"{name}.npy", mode="w+", dtype=pred_array.dtype, shape=(len(dataset), *pred_array.shape[1:]))
                saved["origins"] = np.lib.format.open_memmap(output_dir / "origins.npy", mode="w+", dtype=np.int64, shape=(len(dataset),))
            stop = offset + len(pred_array)
            saved["predictions"][offset:stop] = pred_array
            saved["targets"][offset:stop] = true_array
            saved["origins"][offset:stop] = origins.numpy()
            offset = stop
    for array in saved.values():
        array.flush()
    if not count:
        raise ValueError("Empty evaluation dataset")
    return {"MSE": square_sum / count, "MAE": absolute_sum / count, "evaluated_elements": count, "evaluated_windows": len(dataset)}, np.concatenate(collect) if collect is not None else None


def fit(model, train, validation, config, run_dir, phase="selection", epoch_budget=None):
    budget = epoch_budget or config["epochs"]
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loader = make_loader(train, config, training=True)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    checkpoint_path = run_dir / f"{phase}.pt"
    for epoch in range(budget):
        progress = epoch / max(budget - 1, 1)
        model.set_training_progress(progress)
        model.set_gumbel_tau_progress(progress)
        model.train()
        loss_sum = auxiliary_sum = elements = 0
        for inputs, targets, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions, auxiliary = model(inputs.to(config["device"]), mode="train")
            predictions, targets = select_channels(predictions, targets.to(config["device"]), config)
            forecast_loss = torch.nn.functional.mse_loss(predictions, targets)
            loss = forecast_loss + auxiliary
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_((parameter for parameter in model.parameters() if parameter.requires_grad), config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            loss_sum += forecast_loss.item() * targets.numel()
            auxiliary_sum += auxiliary.item() * targets.numel()
            elements += targets.numel()
        validation_metrics = evaluate(model, validation, config)[0] if validation is not None else None
        monitor = validation_metrics["MSE"] if validation_metrics is not None else None
        record = {"phase": phase, "epoch": epoch + 1, "train_MSE": loss_sum / elements, "cooperation_loss": auxiliary_sum / elements, "validation": validation_metrics, "progress": progress, "gumbel_tau": model.gumbel_tau}
        with (run_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        if monitor is None or monitor < best_loss:
            best_loss = monitor if monitor is not None else best_loss
            best_epoch, stale = epoch + 1, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": best_epoch, "progress": progress, "phase": phase}, checkpoint_path)
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    checkpoint = torch.load(checkpoint_path, map_location=config["device"], weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.set_training_progress(checkpoint["progress"])
    model.set_gumbel_tau_progress(checkpoint["progress"])
    return best_epoch, checkpoint_path


def backbone_record(config):
    root = resolve_path(config["backbone_path"])
    if not root.is_dir() or not (root / "config.json").is_file():
        raise FileNotFoundError(f"Prepare local GPT-2 weights first: {root}; see scripts/prepare_gpt2.py")
    files = sorted(root.glob("*.safetensors")) + sorted(root.glob("pytorch_model*.bin"))
    if not files:
        raise FileNotFoundError(f"No GPT-2 weights in {root}")
    return {path.name: file_sha256(path) for path in [root / "config.json", *files]}


def run(config):
    validate_config(config)
    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --set device=cpu explicitly for CPU runs")
    weights = backbone_record(config)
    set_random_seed(config["seed"], config["deterministic"])
    if config["task"] == "m4":
        naive_path = resolve_path(config["naive2_path"])
        if not naive_path.is_file():
            raise FileNotFoundError(f"Official Naive2 forecasts required for OWA: {naive_path}")
        data = M4Data(config)
        train, validation = data.train, data.validation
        prototypes, metadata = data.prototype_series(), data.metadata()
    else:
        datasets, prototypes, metadata = prepare_csv(config)
        train, validation = datasets["train"], datasets["val"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = resolve_path(config["output_dir"]) / f"{config['task']}_{config['frequency'] or config['data']}_H{config['pred_len']}_s{config['seed']}_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", config)
    environment = environment_record()
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "data.json", metadata)
    write_json(run_dir / "backbone.json", weights)
    try:
        model = make_model(config, prototypes, metadata)
        write_json(run_dir / "trainable_parameters.json", {name: list(parameter.shape) for name, parameter in model.named_parameters() if parameter.requires_grad})
        best_epoch, checkpoint_path = fit(model, train, validation, config, run_dir)
        if config["task"] == "m4" and config["m4_refit"]:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            set_random_seed(config["seed"], config["deterministic"])
            metadata = data.metadata(refit=True)
            write_json(run_dir / "refit_data.json", metadata)
            model = make_model(config, data.prototype_series(refit=True), metadata)
            _, checkpoint_path = fit(model, data.full_training(), None, config, run_dir, phase="refit", epoch_budget=best_epoch)
        if config["task"] == "m4":
            test, test_hash = data.test_data()
            np.savez(run_dir / "scaler.npz", ids=np.asarray(data.identifiers), mean=test.means, scale=test.scales)
        else:
            test, test_hash = datasets["test"], metadata["file_sha256"]
        metrics, predictions = evaluate(model, test, config, run_dir)
        if config["task"] == "m4":
            period = M4_FREQUENCIES[config["frequency"]]["period"]
            targets = np.stack(test.targets)
            if config["save_predictions"]:
                np.save(run_dir / "targets.npy", targets[..., None])
            baseline = read_m4_csv(naive_path)
            if any(identifier not in baseline or len(baseline[identifier]) < config["pred_len"] for identifier in data.identifiers):
                raise ValueError("Naive2 forecasts are missing IDs or horizon values")
            baseline_predictions = np.stack([baseline[identifier][:config["pred_len"]] for identifier in data.identifiers])
            smape, mase = series_errors(data.histories, targets, predictions, period)
            baseline_smape, baseline_mase = series_errors(data.histories, targets, baseline_predictions, period)
            metrics = summarize_errors(smape, mase, baseline_smape, baseline_mase)
            with (run_dir / "per_series.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["id", "sMAPE", "MASE", "Naive2_sMAPE", "Naive2_MASE"])
                writer.writerows(zip(data.identifiers, smape, mase, baseline_smape, baseline_mase))
            write_json(run_dir / "naive2.json", {"path": str(naive_path), "sha256": file_sha256(naive_path), "source": "Mcompetitions/M4-methods official submission-Naive2"})
        result = {"status": "measured", "config": config, "metrics": metrics, "best_selection_epoch": best_epoch, "checkpoint": checkpoint_path.name, "data_sha256": metadata["file_sha256"], "test_sha256": test_hash, "backbone_sha256": weights, "source_sha256": environment["source_sha256"], "protocol_version": "ccmsi-reproduction-v1"}
        write_json(run_dir / "metrics.json", result)
        print(f"Completed: {run_dir}", flush=True)
        return run_dir
    except Exception as error:
        write_json(run_dir / "failure.json", {"error_type": type(error).__name__, "message": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description="CCMSI reproducible long-term, low-data and M4 training/evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = validate_config(load_config(resolve_path(args.config), parse_overrides(args.set)))
    if args.dry_run:
        print(json.dumps(config, indent=2))
    else:
        run(config)


if __name__ == "__main__":
    main()
