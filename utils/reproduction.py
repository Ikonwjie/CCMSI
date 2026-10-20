import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(path, overrides=None, visited=None):
    path = Path(path).resolve()
    visited = set() if visited is None else set(visited)
    if path in visited:
        raise ValueError(f"Circular configuration inheritance: {path}")
    visited.add(path)
    local = json.loads(path.read_text(encoding="utf-8"))
    parent = local.pop("extends", None)
    values = load_config(path.parent / parent, visited=visited) if parent else {}
    values.update(local)
    values.update(overrides or {})
    return values


def validate_config(config):
    allowed = set(json.loads((PROJECT_ROOT / "configs/base.json").read_text(encoding="utf-8")))
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    missing = allowed - set(config)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    if config["task"] not in {"long_term", "low_data", "m4"}:
        raise ValueError("Unsupported task")
    for name in ("seq_len", "pred_len", "enc_in", "batch_size", "eval_batch_size", "epochs", "patience", "patch_len", "stride", "gpt_layers", "mode_clusters", "p2t_topk", "p2t_low_rank", "soft_vocab_topk", "max_centroid_patches", "m4_samples_per_series", "pattern_vocab_chunk"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("deterministic", "save_predictions", "strict_m4_counts", "m4_refit"):
        if type(config[name]) is not bool:
            raise ValueError(f"{name} must be a JSON boolean (true/false)")
    for name in ("percent", "learning_rate", "weight_decay", "gradient_clip", "dropout", "gumbel_tau_start", "gumbel_tau_end", "bridge_scale", "agreement_center", "agreement_width", "bridge_decay_start", "bridge_decay_floor", "adapter_coop_weight"):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]):
            raise ValueError(f"{name} must be a finite number")
    if not isinstance(config["bridge_layers"], list) or not config["bridge_layers"]:
        raise ValueError("bridge_layers must be a nonempty list")
    if config["weight_decay"] < 0 or config["gradient_clip"] <= 0 or config["agreement_width"] <= 0 or config["adapter_coop_weight"] < 0:
        raise ValueError("Invalid optimizer, agreement width or cooperation weight")
    if not 0 <= config["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    if type(config["num_workers"]) is not int or type(config["p2t_anchor_radius"]) is not int or config["p2t_anchor_radius"] < 0:
        raise ValueError("num_workers and p2t_anchor_radius must be integers; radius must be nonnegative")
    if type(config["seed"]) is not int or not 0 <= config["seed"] < 2 ** 32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if not 0 < config["percent"] <= 100:
        raise ValueError("percent must be in (0, 100]")
    if config["seq_len"] < config["patch_len"]:
        raise ValueError("seq_len must be at least patch_len")
    if config["gpt_layers"] > 12 or config["hidden_size"] != 768:
        raise ValueError("The released recipe uses GPT-2 small: hidden_size=768 and at most 12 blocks")
    if any(type(layer) is not int or not 1 <= layer <= config["gpt_layers"] for layer in config["bridge_layers"]):
        raise ValueError("bridge_layers must use one-based block numbers")
    if config["features"] not in {"M", "MS", "S"}:
        raise ValueError("features must be M, MS or S")
    if config["num_workers"] < 0 or config["learning_rate"] <= 0:
        raise ValueError("Invalid worker count or learning rate")
    if config["gumbel_tau_start"] <= 0 or config["gumbel_tau_end"] <= 0:
        raise ValueError("Gumbel temperatures must be positive")
    if config["task"] == "m4":
        from data_provider.m4 import M4_FREQUENCIES
        if config["frequency"] not in M4_FREQUENCIES:
            raise ValueError("Unknown M4 frequency")
        if config["pred_len"] != M4_FREQUENCIES[config["frequency"]]["horizon"] or config["enc_in"] != 1:
            raise ValueError("M4 requires its official horizon and enc_in=1")
        if config["percent"] != 100:
            raise ValueError("M4 low-data subsampling is not part of this recipe")
    return config


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def environment_record():
    packages = {}
    for name in ("torch", "transformers", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "tqdm", "huggingface-hub", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    commit = None
    dirty = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in sorted(PROJECT_ROOT.rglob("*.py")) if not any(part in {".venv", "dataset", "results", "weights"} for part in path.relative_to(PROJECT_ROOT).parts)}
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "git_commit": commit, "git_dirty": dirty, "source_sha256": source_hashes}
