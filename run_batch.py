import argparse
import itertools
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone

from run_experiment import parse_overrides
from utils.reproduction import PROJECT_ROOT, load_config, resolve_path, validate_config, write_json


def expand_suite(suite, extra=None):
    keys = list(suite.get("grid", {}))
    combinations = list(itertools.product(*(suite["grid"][key] for key in keys)))
    for path in suite["configs"]:
        for values in combinations:
            overrides = {**suite.get("overrides", {}), **dict(zip(keys, values)), **(extra or {})}
            yield path, overrides, validate_config(load_config(resolve_path(path), overrides))


def main():
    parser = argparse.ArgumentParser(description="Sequential, fail-fast multi-seed experiment suite")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    suite = json.loads(resolve_path(args.suite).read_text(encoding="utf-8"))
    jobs = list(expand_suite(suite, parse_overrides(args.set)))
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    batch_dir = resolve_path("results/batches") / batch_id
    manifest = {"status": "running", "suite": args.suite, "jobs": [], "expected_jobs": len(jobs)}
    for index, (path, overrides, config) in enumerate(jobs):
        job_dir = resolve_path(config["output_dir"]) / batch_id / f"job_{index:03d}"
        overrides["output_dir"] = str(job_dir)
        command = [sys.executable, str(PROJECT_ROOT / "run_experiment.py"), "--config", str(resolve_path(path))]
        for key, value in overrides.items():
            command += ["--set", f"{key}={json.dumps(value)}"]
        if args.dry_run:
            print(json.dumps({"index": index, "command": command}, ensure_ascii=False))
            continue
        write_json(batch_dir / "manifest.json", manifest)
        completed = subprocess.run(command, cwd=PROJECT_ROOT)
        results = list(job_dir.glob("*/metrics.json"))
        record = {"index": index, "command": command, "returncode": completed.returncode, "metrics": str(results[0]) if len(results) == 1 else None}
        manifest["jobs"].append(record)
        if completed.returncode or len(results) != 1:
            manifest["status"] = "failed"
            write_json(batch_dir / "manifest.json", manifest)
            raise SystemExit(f"Job failed; inspect {batch_dir / 'manifest.json'}")
        write_json(batch_dir / "manifest.json", manifest)
    if not args.dry_run:
        manifest["status"] = "complete"
        write_json(batch_dir / "manifest.json", manifest)
        print(f"Aggregate with: python aggregate_results.py --manifest {batch_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
