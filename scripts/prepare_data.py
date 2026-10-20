import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.reproduction import file_sha256, resolve_path, write_json


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; use an empty --output directory")
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "CCMSI-reproduction"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    temporary.replace(destination)
    return {"url": url, "path": str(destination), "sha256": file_sha256(destination)}


def main():
    parser = argparse.ArgumentParser(description="Download ETT, Weather, Electricity, Traffic or M4 data; archives are not executed or automatically extracted")
    parser.add_argument("--dataset", choices=["ETT", "Weather", "Electricity", "Traffic", "M4"], required=True)
    parser.add_argument("--output")
    parser.add_argument("--revision", help="Source revision (default: main for Weather/Electricity/Traffic, master for ETT/M4)")
    args = parser.parse_args()
    is_tslib = args.dataset in {"Weather", "Electricity", "Traffic"}
    repository = "thuml/Time-Series-Library" if is_tslib else ("zhouhaoyi/ETDataset" if args.dataset == "ETT" else "Mcompetitions/M4-methods")
    requested_revision = urllib.parse.quote(args.revision or ("main" if is_tslib else "master"), safe="")
    revision_url = f"https://huggingface.co/api/datasets/{repository}/revision/{requested_revision}" if is_tslib else f"https://api.github.com/repos/{repository}/commits/{requested_revision}"
    request = urllib.request.Request(revision_url, headers={"User-Agent": "CCMSI-reproduction"})
    with urllib.request.urlopen(request, timeout=60) as response:
        revision = json.load(response)["sha"]
    root = resolve_path(args.output or ("dataset/ETT-small" if args.dataset == "ETT" else f"dataset/{args.dataset.lower()}"))
    if args.dataset == "ETT":
        paths = [(f"ETT-small/{name}.csv", f"{name}.csv") for name in ("ETTh1", "ETTh2", "ETTm1", "ETTm2")]
    elif args.dataset == "M4":
        frequencies = ("Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly")
        paths = [(f"Dataset/{split}/{name}-{split.lower()}.csv", f"{split}/{name}-{split.lower()}.csv") for split in ("Train", "Test") for name in frequencies]
        paths += [("Dataset/M4-info.csv", "M4-info.csv"), ("Point Forecasts/submission-Naive2.rar", "submission-Naive2.rar")]
    else:
        dataset_name = args.dataset.lower()
        paths = [(f"{dataset_name}/{dataset_name}.csv", f"{dataset_name}.csv")]
    records = []
    for source, destination in paths:
        base_url = f"https://huggingface.co/datasets/{repository}/resolve/{revision}" if is_tslib else f"https://raw.githubusercontent.com/{repository}/{revision}"
        url = f"{base_url}/{urllib.parse.quote(source)}"
        records.append(download(url, root / destination))
        write_json(root / "download_manifest.json", {"repository": repository, "revision": revision, "files": records, "complete": len(records) == len(paths)})
        print(destination, flush=True)
    if args.dataset == "M4":
        print("Extract submission-Naive2.rar with a RAR-capable tool, then set naive2_path to its CSV. Do not replace Naive2 with naive forecasts.")


if __name__ == "__main__":
    main()
