import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import HfApi, snapshot_download

from utils.reproduction import file_sha256, resolve_path, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="weights/gpt2")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    directory = resolve_path(args.output)
    metadata = {"repo_id": "openai-community/gpt2"}
    revision = HfApi().model_info(metadata["repo_id"], revision=args.revision).sha
    snapshot_download(metadata["repo_id"], revision=revision, local_dir=directory, allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"])
    metadata["resolved_revision"] = revision
    files = [path for path in directory.iterdir() if path.is_file() and path.name not in {"preparation.json", "verification.json"}]
    metadata["sha256"] = {path.name: file_sha256(path) for path in files}
    write_json(directory / "preparation.json", metadata)
    print(directory)


if __name__ == "__main__":
    main()
