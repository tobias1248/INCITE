"""Validate and snapshot target SHAP caches for isolated ACES campaigns.

This command reads local data only and never computes SHAP or loads a model.
The attack runtime still verifies the cached target against its prediction.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from datasets.common import select_background_per_class
from datasets.keras_cache import get_keras_datasets_dir
from explainability.shap_contract import (
    fingerprint_array,
    fingerprint_file,
    load_target_class_cache,
)


SCHEMA_VERSION = "aces-shap-cache-snapshot-v1"


def _digest(path: Path) -> str:
    return fingerprint_file(path)["sha256"]


def prepare_cache(*, mode: str, output_root: Path, manifest: Path,
                  models: Sequence[str], first_n: int, model_root: Path,
                  dataset_cache: Path, background_per_class: int = 3,
                  background_seed: int = 2233,
                  source_root: Optional[Path] = None) -> Dict[str, Any]:
    """Create or verify a receipt covering exactly the requested model/case set.

    Reuse copies bytes into exclusive new files. Failed preparation can leave a
    partial snapshot for inspection, but never publishes a successful receipt.
    Verification needs only the snapshot, not its original source directory.
    """
    if mode not in ("refresh", "reuse", "verify"):
        raise ValueError("Invalid SHAP cache mode")
    if not models or len(set(models)) != len(models) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", model) for model in models
    ):
        raise ValueError("Invalid or duplicate model names")
    if type(first_n) is not int or first_n < 1 or type(background_per_class) is not int or background_per_class < 1:
        raise ValueError("Case/background counts must be positive integers")
    if output_root.expanduser().is_symlink():
        raise ValueError("SHAP snapshot root must not be a symlink")
    output_root = output_root.expanduser().resolve()
    manifest = manifest.expanduser().resolve()
    dataset_cache = dataset_cache.expanduser().resolve()
    if manifest == output_root or output_root in manifest.parents:
        raise ValueError("SHAP receipt must be outside the cache tree")
    if mode != "verify" and manifest.exists():
        raise ValueError("SHAP cache receipt already exists")
    if mode == "reuse":
        if source_root is None:
            raise ValueError("Reuse requires a source root")
        source_root = source_root.expanduser().resolve()
        if (source_root == output_root or source_root in output_root.parents
                or output_root in source_root.parents):
            raise ValueError("Source and snapshot cache roots must not overlap")
        if manifest == source_root or source_root in manifest.parents:
            raise ValueError("SHAP receipt must be outside the source tree")
        if not source_root.is_dir():
            raise ValueError("SHAP reuse source is missing: {}".format(source_root))
        if output_root.exists():
            raise ValueError("SHAP snapshot root already exists")

    dataset_sha256 = _digest(dataset_cache)
    with dataset_cache.open("rb") as handle:
        batch = pickle.load(handle, encoding="bytes")
    pixels = np.asarray(batch[b"data"])
    labels = np.asarray(batch[b"labels"])
    if (pixels.ndim != 2 or pixels.shape[1] != 3072 or pixels.dtype != np.uint8
            or labels.shape != (len(pixels),) or first_n > len(pixels)):
        raise ValueError("Invalid local CIFAR10 data or case count")
    images = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255
    background = fingerprint_array(select_background_per_class(
        images, labels, per_class=background_per_class, seed=background_seed,
    ))
    settings = {"models": list(models), "first_n": first_n,
                "background_per_class": background_per_class,
                "background_seed": background_seed, "explainer_type": "gradient"}
    model_identity = {model: fingerprint_file(model_root / (model + ".h5")) for model in models}
    identity = {"schema_version": SCHEMA_VERSION, "settings": settings,
                "output_root": str(output_root), "dataset_sha256": dataset_sha256,
                "models": model_identity}
    expected_paths = {"{}/shap_value_{}.json".format(model, index)
                      for model in models for index in range(first_n)}
    if mode == "verify":
        receipt = json.loads(manifest.read_text())
        if not isinstance(receipt, dict) or any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError("Prepared SHAP cache identity differs")
        if not isinstance(receipt.get("files"), dict) or set(receipt["files"]) != expected_paths:
            raise ValueError("Prepared SHAP cache file set differs")
        for relative in sorted(expected_paths):
            path = output_root / relative
            if path.is_symlink() or path.parent.is_symlink() or _digest(path) != receipt["files"][relative]:
                raise ValueError("Prepared SHAP cache changed: {}".format(path))
        return receipt

    files = {}
    read_root = source_root if mode == "reuse" else output_root
    for model in models:
        for index in range(first_n):
            relative = "{}/shap_value_{}.json".format(model, index)
            path = read_root / relative
            if mode == "refresh" and (path.is_symlink() or path.parent.is_symlink()):
                raise ValueError("SHAP snapshot must contain ordinary files: {}".format(path))
            expected = {"case_index": index, "model": model_identity[model],
                        "input": fingerprint_array(images[index:index + 1]),
                        "background": background, "explainer_type": "gradient",
                        "background_per_class": background_per_class,
                        "background_seed": background_seed, "class_count": 10}
            before = _digest(path)
            try:
                _, values = load_target_class_cache(path, expected_identity=expected, case_index=index)
                if not any(key.split("_", 1)[0].isdigit() for key in values):
                    raise ValueError("Cache lacks internal-layer values")
            except ValueError as exc:
                raise ValueError("Incompatible SHAP cache {}: {}".format(path, exc)) from exc
            if mode == "reuse":
                destination = output_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as handle:
                    handle.write(path.read_bytes())
            else:
                destination = path
            if _digest(destination) != before or _digest(path) != before:
                raise ValueError("SHAP cache changed during preparation: {}".format(path))
            files[relative] = before
            print("SHAP validated {}/{}: {}".format(len(files), len(expected_paths), relative), flush=True)
    receipt = dict(identity, mode=mode, source_root=str(read_root),
                   dataset_cache=str(dataset_cache), files=files)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("refresh", "reuse", "verify"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--first-n", type=int, default=100)
    parser.add_argument("--model-root", type=Path, default=Path("model"))
    parser.add_argument("--dataset-cache", type=Path)
    parser.add_argument("--background-per-class", type=int, default=3)
    parser.add_argument("--background-seed", type=int, default=2233)
    parser.add_argument("--source-root", type=Path)
    args = vars(parser.parse_args(argv))
    args["dataset_cache"] = args["dataset_cache"] or get_keras_datasets_dir() / "cifar-10-batches-py/test_batch"
    try:
        receipt = prepare_cache(**args)
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as exc:
        parser.exit(1, "SHAP cache preparation failed: {}\n".format(exc))
    print("SHAP cache {}: {} files".format(args["mode"], len(receipt["files"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
