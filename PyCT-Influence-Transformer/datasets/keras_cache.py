from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence


class DatasetAvailabilityError(RuntimeError):
    """Raised when a Keras dataset is unavailable in the local cache."""


def _truthy_env(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def prepare_keras_cache_env() -> Path:
    keras_home = os.environ.get("PYCT_KERAS_HOME")
    if keras_home:
        os.environ["KERAS_HOME"] = keras_home
    resolved_home = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras")).expanduser()
    return resolved_home


def get_keras_datasets_dir() -> Path:
    return prepare_keras_cache_env() / "datasets"


def allow_dataset_downloads() -> bool:
    return _truthy_env("PYCT_ALLOW_DATASET_DOWNLOAD")


def _format_missing(paths: Sequence[Path]) -> str:
    preview = ", ".join(str(path) for path in paths[:3])
    if len(paths) > 3:
        preview += ", ..."
    return preview


def ensure_local_dataset_files(dataset_name: str, relative_paths: Iterable[str]) -> Path:
    datasets_dir = get_keras_datasets_dir()
    missing = [datasets_dir / relative for relative in relative_paths if not (datasets_dir / relative).exists()]
    if missing and not allow_dataset_downloads():
        raise DatasetAvailabilityError(
            f"{dataset_name} dataset cache is missing ({_format_missing(missing)}). "
            "Pre-populate the Keras cache under ~/.keras/datasets or set "
            "PYCT_KERAS_HOME to a local cache directory. To allow on-demand "
            "downloads, set PYCT_ALLOW_DATASET_DOWNLOAD=1."
        )
    return datasets_dir


def resolve_mnist_path(filename: str = "mnist.npz") -> str:
    datasets_dir = get_keras_datasets_dir()
    local_path = datasets_dir / filename
    if local_path.is_file():
        return str(local_path)
    if allow_dataset_downloads():
        return filename
    raise DatasetAvailabilityError(
        f"MNIST dataset cache is missing ({local_path}). Pre-populate the Keras "
        "cache under ~/.keras/datasets or set PYCT_KERAS_HOME to a local cache "
        "directory. To allow on-demand downloads, set PYCT_ALLOW_DATASET_DOWNLOAD=1."
    )


__all__ = [
    "DatasetAvailabilityError",
    "allow_dataset_downloads",
    "ensure_local_dataset_files",
    "get_keras_datasets_dir",
    "prepare_keras_cache_env",
    "resolve_mnist_path",
]
