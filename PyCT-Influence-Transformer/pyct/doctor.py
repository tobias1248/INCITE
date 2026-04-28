from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from datasets.keras_cache import get_keras_datasets_dir


FindSpec = Callable[[str], object]
Which = Callable[[str], Optional[str]]

RUNTIME_PACKAGES = {
    "func-timeout": "func_timeout",
    "keras": "keras",
    "keras-resnet": "keras_resnet",
    "numpy": "numpy",
    "opencv-python": "cv2",
    "shap": "shap",
    "tensorflow": "tensorflow",
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    required: bool = True

    @property
    def ok(self) -> bool:
        return self.status in {"ok", "warn"} or not self.required

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "required": self.required,
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check local prerequisites for running PyCT experiments."
    )
    parser.add_argument(
        "--solver",
        default="cvc5",
        help="Solver executable expected on PATH (default: cvc5).",
    )
    parser.add_argument(
        "--model-dir",
        default="model",
        help="Directory containing .h5 model files (default: model).",
    )
    parser.add_argument(
        "--model-name",
        help="Optional model basename to verify under --model-dir, without .h5.",
    )
    parser.add_argument(
        "--dataset-cache",
        help="Optional Keras datasets cache directory. Defaults to PYCT_KERAS_HOME/KERAS_HOME resolution.",
    )
    parser.add_argument(
        "--shap-root",
        default="shap_value_all_layer",
        help="Directory containing SHAP artifacts (default: shap_value_all_layer).",
    )
    parser.add_argument(
        "--output-dir",
        default="exp",
        help="Experiment output directory to check for writability (default: exp).",
    )
    parser.add_argument(
        "--skip-runtime-packages",
        action="store_true",
        help="Skip import-discovery checks for Python runtime dependencies.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    return parser.parse_args(argv)


def check_python_version() -> CheckResult:
    version = sys.version_info
    label = f"{version.major}.{version.minor}.{version.micro}"
    if version.major == 3 and version.minor == 9:
        return CheckResult("python", "ok", f"Python {label}")
    return CheckResult(
        "python",
        "warn",
        f"Python {label}; project metadata targets >=3.9,<3.10",
        required=False,
    )


def check_runtime_packages(find_spec_fn: FindSpec = find_spec) -> List[CheckResult]:
    results = []
    for package_name, module_name in sorted(RUNTIME_PACKAGES.items()):
        spec = find_spec_fn(module_name)
        if spec is None:
            results.append(
                CheckResult(
                    f"package:{package_name}",
                    "fail",
                    f"module '{module_name}' is not import-discoverable",
                )
            )
        else:
            results.append(
                CheckResult(
                    f"package:{package_name}",
                    "ok",
                    f"module '{module_name}' is import-discoverable",
                )
            )
    return results


def check_solver(solver: str, which_fn: Which = shutil.which) -> CheckResult:
    resolved = which_fn(solver)
    if resolved:
        return CheckResult("solver", "ok", f"{solver} found at {resolved}")
    return CheckResult("solver", "fail", f"{solver} was not found on PATH")


def check_model_dir(model_dir: Path, model_name: Optional[str]) -> CheckResult:
    if not model_dir.is_dir():
        return CheckResult("model", "fail", f"model directory not found: {model_dir}")
    if model_name:
        model_path = model_dir / f"{model_name}.h5"
        if model_path.is_file():
            return CheckResult("model", "ok", f"model file found: {model_path}")
        return CheckResult("model", "fail", f"model file not found: {model_path}")

    models = sorted(path.name for path in model_dir.glob("*.h5"))
    if models:
        return CheckResult("model", "ok", f"{len(models)} .h5 model file(s) found")
    return CheckResult("model", "fail", f"no .h5 model files found under {model_dir}")


def check_dataset_cache(dataset_cache: Path) -> CheckResult:
    if dataset_cache.is_dir():
        return CheckResult("dataset-cache", "ok", f"cache directory found: {dataset_cache}")
    return CheckResult(
        "dataset-cache",
        "warn",
        f"cache directory not found: {dataset_cache}",
        required=False,
    )


def check_shap_root(shap_root: Path, model_name: Optional[str]) -> CheckResult:
    if not shap_root.is_dir():
        return CheckResult(
            "shap-root",
            "warn",
            f"SHAP root not found: {shap_root}",
            required=False,
        )
    if model_name:
        candidate = shap_root / model_name
        if candidate.exists():
            return CheckResult("shap-root", "ok", f"SHAP artifact path found: {candidate}")
        return CheckResult(
            "shap-root",
            "warn",
            f"SHAP artifact path not found for model: {candidate}",
            required=False,
        )
    return CheckResult("shap-root", "ok", f"SHAP root found: {shap_root}")


def check_output_dir(output_dir: Path) -> CheckResult:
    if output_dir.exists():
        if output_dir.is_dir() and _is_writable(output_dir):
            return CheckResult("output-dir", "ok", f"writable directory: {output_dir}")
        return CheckResult("output-dir", "fail", f"not a writable directory: {output_dir}")

    parent = output_dir.parent if output_dir.parent != Path("") else Path(".")
    if parent.is_dir() and _is_writable(parent):
        return CheckResult(
            "output-dir",
            "warn",
            f"directory does not exist but parent is writable: {output_dir}",
            required=False,
        )
    return CheckResult("output-dir", "fail", f"parent is not writable: {parent}")


def run_checks(
    args: argparse.Namespace,
    find_spec_fn: FindSpec = find_spec,
    which_fn: Which = shutil.which,
) -> List[CheckResult]:
    dataset_cache = (
        Path(args.dataset_cache).expanduser()
        if args.dataset_cache
        else get_keras_datasets_dir()
    )
    results = [
        check_python_version(),
        check_solver(args.solver, which_fn=which_fn),
        check_model_dir(Path(args.model_dir), args.model_name),
        check_dataset_cache(dataset_cache),
        check_shap_root(Path(args.shap_root), args.model_name),
        check_output_dir(Path(args.output_dir)),
    ]
    if not args.skip_runtime_packages:
        results.extend(check_runtime_packages(find_spec_fn=find_spec_fn))
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    results = run_checks(args)
    ok = all(result.ok for result in results)

    if args.json:
        payload = {
            "ok": ok,
            "checks": [result.to_dict() for result in results],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_text(results, ok=ok)
    return 0 if ok else 1


def _is_writable(path: Path) -> bool:
    return path.exists() and path.is_dir() and os.access(str(path), os.W_OK)


def _print_text(results: Iterable[CheckResult], ok: bool) -> None:
    for result in results:
        required = "required" if result.required else "optional"
        print(f"[{result.status.upper()}] {result.name} ({required}): {result.detail}")
    print(f"PyCT doctor {'passed' if ok else 'failed'}.")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CheckResult",
    "check_dataset_cache",
    "check_model_dir",
    "check_output_dir",
    "check_python_version",
    "check_runtime_packages",
    "check_shap_root",
    "check_solver",
    "main",
    "parse_args",
    "run_checks",
]
