#!/usr/bin/env python3
"""
Offline utility to prepare a small ImageNet-mini subset with fixed size samples.

Reads images one by one (to keep memory usage low),只保留「原生尺寸 >= 目標尺寸」
的影像，並縮放 (resize) 到目標大小（不放大小於目標的影像），存入 utils_out/dataset 方便重複使用。
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tensorflow.keras.utils import img_to_array, load_img, array_to_img


def _iter_image_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def _collect_subset(
    source_root: Path,
    target_size: Tuple[int, int],
    sample_count: int,
) -> Tuple[np.ndarray, List[str]]:
    images = []
    picked = []
    target_h, target_w = int(target_size[0]), int(target_size[1])
    target_shape = (target_h, target_w, 3)
    skipped_too_small = 0
    skipped_wrong_channels = 0
    resized_from_larger = 0

    for path in _iter_image_files(source_root):
        img = load_img(path)
        arr = img_to_array(img)
        if arr.ndim != 3 or arr.shape[2] != 3:
            skipped_wrong_channels += 1
            continue
        h, w, _ = arr.shape
        if h < target_h or w < target_w:
            skipped_too_small += 1
            continue
        if (h, w) != (target_h, target_w):
            resized_from_larger += 1
            arr = img_to_array(img.resize((target_w, target_h)))
        arr = arr.astype("float32") / 255.0
        images.append(arr)
        picked.append(str(path.relative_to(source_root)))
        if len(images) >= sample_count:
            break
    if len(images) < sample_count:
        raise RuntimeError(
            f"只找到 {len(images)} 張符合條件 (原圖 >= {target_h}x{target_w} 並縮放為 {target_shape}) 的影像，"
            f"無法滿足要求的 {sample_count} 張（跳過 {skipped_too_small} 張過小、"
            f"{skipped_wrong_channels} 張通道不符，成功從較大影像縮放 {resized_from_larger} 張）。"
        )
    return np.stack(images), picked


def build_subset(
    source_dir: Path,
    output_dir: Path,
    sample_count: int,
    target_size: Tuple[int, int],
):
    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_images_root = output_dir / "images"
    output_images_root.mkdir(parents=True, exist_ok=True)

    data, picked_relative_paths = _collect_subset(
        source_dir, target_size=target_size, sample_count=sample_count
    )

    # 將處理後的影像也存成檔案，方便 compare 腳本直接用目錄
    for rel_path, arr in zip(picked_relative_paths, data):
        out_path = output_images_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        array_to_img(arr).save(out_path)

    subset_path = output_dir / "images.npy"
    np.save(subset_path, data)

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(
            {
                "source_dir": str(source_dir),
                "output_file": str(subset_path),
                "saved_images_dir": str(output_images_root),
                "count": int(sample_count),
                "target_size": list(target_size),
                "mode": "native_ge_target_resize_down_no_upscale",
                "picked_files": picked_relative_paths,
            },
            mf,
            ensure_ascii=False,
            indent=2,
        )

    print(f"完成：儲存 {sample_count} 張影像到 {subset_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="產生固定尺寸的 ImageNet-mini 子集（預設 224x224，100 張）"
    )
    default_root = (
        Path(__file__).resolve().parents[1]
        / "utils_out"
        / "dataset"
        / "imagenet-mini"
    )
    default_output = (
        Path(__file__).resolve().parents[1]
        / "utils_out"
        / "dataset"
        / "imagenet-mini-224-subset"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=default_root,
        help="ImageNet-mini 原始資料夾路徑",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help="輸出子集儲存資料夾（會建立 images.npy 與 metadata.json）",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="子集取樣張數（預設 100）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="輸出影像高度",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="輸出影像寬度",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_subset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        sample_count=args.samples,
        target_size=(args.height, args.width),
    )


if __name__ == "__main__":
    main()
