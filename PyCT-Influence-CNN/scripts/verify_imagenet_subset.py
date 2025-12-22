#!/usr/bin/env python3
"""
Verify that the sampled ImageNet-mini subset really contains 224x224x3 images
（預期已經經過前處理/縮放後的成品檔 images.npy）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.utils import img_to_array, load_img


def parse_args():
    parser = argparse.ArgumentParser(
        description="檢查 imagenet-mini 子集 images.npy 是否全部為 224x224x3"
    )
    default_dir = (
        Path(__file__).resolve().parents[1]
        / "utils_out"
        / "dataset"
        / "imagenet-mini-224-subset"
    )
    parser.add_argument(
        "--subset-dir",
        type=Path,
        default=default_dir,
        help="存放 images.npy/metadata.json 的資料夾",
    )
    parser.add_argument(
        "--expected-shape",
        type=str,
        default="224,224,3",
        help="期望的影像形狀，預設 224,224,3",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    subset_dir = args.subset_dir.resolve()
    metadata_path = subset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 metadata：{metadata_path}")

    expected_shape = tuple(int(dim) for dim in args.expected_shape.split(","))
    with open(metadata_path, "r", encoding="utf-8") as mf:
        meta = json.load(mf)

    images_path = subset_dir / "images.npy"
    if not images_path.exists():
        raise FileNotFoundError(f"找不到 images.npy：{images_path}")

    data = np.load(images_path)
    if data.ndim != 4:
        raise ValueError(f"images.npy 維度應為 (N, H, W, C)，目前 {data.shape}")

    mismatches = []
    for idx in range(data.shape[0]):
        shape = tuple(data[idx].shape)
        if shape != expected_shape:
            mismatches.append((idx, shape))

    total = data.shape[0]
    if mismatches:
        print(f"總樣本 {total} 張，其中 {len(mismatches)} 張與期望形狀 {expected_shape} 不符：")
        for idx, shape in mismatches:
            print(f" - sample #{idx}: {shape}")
    else:
        print(f"檢查完成，共 {total} 張，全部符合形狀 {expected_shape}")


if __name__ == "__main__":
    main()
