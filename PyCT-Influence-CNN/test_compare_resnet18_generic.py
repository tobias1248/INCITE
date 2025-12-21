#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

import dnn_predict_common_generic as dpc
from utils_out.dataset import MnistDataset, ImagenetMiniDataset
from tensorflow.keras.utils import img_to_array, load_img

_DATASET_CACHE = {}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
_DEFAULT_IMAGENET_MINI_ROOT = Path(__file__).resolve().parent / "utils_out" / "dataset" / "imagenet-mini"


def _load_images_from_dir(dataset_root, dataset_name, max_samples=None):
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(
            f"{dataset_name} 數據集目錄不存在：'{root}'，請先下載 mini 版或指定正確路徑")
    image_files = sorted([
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
    ])
    if not image_files:
        raise FileNotFoundError(
            f"{dataset_name} 沒找到影像檔案，請檢查資料集相關目錄")
    if max_samples is not None:
        image_files = image_files[:max_samples]

    images = []
    shapes = set()
    for path in image_files:
        img = load_img(path)
        arr = img_to_array(img).astype("float32") / 255.0
        images.append(arr)
        shapes.add(arr.shape)

    if len(shapes) != 1:
        raise ValueError(f"{dataset_name} 影像尺寸不一致，找到的尺寸集合: {shapes}")
    return np.stack(images), dataset_name


def _load_mnist_gray(dataset_root=None, max_samples=None):
    ds = MnistDataset()
    data = ds.x_test.astype("float32")
    if max_samples is not None:
        data = data[:max_samples]
    return data, "mnist_gray"


def _load_cifar10_rgb(dataset_root=None, max_samples=None):
    from tensorflow.keras.datasets import cifar10

    (_, _), (x_test, _) = cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    if max_samples is not None:
        x_test = x_test[:max_samples]
    return x_test, "cifar10_rgb"


def _load_imagenet_mini_rgb(dataset_root=None, max_samples=None):
    ds = ImagenetMiniDataset(
        dataset_root=dataset_root or _DEFAULT_IMAGENET_MINI_ROOT,
        max_samples=max_samples,
    )
    return ds.x_test, "imagenet_mini_rgb"


def _load_custom_rgb(dataset_root=None, max_samples=None):
    if not dataset_root:
        raise ValueError("custom_rgb 請提供 dataset_root （--dataset-root）")
    return _load_images_from_dir(dataset_root, "custom_rgb", max_samples=max_samples)


DATASET_LOADERS = {
    "mnist_gray": _load_mnist_gray,
    "cifar10_rgb": _load_cifar10_rgb,
    "imagenet_mini_rgb": _load_imagenet_mini_rgb,
    "custom_rgb": _load_custom_rgb,
}


def _get_dataset(dataset_name, dataset_root=None, max_samples=None):
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"未知的 dataset: {dataset_name}")
    cache_key = (dataset_name, str(dataset_root) if dataset_root else None,
                 max_samples)
    if cache_key not in _DATASET_CACHE:
        data, name = DATASET_LOADERS[dataset_name](
            dataset_root=dataset_root,
            max_samples=max_samples,
        )
        _DATASET_CACHE[cache_key] = data
    return _DATASET_CACHE[cache_key]


def load_sample(sample_idx, dataset_name, dataset_root=None, max_samples=None):
    data = _get_dataset(dataset_name, dataset_root=dataset_root,
                        max_samples=max_samples)
    if sample_idx >= len(data):
        raise IndexError(
            f"{dataset_name} 包含 {len(data)} 張，被選索引 {sample_idx}")
    return data[sample_idx]


def image_to_input_dict(img):
    h, w, c = img.shape
    in_dict = {}
    for i in range(h):
        for j in range(w):
            for k in range(c):
                in_dict[f"v_{i}_{j}_{k}"] = float(img[i, j, k])
    return in_dict

# 比對Channel與sample size是否匹配
def adapt_image_channels(img, target_shape):
    if img.ndim == 2:
        img = img[:, :, None]
    if target_shape is None or len(target_shape) < 3:
        return img
    target_c = target_shape[2]
    if target_c is None or img.shape[2] == target_c:
        return img
    if img.shape[2] == 1 and target_c == 3:
        return np.repeat(img, target_c, axis=2)
    raise ValueError(
        f"無法將輸入通道 {img.shape[2]} 匹配到模型需求 {target_c}")


def _assert_shape_compatible(actual_shape, expected_shape, context, dataset_name):
    if expected_shape is None:
        return
    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{context}: {dataset_name} 數據使用維度 {actual_shape} 與要求 {expected_shape} 不符")
    mismatches = []
    for idx, (act, exp) in enumerate(zip(actual_shape, expected_shape)):
        if exp is None:
            continue
        if int(act) != int(exp):
            mismatches.append((idx, act, exp))
    if mismatches:
        raise ValueError(
            f"{context}: {dataset_name} 形狀不匹配；現有 {actual_shape}，要求 {expected_shape}")


def compare_logits(model_path,
                   sample_idx,
                   atol,
                   input_shape=None,
                   num_classes=None,
                   dataset_name="mnist_gray",
                   dataset_root=None,
                    max_dataset_samples=None,
                   verbose=True,
                   init_verbose=False):
    img = load_sample(
        sample_idx,
        dataset_name,
        dataset_root=dataset_root,
        max_samples=max_dataset_samples,
    )
    sample_shape = tuple(int(dim) for dim in img.shape) if img.ndim == 3 else (
        img.shape[0], img.shape[1], 1) ## 推斷sample_size
    target_shape = input_shape or sample_shape
    img = adapt_image_channels(img, target_shape)
    sample_shape = tuple(int(dim) for dim in img.shape)
    _assert_shape_compatible(sample_shape, target_shape,
                             "dataset vs input_shape", dataset_name)

    keras_model = dpc.load_keras_model(
        model_path, input_shape_override=input_shape, num_classes_override=num_classes)
    model_input_shape = tuple(
        None if dim is None else int(dim) for dim in keras_model.input_shape[1:])
    _assert_shape_compatible(sample_shape, model_input_shape,
                             "模型 input shape 實例檢查", dataset_name)

    effective_input_shape = input_shape or model_input_shape or sample_shape
    if effective_input_shape is None or any(dim is None for dim in effective_input_shape):
        effective_input_shape = tuple(
            sample_shape[idx] if effective_input_shape is None or effective_input_shape[idx] is None else int(effective_input_shape[idx])
            for idx in range(len(sample_shape)))

    in_dict = image_to_input_dict(img)

    dpc.init_model(model_path, verbose=init_verbose,
                   input_shape_override=effective_input_shape,
                   num_classes_override=num_classes)

    py_logits = np.asarray(dpc.predict_logits(**in_dict), dtype=np.float32)
    keras_logits = keras_model.predict(
        img[np.newaxis, ...], verbose=0).astype(np.float32)[0]

    logits_close = np.allclose(py_logits, keras_logits, atol=atol)
    argmax_match = int(np.argmax(py_logits)) == int(np.argmax(keras_logits))
    max_abs_diff = float(np.max(np.abs(py_logits - keras_logits)))

    if verbose:
        print(f"模型: {model_path}")
        print(f"測試影像索引: {sample_idx}")
        print(f"logits 是否一致 (atol={atol}): {logits_close}")
        print(f"分類結果 argmax 是否一致: {argmax_match}")
        print(f"logits 最大絕對誤差: {max_abs_diff:.6e}")

        if not logits_close:
            print("警告: logits 差異過大，請逐層檢查 NNModel 實作。")

    return logits_close, max_abs_diff, argmax_match


def _parse_shape(shape_str):
    try:
        return tuple(int(dim) for dim in shape_str.split(","))
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"input-shape '{shape_str}' 格式錯誤，請使用逗號分隔的整數 (例如 28,28,1)"
        ) from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="比較 Keras ResNet18 與 NNModel 的 logits 是否一致")
    parser.add_argument(
        "--model-path",
        default="model/simple_resnet_fashion_mnist.h5",
        help="Keras 模型路徑，預設為 MNIST 版 ResNet-18")
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="MNIST 測試影像索引，預設 0")
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="比較 logits 時允許的絕對誤差"
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="（選填）模型輸入形狀 h,w,c。若不填則由模型自動推斷"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="（選填）分類數，若不填則由模型自動推斷"
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist_gray", "cifar10_rgb", "imagenet_mini_rgb", "custom_rgb"],
        default="mnist_gray",
        help="選擇資料集：mnist_gray（灰階）/ cifar10_rgb（原生尺寸）/ imagenet_mini_rgb（預期224x224 RGB，尺寸不符會報錯）/ custom_rgb（自備資料夾，尺寸不符會報錯）"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="選填：指定 imagenet_mini_rgb 或 custom_rgb 的本地資料集路徑"
    )
    parser.add_argument(
        "--max-dataset-samples",
        type=int,
        default=None,
        help="選填：載入資料集的最大樣本數"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="要連續測試的樣本數，會從 sample-idx 開始"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    total = args.num_samples
    results = []
    init_verbose = total == 1
    input_shape = _parse_shape(args.input_shape) if args.input_shape else None
    num_classes = args.num_classes
    dataset_name = args.dataset
    dataset_root = args.dataset_root
    max_dataset_samples = args.max_dataset_samples
    for offset in range(total):
        idx = args.sample_idx + offset
        ok, diff, argmax_match = compare_logits(
            args.model_path,
            idx,
            args.atol,
            input_shape,
            num_classes,
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            max_dataset_samples=max_dataset_samples,
            init_verbose=init_verbose,
        )
        results.append((idx, ok, diff, argmax_match))
        print("-" * 60)

    if total > 1:
        logits_matches = sum(1 for _, ok, _, _ in results if ok)
        argmax_matches = sum(1 for _, _, _, m in results if m)
        max_diff = max(diff for _, _, diff, _ in results)
        avg_diff = sum(diff for _, _, diff, _ in results) / total
        print("===== Summary =====")
        print(f"總樣本數: {total}")
        print(f"logits 完全一致數: {logits_matches} ({logits_matches/total:.2%})")
        print(f"分類一致數: {argmax_matches} ({argmax_matches/total:.2%})")
        print(f"logits 最大誤差: {max_diff:.6e}")
        print(f"logits 平均誤差: {avg_diff:.6e}")


if __name__ == "__main__":
    main()
