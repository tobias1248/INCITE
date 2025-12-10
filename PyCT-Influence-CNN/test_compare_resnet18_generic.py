#!/usr/bin/env python3
import argparse
import numpy as np

import dnn_predict_common_generic as dpc
from utils_out.dataset import MnistDataset
from tensorflow.image import resize

_DATASET_CACHE = {}


def _load_mnist_gray():
    ds = MnistDataset()
    return ds.x_test.astype("float32"), "mnist_gray"


def _load_cifar10_rgb():
    from tensorflow.keras.datasets import cifar10

    (_, _), (x_test, _) = cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = resize(x_test, (224, 224)).numpy()
    return x_test, "cifar10_rgb"


DATASET_LOADERS = {
    "mnist_gray": _load_mnist_gray,
    "cifar10_rgb": _load_cifar10_rgb,
}


def _get_dataset(dataset_name):
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(f"未知的 dataset: {dataset_name}")
    if dataset_name not in _DATASET_CACHE:
        data, name = DATASET_LOADERS[dataset_name]()
        _DATASET_CACHE[dataset_name] = data
    return _DATASET_CACHE[dataset_name]


def load_sample(sample_idx, dataset_name):
    data = _get_dataset(dataset_name)
    return data[sample_idx]


def image_to_input_dict(img):
    h, w, c = img.shape
    in_dict = {}
    for i in range(h):
        for j in range(w):
            for k in range(c):
                in_dict[f"v_{i}_{j}_{k}"] = float(img[i, j, k])
    return in_dict


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


def compare_logits(model_path,
                   sample_idx,
                   atol,
                   input_shape=None,
                   num_classes=None,
                   dataset_name="mnist_gray",
                   verbose=True,
                   init_verbose=False):
    img = load_sample(sample_idx, dataset_name)
    sample_shape = tuple(int(dim) for dim in img.shape) if img.ndim == 3 else (
        img.shape[0], img.shape[1], 1)
    target_shape = input_shape or sample_shape
    img = adapt_image_channels(img, target_shape)
    effective_input_shape = target_shape or tuple(int(dim)
                                                  for dim in img.shape)
    in_dict = image_to_input_dict(img)

    keras_model = dpc.load_keras_model(
        model_path, input_shape_override=effective_input_shape, num_classes_override=num_classes)
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
        choices=["mnist_gray", "cifar10_rgb"],
        default="mnist_gray",
        help="選擇資料集：mnist_gray（灰階）或 cifar10_rgb（224x224 RGB）"
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
    for offset in range(total):
        idx = args.sample_idx + offset
        ok, diff, argmax_match = compare_logits(
            args.model_path,
            idx,
            args.atol,
            input_shape,
            num_classes,
            dataset_name=dataset_name,
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
