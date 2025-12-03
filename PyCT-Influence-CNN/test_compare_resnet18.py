#!/usr/bin/env python3
import argparse
import numpy as np

import dnn_predict_common as dpc
from utils_out.dataset import MnistDataset

_dataset_cache = None


def load_sample(sample_idx):
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = MnistDataset()
    dataset = _dataset_cache
    in_dict, _ = dataset.get_mnist_test_data(sample_idx)
    img = dataset.x_test[sample_idx]
    return in_dict, img


def compare_logits(model_path,
                   sample_idx,
                   atol,
                   input_shape,
                   num_classes,
                   verbose=True,
                   init_verbose=False):
    keras_model = dpc.load_keras_model(
        model_path, input_shape_override=input_shape, num_classes_override=num_classes)
    dpc.init_model(model_path, verbose=init_verbose)

    in_dict, img = load_sample(sample_idx)

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
        type=_parse_shape,
        default=(28, 28, 1),
        help="模型輸入形狀，格式為 h,w,c（預設 28,28,1）"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="分類數（預設 10）"
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
    for offset in range(total):
        idx = args.sample_idx + offset
        ok, diff, argmax_match = compare_logits(
            args.model_path,
            idx,
            args.atol,
            args.input_shape,
            args.num_classes,
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
