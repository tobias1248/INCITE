"""Dataset-family-specific task builders."""

from tasks.builders.cifar10 import (
    JsonShapPixelProvider as Cifar10JsonShapPixelProvider,
    cifar10_cal_shap_specs,
    cifar10_transformer_random,
    cifar10_transformer_shap,
)
from tasks.builders.fashion_mnist import (
    JsonShapPixelProvider as FashionMnistJsonShapPixelProvider,
    fashion_mnist_transformer_random,
    fashion_mnist_transformer_shap,
    fashion_mnist_transformer_shap_calculate_all,
)
from tasks.builders.legacy import (
    imdb_shap_1_2_3_4_8_range02,
    imdb_transformer_shap_1_2_3_4_8_range02,
    mnist_lstm_1_2_3_4_8_range02,
    mnist_lstm_15_1_2_3_4_8_range02,
    pyct_random_1_4_8_16_32,
    pyct_rnn_random_1_4_8_16_32,
    pyct_rnn_shap_1_4_8_16_32,
    pyct_shap_1_4_8_16_32,
    sentiment_lstm_lstm_15_1_2_3_4_8_range02,
    stock_random_1_2_3_4_8_range02,
    stock_shap_1_2_3_4_8_limit_range02,
)
from tasks.builders.mnist import (
    JsonShapPixelProvider as MnistJsonShapPixelProvider,
    mnist_transformer_random,
    mnist_transformer_shap,
    mnist_transformer_shap_calculate_all,
)

__all__ = [
    "Cifar10JsonShapPixelProvider",
    "FashionMnistJsonShapPixelProvider",
    "MnistJsonShapPixelProvider",
    "cifar10_cal_shap_specs",
    "cifar10_transformer_random",
    "cifar10_transformer_shap",
    "fashion_mnist_transformer_random",
    "fashion_mnist_transformer_shap",
    "fashion_mnist_transformer_shap_calculate_all",
    "imdb_shap_1_2_3_4_8_range02",
    "imdb_transformer_shap_1_2_3_4_8_range02",
    "mnist_lstm_1_2_3_4_8_range02",
    "mnist_lstm_15_1_2_3_4_8_range02",
    "mnist_transformer_random",
    "mnist_transformer_shap",
    "mnist_transformer_shap_calculate_all",
    "pyct_random_1_4_8_16_32",
    "pyct_rnn_random_1_4_8_16_32",
    "pyct_rnn_shap_1_4_8_16_32",
    "pyct_shap_1_4_8_16_32",
    "sentiment_lstm_lstm_15_1_2_3_4_8_range02",
    "stock_random_1_2_3_4_8_range02",
    "stock_shap_1_2_3_4_8_limit_range02",
]
