from __future__ import annotations

import numpy as np

from datasets.common import enable_attack_pixels, tensor_to_in_dict_and_con_dict


class RNN_MnistDataset:
    def __init__(self) -> None:
        from tensorflow.keras.datasets.mnist import load_data

        (x_train, y_train), (x_test, _y_test) = load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        del x_train
        del y_train

        self.x_test = np.expand_dims(x_test, -1)

    def get_mnist_test_data(self, idx):
        test_img = self.x_test[idx]
        if test_img.ndim == 3 and test_img.shape[-1] == 1:
            test_img = np.squeeze(test_img, axis=-1)
        return tensor_to_in_dict_and_con_dict(test_img)

    def get_mnist_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_mnist_test_data(idx)
        enable_attack_pixels(con_dict, attack_pixels)
        return in_dict, con_dict


class MSstock_Dataset:
    def __init__(self) -> None:
        data = np.load(
            "utils/dataset/LSTM_DenseF_day20_09262/data_sc.npy",
            allow_pickle=True,
        )[None][0]
        self.x_test = data["X_test"]

    def get_stock_test_data(self, idx):
        return tensor_to_in_dict_and_con_dict(self.x_test[idx])

    def get_stock_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_stock_test_data(idx)
        enable_attack_pixels(con_dict, attack_pixels)
        return in_dict, con_dict


class IMDB_Dataset:
    def __init__(self) -> None:
        data = np.load("utils/dataset/sent_emb_sample200.npy", allow_pickle=True)[None][0]
        self.x_test = data["X_test"]

    def get_imdb_test_data(self, idx):
        return tensor_to_in_dict_and_con_dict(self.x_test[idx])

    def get_imdb_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_imdb_test_data(idx)
        enable_attack_pixels(con_dict, attack_pixels)
        return in_dict, con_dict


__all__ = ["RNN_MnistDataset", "MSstock_Dataset", "IMDB_Dataset"]
