import itertools
import numpy as np
import os


def _get_background_per_class(default: int = 3) -> int:
    # Read per-class sample count from env; clamp to at least 1.
    try:
        value = int(os.environ.get("PYCT_BG_PER_CLASS", default))
    except ValueError:
        value = default
    return max(value, 1)

def _get_background_seed(default: int = 2233) -> int:
    # Read sampling seed from env; fall back to default if missing/invalid.
    try:
        return int(os.environ.get("PYCT_BG_SEED", default))
    except ValueError:
        return default


def _select_background_per_class(
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    per_class: int,
    seed: int = 0,
) -> np.ndarray:
    # Build a class-balanced background set by sampling k items per label.
    y_flat = np.asarray(y_test).reshape(-1)
    # Fixed seed for reproducibility across runs.
    rng = np.random.default_rng(seed)
    selected_indices = []
    for label in sorted(set(int(v) for v in y_flat)):
        candidates = np.where(y_flat == label)[0]
        if candidates.size == 0:
            continue
        if candidates.size <= per_class:
            # Not enough samples in this class; take all.
            chosen = candidates
        else:
            # Randomly sample k indices from this class.
            chosen = rng.choice(candidates, size=per_class, replace=False)
        selected_indices.extend(chosen.tolist())
    if not selected_indices:
        # Fallback if labels are missing/empty.
        return x_test[:per_class]
    # Return background images in the order of selected indices.
    return x_test[np.array(selected_indices)]

class MnistDataset:
    def __init__(self):
        from tensorflow.keras.datasets.mnist import load_data        
        
        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = load_data()
        
        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        
        del x_train
        del y_train
        
        # self.x_train = np.expand_dims(x_train, -1)
        self.x_test = np.expand_dims(x_test, -1)
        self.y_test = y_test

    def get_mnist_test_data(self, idx):        
        in_dict = dict()
        con_dict = dict()

        test_img = self.x_test[idx]
        input_for_shap = test_img
        background_dataset_for_shap = _select_background_per_class(
            self.x_test,
            self.y_test,
            per_class=_get_background_per_class(),
            seed=_get_background_seed(),
        )

        for i,j,k in itertools.product(
            range(test_img.shape[0]),
            range(test_img.shape[1]),
            range(test_img.shape[2])
        ):
            key = f"v_{i}_{j}_{k}"
            in_dict[key] = float(test_img[i][j][k])
            con_dict[key] = 0
        
        return in_dict, con_dict, input_for_shap, background_dataset_for_shap
    
    
    def get_mnist_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_mnist_test_data(idx)
        
        for i,j,k in attack_pixels:
            key = f"v_{i}_{j}_{k}"
            con_dict[key] = 1
        
        return in_dict, con_dict
        
    
class RNN_MnistDataset:
    def __init__(self):
        from tensorflow.keras.datasets.mnist import load_data        
        
        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = load_data()
        
        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        
        del x_train
        del y_train
        
        # self.x_train = np.expand_dims(x_train, -1)
        self.x_test = np.expand_dims(x_test, -1)

    def get_mnist_test_data(self, idx):        
        in_dict = dict()
        con_dict = dict()

        test_img = self.x_test[idx]

        for i,j in itertools.product(
            range(test_img.shape[0]),
            range(test_img.shape[1])
        ):
            key = f"v_{i}_{j}"
            in_dict[key] = float(test_img[i][j])
            con_dict[key] = 0
        
        return in_dict, con_dict
    
    
    def get_mnist_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_mnist_test_data(idx)
        
        for i,j in attack_pixels:
            key = f"v_{i}_{j}"
            con_dict[key] = 1
        
        return in_dict, con_dict

class MSstock_Dataset:
    def __init__(self):
               
        
        # Load the data and split it between train and test sets
        data = np.load(os.path.join('utils/dataset/LSTM_DenseF_day20_09262', 'data_sc.npy'), allow_pickle=True)[None][0]
        x_test = data['X_test']
        
        self.x_test = x_test

    def get_stock_test_data(self, idx):        
        in_dict = dict()
        con_dict = dict()

        test_data = self.x_test[idx]

        for i,j in itertools.product(
            range(test_data.shape[0]),
            range(test_data.shape[1])
        ):
            key = f"v_{i}_{j}"
            in_dict[key] = float(test_data[i][j])
            con_dict[key] = 0
        
        return in_dict, con_dict
    
    
    def get_stock_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_stock_test_data(idx)
        
        for i,j in attack_pixels:
            key = f"v_{i}_{j}"
            con_dict[key] = 1
        
        return in_dict, con_dict

class IMDB_Dataset:
    def __init__(self):
               
        
        # Load the data and split it between train and test sets
        data = np.load(os.path.join('utils/dataset/', 'sent_emb_sample200.npy'), allow_pickle=True)[None][0]
        x_test = data['X_test']
        
        self.x_test = x_test

    def get_imdb_test_data(self, idx):
        in_dict = dict()
        con_dict = dict()

        test_data = self.x_test[idx]

        for i,j in itertools.product(
            range(test_data.shape[0]),
            range(test_data.shape[1])
        ):
            key = f"v_{i}_{j}"
            in_dict[key] = float(test_data[i][j])
            con_dict[key] = 0
        
        return in_dict, con_dict
    
    
    def get_imdb_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_imdb_test_data(idx)
        
        for i,j in attack_pixels:
            key = f"v_{i}_{j}"
            con_dict[key] = 1
        
        return in_dict, con_dict

class FashionMnistDataset:
    def __init__(self):
        from tensorflow.keras.datasets.fashion_mnist import load_data        
        
        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = load_data()
        
        # Scale images to the [0, 1] range
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # Make sure images have shape (28, 28, 1)
        
        del x_train
        del y_train
        
        # self.x_train = np.expand_dims(x_train, -1)
        self.x_test = np.expand_dims(x_test, -1)
        self.y_test = y_test
        print("self.x_test.shape:",self.x_test.shape)

    def get_fashion_mnist_test_data(self, idx):        
        in_dict = dict()
        con_dict = dict()

        test_img = self.x_test[idx]

        for i,j,k in itertools.product(
            range(test_img.shape[0]),
            range(test_img.shape[1]),
            range(test_img.shape[2])
        ):
            key = f"v_{i}_{j}_{k}"
            in_dict[key] = float(test_img[i][j][k])
            con_dict[key] = 0
        
        return in_dict, con_dict
    
    
    def get_fashion_mnist_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_fashion_mnist_test_data(idx)
        input_for_shap = self.x_test[idx]
        background_dataset_for_shap = _select_background_per_class(
            self.x_test,
            self.y_test,
            per_class=_get_background_per_class(),
            seed=_get_background_seed(),
        )
        
        for i,j,k in attack_pixels:
            key = f"v_{i}_{j}_{k}"
            con_dict[key] = 1
        
        return in_dict, con_dict, input_for_shap, background_dataset_for_shap


class Cifar10Dataset:
    def __init__(self):
        from tensorflow.keras.datasets.cifar10 import load_data

        (x_train, y_train), (x_test, y_test) = load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        del x_train
        del y_train

        self.x_test = x_test
        self.y_test = y_test

    def get_cifar10_test_data(self, idx):
        in_dict = dict()
        con_dict = dict()

        test_img = self.x_test[idx]

        for i, j, k in itertools.product(
            range(test_img.shape[0]),
            range(test_img.shape[1]),
            range(test_img.shape[2]),
        ):
            key = f"v_{i}_{j}_{k}"
            in_dict[key] = float(test_img[i][j][k])
            con_dict[key] = 0

        return in_dict, con_dict

    def get_cifar10_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_cifar10_test_data(idx)
        input_for_shap = self.x_test[idx]
        background_dataset_for_shap = _select_background_per_class(
            self.x_test,
            self.y_test,
            per_class=_get_background_per_class(),
            seed=_get_background_seed(),
        )

        for i, j, k in attack_pixels:
            key = f"v_{i}_{j}_{k}"
            con_dict[key] = 1

        return in_dict, con_dict, input_for_shap, background_dataset_for_shap
