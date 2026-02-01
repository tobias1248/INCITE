import itertools
import numpy as np
import os

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
        background_dataset_for_shap = self.x_test[:100]

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
        background_dataset_for_shap = self.x_test[:5]
        
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
        background_dataset_for_shap = self.x_test[:5]

        for i, j, k in attack_pixels:
            key = f"v_{i}_{j}_{k}"
            con_dict[key] = 1

        return in_dict, con_dict, input_for_shap, background_dataset_for_shap

class ImagenetMiniSubsetDataset:
    def __init__(self, dataset_root=None, subset_dir="imagenet-mini-224-subset"):
        from tensorflow.keras.utils import load_img, img_to_array

        base = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        root = dataset_root if dataset_root else os.path.join(base, "utils_out", "dataset", subset_dir)

        npy_path = os.path.join(root, "images.npy")
        images_dir = os.path.join(root, "images")

        if os.path.isfile(npy_path):
            self.x_test = np.load(npy_path).astype("float32")
            return

        files = []
        for dirpath, _, filenames in os.walk(images_dir):
            for fname in filenames:
                if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    files.append(os.path.join(dirpath, fname))
        files.sort()

        if not files:
            raise FileNotFoundError(f"No images found under {images_dir}")

        arrs = []
        for f in files:
            arr = img_to_array(load_img(f, target_size=(224, 224)))  # force 224x224x3
            arrs.append(arr.astype("float32") / 255.0)
        self.x_test = np.stack(arrs)

    def get_imagenet_mini_test_data(self, idx):
        in_dict, con_dict = {}, {}
        test_img = self.x_test[idx]
        for i, j, k in itertools.product(
            range(test_img.shape[0]), range(test_img.shape[1]), range(test_img.shape[2])
        ):
            key = f"v_{i}_{j}_{k}"
            in_dict[key] = float(test_img[i][j][k])
            con_dict[key] = 0
        return in_dict, con_dict

    def get_imagenet_mini_test_data_and_set_condict(self, idx, attack_pixels):
        in_dict, con_dict = self.get_imagenet_mini_test_data(idx)
        input_for_shap = self.x_test[idx]
        background_dataset_for_shap = self.x_test[:5]
        for i, j, k in attack_pixels:
            con_dict[f"v_{i}_{j}_{k}"] = 1
        return in_dict, con_dict, input_for_shap, background_dataset_for_shap