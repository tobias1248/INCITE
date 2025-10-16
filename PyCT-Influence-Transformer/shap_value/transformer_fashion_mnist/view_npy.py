import numpy as np

data = np.load("./transformer_fashion_mnist_shap_values.npy")
np.set_printoptions(threshold=np.inf)
print(data[0][0][0])

# data = np.load(
#     "transformer_fashion_mnist_shap_values.npy",
#     allow_pickle=True,
# )
# img0 = data[0]              # shape (28, 28, 1, 10)
# print(img0)

