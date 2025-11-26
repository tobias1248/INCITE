from keras.datasets import fashion_mnist
import keras
from keras import layers, models


def residual_block(x, filters, downsample=False, name=None):
    """
    一個最基本的 ResNet block:
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN + shortcut -> ReLU
    如果 downsample=True 或 channel 不同，就用 1x1 conv 做 shortcut。
    """
    stride = 2 if downsample else 1
    shortcut = x

    y = layers.Conv2D(filters, kernel_size=3, strides=stride,
                      padding="same", use_bias=False,
                      name=None if name is None else name + "_conv1")(x)
    y = layers.BatchNormalization(
        name=None if name is None else name + "_bn1")(y)
    y = layers.Activation(
        "relu", name=None if name is None else name + "_relu1")(y)

    y = layers.Conv2D(filters, kernel_size=3, strides=1,
                      padding="same", use_bias=False,
                      name=None if name is None else name + "_conv2")(y)
    y = layers.BatchNormalization(
        name=None if name is None else name + "_bn2")(y)

    # 如果需要改變維度或 downsample，就用 1x1 conv 處理 shortcut
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=stride,
                                 padding="same", use_bias=False,
                                 name=None if name is None else name + "_conv_short")(shortcut)
        shortcut = layers.BatchNormalization(
            name=None if name is None else name + "_bn_short")(shortcut)

    out = layers.Add(name=None if name is None else name +
                     "_add")([y, shortcut])
    out = layers.Activation(
        "relu", name=None if name is None else name + "_relu2")(out)
    return out


def build_resnet_fashion_mnist(input_shape=(28, 28, 1), num_classes=10):
    inputs = keras.Input(shape=input_shape, name="input")

    # stem
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding="same",
                      use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    # stage 1: 32 filters, 不降採樣
    x = residual_block(x, 32, downsample=False, name="stage1_block1")
    x = residual_block(x, 32, downsample=False, name="stage1_block2")

    # stage 2: 64 filters, 一開始降採樣 (stride=2)
    x = residual_block(x, 64, downsample=True, name="stage2_block1")
    x = residual_block(x, 64, downsample=False, name="stage2_block2")

    # stage 3: 128 filters, 再降一次
    x = residual_block(x, 128, downsample=True, name="stage3_block1")
    x = residual_block(x, 128, downsample=False, name="stage3_block2")

    # global average pooling + classifier
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = models.Model(inputs=inputs, outputs=outputs,
                         name="ResNet_FashionMNIST")
    return model


def train_and_save(model_path="simple_resnet_fashion_mnist.h5"):
    # 這裡改成 fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # 正規化 + 增加 channel 維度
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]   # (batch, 28, 28, 1)
    x_test = x_test[..., None]

    model = build_resnet_fashion_mnist()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=5,          # 想快一點就改小
        validation_split=0.1,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test acc = {test_acc:.4f}")

    model.save(model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    train_and_save("models/simple_resnet_fashion_mnist.h5")
