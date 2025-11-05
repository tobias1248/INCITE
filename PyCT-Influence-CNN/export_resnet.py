#!/usr/bin/env python3
import os

import keras.layers
from keras.models import save_model


def main():
    target_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(target_dir, exist_ok=True)

    try:
        from keras_resnet.models import ResNet18
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency `keras-resnet`. "
            "Install it first, e.g. `pip install keras-resnet`."
        ) from exc

    input_shape = (28, 28, 1)
    classes = 10
    inputs = keras.layers.Input(shape=input_shape)

    print("Building ResNet18 (random initialization)…")
    model = ResNet18(
        inputs=inputs,
        classes=classes,
        include_top=True,
        freeze_bn=False,
    )

    print(model.summary())

    target_path = os.path.join(target_dir, "resnet18_mnist.h5")
    print(f"Saving model to {target_path}")
    save_model(model, target_path)
    print("Done.")


if __name__ == "__main__":
    main()
