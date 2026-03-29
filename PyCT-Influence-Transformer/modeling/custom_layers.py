from __future__ import annotations

import keras


class AddPositionEmbedding(keras.layers.Layer):
    """Inference-compatible positional embedding layer."""

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError("AddPositionEmbedding expects rank-3 inputs [B, L, D].")
        seq_len = int(input_shape[1])
        dim = int(input_shape[2])
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, seq_len, dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.pos_embedding


class AddClsToken(keras.layers.Layer):
    """Prepend a learnable CLS token to a token sequence."""

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError("AddClsToken expects rank-3 inputs [B, L, D].")
        dim = int(input_shape[2])
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, dim),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        if hasattr(keras, "ops"):
            batch = keras.ops.shape(inputs)[0]
            cls = keras.ops.broadcast_to(
                self.cls_token,
                (batch, 1, int(self.cls_token.shape[-1])),
            )
            return keras.ops.concatenate([cls, inputs], axis=1)
        import tensorflow as tf

        batch = tf.shape(inputs)[0]
        cls = tf.repeat(self.cls_token, repeats=batch, axis=0)
        return tf.concat([cls, inputs], axis=1)


class ExtractClsToken(keras.layers.Layer):
    """Return the CLS token (index 0) from a token sequence."""

    def call(self, inputs):
        return inputs[:, 0, :]


class DropPath(keras.layers.Layer):
    """DropPath as identity during inference."""

    def __init__(self, drop_prob=0.0, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = float(drop_prob)

    def call(self, inputs, training=None):
        del training
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"drop_prob": self.drop_prob})
        return config


class SequencePooling(keras.layers.Layer):
    """Sequence pooling with learnable token attention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score = keras.layers.Dense(1, name="dense_16")

    def call(self, inputs):
        scores = self.score(inputs)
        if hasattr(keras, "ops"):
            weights = keras.ops.softmax(scores, axis=1)
            return keras.ops.sum(weights * inputs, axis=1)
        import tensorflow as tf

        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(weights * inputs, axis=1)


def get_transformer_custom_objects():
    return {
        "AddClsToken": AddClsToken,
        "AddPositionEmbedding": AddPositionEmbedding,
        "DropPath": DropPath,
        "ExtractClsToken": ExtractClsToken,
        "SequencePooling": SequencePooling,
        "Custom>AddClsToken": AddClsToken,
        "Custom>AddPositionEmbedding": AddPositionEmbedding,
        "Custom>DropPath": DropPath,
        "Custom>ExtractClsToken": ExtractClsToken,
        "Custom>SequencePooling": SequencePooling,
    }


__all__ = [
    "AddClsToken",
    "AddPositionEmbedding",
    "DropPath",
    "ExtractClsToken",
    "SequencePooling",
    "get_transformer_custom_objects",
]
