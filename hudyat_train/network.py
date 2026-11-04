"""Exact Keras counterpart of HUDYAT's category-core.js buildModel."""
from __future__ import annotations

import os


def tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass  # already initialized by another operation in this process
    return tf


def build_model(classes: int, features: int = 128):
    tf = tensorflow()
    if classes < 2 or features not in (128, 141):
        raise ValueError("The model needs at least 2 classes and 128 or 141 features.")
    l = tf.keras.layers
    return tf.keras.Sequential([
        l.Input(shape=(32, features)),
        l.Conv1D(64, 3, padding="same", activation="relu"),
        l.BatchNormalization(momentum=0.99, epsilon=0.0001),
        l.MaxPooling1D(pool_size=2), l.Dropout(0.3),
        l.Conv1D(128, 3, padding="same", activation="relu"),
        l.BatchNormalization(momentum=0.99, epsilon=0.0001), l.Dropout(0.3),
        l.Bidirectional(l.LSTM(128, return_sequences=True, activation="tanh", recurrent_activation="sigmoid",
                               dropout=0, recurrent_dropout=0)), l.Dropout(0.3),
        l.Bidirectional(l.LSTM(64, activation="tanh", recurrent_activation="sigmoid",
                              dropout=0, recurrent_dropout=0)), l.Dropout(0.3),
        l.Dense(64, activation="relu"), l.Dense(classes, activation="softmax"),
    ])


def architecture(config):
    if isinstance(config, dict):
        return {k: architecture(v) for k, v in config.items() if k != "name"}
    if isinstance(config, (tuple, list)):
        return [architecture(v) for v in config]
    return config
