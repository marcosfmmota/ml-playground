# %%
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
from keras.datasets import imdb

# %%
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# %%
# Word index decoding. There is a offset of 3 because 0, 1, 2 are reversed indices for
# "padding", "start of sequence", "unknown".
def decode_sample(imdb_sample: list) -> str:
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_sample = " ".join([reverse_word_index.get(i - 3, "?") for i in imdb_sample])
    return decoded_sample


# %%
def multi_hot_encoding(sequences, num_classes):
    results = np.zeros((len(sequences), num_classes))
    for i, sequence in enumerate(sequences):
        results[i][sequence] = 1
    return results


# %%
# encoding and float casting
x_train = multi_hot_encoding(train_data, num_classes=10000)
x_test = multi_hot_encoding(test_data, num_classes=10000)
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")
# %%
