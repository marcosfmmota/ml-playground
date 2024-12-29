# %%
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import layers


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
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

# %%
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.2)
# %%
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "r--", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("[IMDB] Training and validation loss")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "r--", label="Training acc")
plt.plot(epochs, val_acc, "b", label="Validation acc")
plt.title("[IMDB] Training and validation accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("imdb_accuracy_plot.png", dpi=300)
plt.show()

# %%
# Since the model overfit after 4 epochs we retrain with only 4 epochs and evaluate on test data.
model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
print(f"Test set loss {results[0]} - accuracy {results[1]}")
