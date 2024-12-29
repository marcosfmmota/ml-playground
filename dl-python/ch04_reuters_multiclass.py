# %%
import os

os.environ["KERAS_BACKEND"] = "jax"
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras.utils import to_categorical
from keras import layers

# %%
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


# %%
# Word index decoding. There is a offset of 3 because 0, 1, 2 are reversed indices for
# "padding", "start of sequence", "unknown".
def decode_sample(imdb_sample: list) -> str:
    word_index = reuters.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    decoded_sample = " ".join([reverse_word_index.get(i - 3, "?") for i in imdb_sample])
    return decoded_sample


def multi_hot_encoding(sequences, num_classes):
    results = np.zeros((len(sequences), num_classes))
    for i, sequence in enumerate(sequences):
        results[i][sequence] = 1
    return results


# %%
print(decode_sample(train_data[0]))
# %%
x_train = multi_hot_encoding(train_data, num_classes=10000)
x_test = multi_hot_encoding(test_data, num_classes=10000)
# Keras way to do label one-hot encoding
# Use "sparse_categorical_crossentropy" as model loss in case of integer labels
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)
# %%
model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(46, activation="softmax"),
    ]
)

top_3_accuracy = keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", top_3_accuracy]
)

# %%
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]
# %%
history = model.fit(
    partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val)
)

# %%
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "r--", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("[Reuters] Training and validation loss")
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
plt.title("[Reuters] Training and validation accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# %%
plt.clf()
acc = history.history["top_3_accuracy"]
val_acc = history.history["val_top_3_accuracy"]
plt.plot(epochs, acc, "r--", label="Training top-3 accuracy")
plt.plot(epochs, val_acc, "b", label="Validation top-3 accuracy")
plt.title("[Reuters] Training and validation top-3 accuracy")
plt.xlabel("Epochs")
plt.xticks(epochs)
plt.ylabel("Top-3 accuracy")
plt.legend()
plt.show()

# %%
# Fit to the 9th epoch (where model start to overfit) and evaluate on test set.
model.fit(x_train, y_train, epochs=9, batch_size=512)
results = model.evaluate(x_test, y_test)
print(f"Model test loss: {results[0]} - Test Accuracy: {results[1]}")
# %%
print("Bottleneck example with hidden layer with 4 neurons.")

model = keras.Sequential(
    [
        layers.Dense(64, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(46, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(
    partial_x_train, partial_y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val)
)
