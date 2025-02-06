# %%
import os

os.environ["KERAS_BACKEND"] = "jax"

import numpy as np
import keras
from keras import layers
from keras.datasets import imdb
import matplotlib.pyplot as plt


# %%
def vectorize_sequences(sequences, dimension=10_000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results


# %%
(train_data, train_labels), _ = imdb.load_data(num_words=10_000)
train_data = vectorize_sequences(train_data)
# %%
# Original model that overfit
model = keras.Sequential(
    [
        layers.Dense(16, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history_original_model = model.fit(
    train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4
)

# %%
# Smaller model
model = keras.Sequential(
    [
        layers.Dense(4, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
history_smaller_model = model.fit(
    train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4
)


# %%
def plot_comparision_history(history_model1, history_model2, plot_config):
    val_loss1 = history_model1.history["val_loss"]
    val_loss2 = history_model2.history["val_loss"]
    epochs = range(1, 21)
    plt.plot(epochs, val_loss1, "r--", label=plot_config["label1"])
    plt.plot(epochs, val_loss2, "b-", label=plot_config["label2"])
    plt.title(plot_config["title"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(epochs)
    plt.legend()
    plt.show()


# %%
plot_config = {
    "title": "Original model vs. smaller model (IMDB review classification)",
    "label1": "Validation loss of original model",
    "label2": "Validation loss of smaller model",
}
plot_comparision_history(history_original_model, history_smaller_model, plot_config)
