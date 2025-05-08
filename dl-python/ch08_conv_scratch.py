# %%
import os

os.environ["KERAS_BACKEND"] = "jax"

import random
import shutil
from pathlib import Path

import kagglehub
import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.utils import image_dataset_from_directory
import tensorflow as tf


# %%
def make_small_dataset(n_split: list[int], download_path: Path, target_path: Path = Path(".")):
    for category in ("Cat", "Dog"):
        category_path = download_path / "PetImages" / category
        samples = set(category_path.glob("*.jpg"))

        for split, n in zip(["train", "validation", "test"], n_split):
            split_samples = random.sample(list(samples), n)

            (target_path / split / category).mkdir(parents=True)
            for file in split_samples:
                shutil.copy(src=category_path / file, dst=target_path / split / category)

            samples -= set(split_samples)


# %%
# Run this cell if need to create dataset
download_path = Path(kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset"))
make_small_dataset([1000, 500, 1000], download_path, Path("ch08-cats-vs-dogs"))


# %%
def get_model() -> keras.Model:
    inputs = keras.Input(shape=(180, 180, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    return model


model = get_model()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# %%
# Load dataset keras function
batch_size = 32
image_size = (180, 180)
train_dataset = image_dataset_from_directory(
    "ch08-cats-vs-dogs/train", image_size=image_size, batch_size=batch_size
)
validation_dataset = image_dataset_from_directory(
    "ch08-cats-vs-dogs/validation", image_size=image_size, batch_size=batch_size
)
test_dataset = image_dataset_from_directory(
    "ch08-cats-vs-dogs/test", image_size=image_size, batch_size=batch_size
)

# %%
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch.keras", save_best_only=True, monitor="val_loss"
    )
]
history = model.fit(
    train_dataset, epochs=50, validation_data=validation_dataset, callbacks=callbacks
)

# %%
accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

plt.plot(epochs, accuracy, "r--", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()

plt.plot(epochs, loss, "r--", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# %%
test_model = keras.models.load_model("convnet_from_scratch.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")
# %%
# Add data augmentation for training data
data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
]


def data_augmentation(images, targets):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images, targets


augmented_train_dataset = train_dataset.map(data_augmentation, num_parallel_calls=8)
augmented_train_dataset = augmented_train_dataset.prefetch(tf.data.AUTOTUNE)


# %%
# New model to deal with overfitting using augmentation and dropout
def get_dropout_model() -> keras.Model:
    inputs = keras.Input(shape=(180, 180, 3))
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(filters=512, kernel_size=3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


new_model = get_dropout_model()
new_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# %%
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss",
    )
]
history = new_model.fit(
    augmented_train_dataset, epochs=100, validation_data=validation_dataset, callbacks=callbacks
)
# %%
