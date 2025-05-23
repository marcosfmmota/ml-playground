# %%
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

# %%
model = keras.models.load_model("xception_from_scratch.keras")
model.summary(line_length=80)  # type: ignore
# %%
img_path = keras.utils.get_file(
    fname="cat.jpg", origin="https://img-datasets.s3.amazonaws.com/cat.jpg"
)


def get_img_array(img_path: str, target_size: tuple) -> np.ndarray:
    img = keras.utils.load_img(img_path, target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


img_tensor = get_img_array(img_path, (180, 180))
# %%
plt.axis("off")
plt.imshow(img_tensor[0].astype("uint8"))
plt.show()

# %%
layers_outputs = []
layer_names = []
for layer in model.layers:  # type: ignore
    if isinstance(layer, (layers.SeparableConv2D, layers.MaxPooling2D)):
        layers_outputs.append(layer.output)
        layer_names.append(layer.name)

activation_model = keras.Model(inputs=model.input, outputs=layers_outputs)  # type: ignore

# %%
activations = activation_model(img_tensor)
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 5], cmap="viridis")
# %%
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros(((size + 1) * n_cols - 1, images_per_row * (size + 1) - 1))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy()
            if channel_image.sum() != 0:
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype("uint8")
            display_grid[
                col * (size + 1) : (col + 1) * size + col,
                row * (size + 1) : (row + 1) * size + row,
            ] = channel_image
    scale = 1.0 / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.axis("off")
    plt.imshow(display_grid, aspect="auto", cmap="viridis")

# %%
