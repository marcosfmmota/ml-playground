# %%
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax
from keras import layers
from keras.datasets import mnist


# %%
def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(64, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model


# %%

model = get_mnist_model()
loss_fn = keras.losses.SparseCategoricalCrossentropy()


def compute_loss_and_updates(trainable_variables, non_trainable_variables, inputs, targets):
    # Keras has a stateless model call to facilitate working with jax and its stateless approach
    outputs, non_trainable_variables = model.stateless_call(
        trainable_variables, non_trainable_variables, inputs
    )
    loss = loss_fn(targets, outputs)
    return loss, non_trainable_variables


grad_fn = jax.value_and_grad(compute_loss_and_updates, has_aux=True)
optimizer = keras.optimizers.Adam()
optimizer.build(model.trainable_variables)


def train_step(state, inputs, targets):
    (trainable_variables, non_trainable_variables, optimizer_variables) = state
    (loss, non_trainable_variables), grads = grad_fn(
        trainable_variables, non_trainable_variables, inputs, targets
    )
    trainable_variables, optimizer_variables = optimizer.stateless_apply(
        optimizer_variables, grads, trainable_variables
    )
    return loss, (
        trainable_variables,
        non_trainable_variables,
        optimizer_variables,
    )


# %%
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255
# run on a batch
batch_size = 32
inputs = train_images[:batch_size]
targets = train_labels[:batch_size]

# %%
trainable_variables = [v.value for v in model.trainable_variables]
non_trainable_variables = [v.value for v in model.non_trainable_variables]
optimizer_variables = [v.value for v in optimizer.variables]

state = (trainable_variables, non_trainable_variables, optimizer_variables)
loss, state = train_step(state, inputs, targets)

print(loss)
# %%
