# %%
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import jax
from keras import layers
from keras.datasets import mnist


# %%
class CustomModel(keras.Model):
    def compute_loss_and_updates(
        self, trainable_variables, non_trainable_variables, inputs, targets, training=False
    ):
        predictions, non_trainable_variables = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            inputs,
            training=training,
        )
        loss = self.compute_loss(y=targets, y_pred=predictions)
        return loss, (predictions, non_trainable_variables)

    def train_step(self, state, data):
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        inputs, targets = data
        grad_fn = jax.value_and_grad(self.compute_loss_and_updates, has_aux=True)

        (loss, (predictions, non_trainable_variables)), grads = grad_fn(
            trainable_variables,
            non_trainable_variables,
            inputs,
            targets,
            training=True,
        )
        (trainable_variables, optimizer_variables) = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        new_metrics_vars = []
        logs = {}
        for metric in self.metrics:
            num_new_metrics = len(new_metrics_vars)
            this_metric_vars = metrics_variables[
                num_new_metrics : num_new_metrics + len(metric.variables)
            ]
            if metric.name == "loss":
                this_metric_vars = metric.stateless_update_state(this_metric_vars, loss)
            else:
                this_metric_vars = metric.stateless_update_state(
                    this_metric_vars, targets, predictions
                )
            logs[metric.name] = metric.stateless_result(this_metric_vars)
            new_metrics_vars += this_metric_vars

        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            new_metrics_vars,
        )
        return logs, state


# %%
def get_custom_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = CustomModel(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy],
    )
    return model


# %%

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255


model = get_custom_model()
model.fit(train_images, train_labels, epochs=3)

# %%
