"""Calibrating an overconfident image classifier with temperature scaling.

Dataset: Fashion-MNIST (clothing images, 10 classes).
Technique (post-processing): `keras_proba.postprocessing.TemperatureScaling` divides the logits of
a trained network by a single temperature, fitted on held-out data to minimize the negative
log-likelihood. It never changes the predicted class, only how confident the predictions are.

A network trained to convergence is often overconfident: it predicts with 95% confidence but is
right less often. The script measures this with the expected calibration error (ECE) and a
reliability table, before and after scaling.

Run:
    KERAS_BACKEND=jax python examples/classification/postprocessing/temperature_scaling.py
"""

import os

import numpy as np

import keras
from keras_proba import metrics, postprocessing

QUICK = bool(os.environ.get("KERAS_PROBA_QUICK"))  # Small run for smoke tests.
EPOCHS = 2 if QUICK else 15
keras.utils.set_random_seed(0)

# ---------------------------------------------------------------------------------------------
# Data: training images, 10,000 held-out images to fit the temperature, and the test set.
# ---------------------------------------------------------------------------------------------
(x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x, x_test = (x[..., None] / 255.0).astype("float32"), (x_test[..., None] / 255.0).astype("float32")
x_train, y_train, x_val, y_val = x[10_000:], y[10_000:], x[:10_000], y[:10_000]
if QUICK:
    x_train, y_train = x_train[:5000], y_train[:5000]

# ---------------------------------------------------------------------------------------------
# A small CNN trained long enough to become overconfident.
# ---------------------------------------------------------------------------------------------
model = keras.Sequential(
    [
        keras.Input((28, 28, 1)),
        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10),  # Logits.
    ]
)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=loss, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, verbose=2)

# ---------------------------------------------------------------------------------------------
# Fit the temperature on held-out data, and append it to the model.
# ---------------------------------------------------------------------------------------------
scaling = postprocessing.TemperatureScaling()
temperature = scaling.fit(model.predict(x_val, verbose=0), y_val)
calibrated = keras.Sequential([model, scaling])
print(f"\nFitted temperature: {temperature:.2f} (> 1: the network was overconfident)")


def evaluate(network):
    logits = network.predict(x_test, batch_size=1024, verbose=0)
    ece = metrics.ExpectedCalibrationError(num_bins=15, from_logits=True)
    brier = metrics.BrierScore(from_logits=True)
    ece.update_state(y_test, logits)
    brier.update_state(y_test, logits)
    probs = keras.ops.convert_to_numpy(keras.ops.softmax(logits))
    return {
        "accuracy": (probs.argmax(-1) == y_test).mean(),
        "NLL": float(keras.ops.convert_to_numpy(loss(y_test, logits))),
        "ECE": float(keras.ops.convert_to_numpy(ece.result())),
        "Brier": float(keras.ops.convert_to_numpy(brier.result())),
    }, probs


before, probs_before = evaluate(model)
after, probs_after = evaluate(calibrated)
print(f"\n{'':10s} {'accuracy':>9s} {'NLL':>7s} {'ECE':>7s} {'Brier':>7s}")
for name, result in [("before", before), ("after", after)]:
    print(f"{name:10s} {result['accuracy']:9.1%} {result['NLL']:7.3f} {result['ECE']:7.3f} {result['Brier']:7.3f}")

# Reliability table: within each confidence range, how often are the predictions right?
print("\nReliability on the test set: accuracy of the predictions in each confidence range")
print(f"  {'confidence':>12s} {'before: n, accuracy':>22s} {'after: n, accuracy':>22s}")
edges = [0.0, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0]
for low, high in zip(edges[:-1], edges[1:]):
    cells = []
    for probs in (probs_before, probs_after):
        confidence = probs.max(-1)
        mask = (confidence > low) & (confidence <= high)
        accuracy = (probs.argmax(-1)[mask] == y_test[mask]).mean() if mask.any() else float("nan")
        cells.append(f"{mask.sum():6d}, {accuracy:6.1%}")
    print(f"  {f'{low:.2f}-{high:.2f}':>12s} {cells[0]:>22s} {cells[1]:>22s}")
