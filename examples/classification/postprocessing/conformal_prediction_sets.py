"""Conformal prediction sets for image classification.

Dataset: Fashion-MNIST (clothing images, 10 classes).
Technique (post-processing): `keras_proba.postprocessing.SplitConformalClassifier` turns the
predicted probabilities of any classifier into sets of classes that contain the true class with
probability at least 90%. Easy images get a single class; ambiguous ones (shirt or T-shirt?) get
several.

- LAC sets contain the classes whose probability is above a threshold: the smallest sets on
  average.
- APS sets contain the most probable classes until their total probability reaches a threshold:
  their size adapts to each image.

Run:
    KERAS_BACKEND=jax python examples/classification/postprocessing/conformal_prediction_sets.py
"""

import os

import numpy as np

import keras
from keras_proba import postprocessing

QUICK = bool(os.environ.get("KERAS_PROBA_QUICK"))  # Small run for smoke tests.
EPOCHS = 2 if QUICK else 10
ALPHA = 0.1  # Target miscoverage: sets contain the true class 90% of the time.
CLASSES = ["T-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]
keras.utils.set_random_seed(0)

# ---------------------------------------------------------------------------------------------
# Data: training images, 10,000 held-out calibration images, and the test set.
# ---------------------------------------------------------------------------------------------
(x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x, x_test = (x[..., None] / 255.0).astype("float32"), (x_test[..., None] / 255.0).astype("float32")
x_train, y_train, x_cal, y_cal = x[10_000:], y[10_000:], x[:10_000], y[:10_000]
if QUICK:
    x_train, y_train = x_train[:5000], y_train[:5000]

# ---------------------------------------------------------------------------------------------
# Any trained classifier works: conformal prediction only needs its probabilities.
# ---------------------------------------------------------------------------------------------
model = keras.Sequential(
    [
        keras.Input((28, 28, 1)),
        keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, verbose=2)

probs_cal = model.predict(x_cal, batch_size=1024, verbose=0)
probs_test = model.predict(x_test, batch_size=1024, verbose=0)
print(f"\nTest accuracy of the top class: {(probs_test.argmax(-1) == y_test).mean():.1%}")

# ---------------------------------------------------------------------------------------------
# Calibrate the sets on held-out images, and check them on the test set.
# ---------------------------------------------------------------------------------------------
print(f"\n{1 - ALPHA:.0%} prediction sets on the test set:")
results = {}
for score in ["lac", "aps"]:
    conformal = postprocessing.SplitConformalClassifier(alpha=ALPHA, score=score, seed=0).fit(y_cal, probs_cal)
    sets = keras.ops.convert_to_numpy(conformal.predict_set(probs_test))
    results[score] = sets
    sizes = sets.sum(-1)
    coverage = float(keras.ops.convert_to_numpy(postprocessing.set_coverage(y_test, sets)))
    histogram = ", ".join(f"{k}: {np.mean(sizes == k):.0%}" for k in range(0, 4)) + f", 4+: {np.mean(sizes >= 4):.0%}"
    print(f"  {score.upper()}: coverage {coverage:.1%}, mean size {sizes.mean():.2f}  (sizes {histogram})")

# The guarantee is over all images: classes the model confuses (e.g. shirts) can be covered less.
print("\nCoverage and mean set size per class (LAC | APS):")
for k, name in enumerate(CLASSES):
    mask = y_test == k
    cells = [f"{sets[mask, k].mean():.0%} / {sets[mask].sum(-1).mean():.2f}" for sets in results.values()]
    print(f"  {name:9s} {cells[0]:>14s} | {cells[1]:>14s}")

# The guarantee is on average over the choice of calibration images, and assumes calibration and
# test images are exchangeable. One calibration set can land a little off target, and the official
# test images are slightly harder than the training images held out for calibration: LAC, whose
# threshold sits where scores are sparse, is the most affected. Re-splitting the pooled
# calibration and test predictions at random makes them exchangeable, and the coverage averages
# to the target.
pooled_probs, pooled_labels = np.concatenate([probs_cal, probs_test]), np.concatenate([y_cal, y_test])
rng = np.random.default_rng(0)
print("\nMean coverage over 20 random calibration / test splits of the pooled predictions:")
for score in ["lac", "aps"]:
    coverages = []
    for _ in range(20):
        order = rng.permutation(len(pooled_labels))
        cal, test = order[: len(y_cal)], order[len(y_cal) :]
        conformal = postprocessing.SplitConformalClassifier(alpha=ALPHA, score=score, seed=0).fit(pooled_labels[cal], pooled_probs[cal])
        sets = conformal.predict_set(pooled_probs[test])
        coverages.append(float(keras.ops.convert_to_numpy(postprocessing.set_coverage(pooled_labels[test], sets))))
    print(f"  {score.upper()}: {np.mean(coverages):.1%} (min {np.min(coverages):.1%}, max {np.max(coverages):.1%})")

# Set sizes reflect difficulty: larger sets where the top class is wrong more often.
correct = probs_test.argmax(-1) == y_test
for score, sets in results.items():
    sizes = sets.sum(-1)
    print(f"\n{score.upper()} mean set size when the top class is right: {sizes[correct].mean():.2f}, when it is wrong: {sizes[~correct].mean():.2f}")
