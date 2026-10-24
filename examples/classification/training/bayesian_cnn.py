"""A Bayesian convolutional network with Monte Carlo prediction.

Dataset: Fashion-MNIST (clothing images, 10 classes) for training and testing; MNIST (digits) as
out-of-distribution inputs.
Technique (training): Bayesian `Conv2DFlipout` and `DenseFlipout` layers learn a distribution over
their weights with variational inference. Each prediction samples new weights; averaging several
gives the predictive distribution, and their disagreement measures what the model does not know.

The script reports the accuracy and calibration of single and averaged predictions, and how well
the entropy of the average and the disagreement between samples (mutual information) detect
digits.

Run:
    KERAS_BACKEND=jax python examples/classification/training/bayesian_cnn.py
"""

import os

import numpy as np

import keras
from keras_proba import layers, metrics

QUICK = bool(os.environ.get("KERAS_PROBA_QUICK"))  # Small run for smoke tests.
EPOCHS = 1 if QUICK else 5
NUM_SAMPLES = 3 if QUICK else 20  # Monte Carlo predictions.
keras.utils.set_random_seed(0)

# ---------------------------------------------------------------------------------------------
# Data: 28x28x1 images in [0, 1].
# ---------------------------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
(_, _), (x_ood, _) = keras.datasets.mnist.load_data()
if QUICK:
    x_train, y_train, x_test, y_test, x_ood = x_train[:5000], y_train[:5000], x_test[:1000], y_test[:1000], x_ood[:1000]


def preprocess(images):
    return (images[..., None] / 255.0).astype("float32")


x_train, x_test, x_ood = preprocess(x_train), preprocess(x_test), preprocess(x_ood)

# ---------------------------------------------------------------------------------------------
# Model. The KL term is added on every batch, so weight it by 1 / num_train_examples to count it
# once per epoch.
# ---------------------------------------------------------------------------------------------
kl_weight = 1.0 / len(x_train)
model = keras.Sequential(
    [
        keras.Input((28, 28, 1)),
        layers.Conv2DFlipout(16, 3, padding="same", activation="relu", kl_weight=kl_weight),
        keras.layers.MaxPooling2D(),
        layers.Conv2DFlipout(32, 3, padding="same", activation="relu", kl_weight=kl_weight),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        layers.DenseFlipout(128, activation="relu", kl_weight=kl_weight),
        layers.DenseFlipout(10, kl_weight=kl_weight),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=128, validation_split=0.1, verbose=2)


# ---------------------------------------------------------------------------------------------
# Monte Carlo prediction: every `predict` call samples new weights.
# ---------------------------------------------------------------------------------------------
def predict_samples(x):
    logits = [model.predict(x, batch_size=512, verbose=0) for _ in range(NUM_SAMPLES)]
    return np.stack([keras.ops.convert_to_numpy(keras.ops.softmax(l)) for l in logits])  # (samples, n, classes)


def entropy(p):
    return -np.sum(p * np.log(p + 1e-12), axis=-1)


def auroc(in_scores, out_scores):
    """Probability that an out-of-distribution input scores higher than an in-distribution one."""
    ranks = np.concatenate([in_scores, out_scores]).argsort().argsort() + 1
    n_in, n_out = len(in_scores), len(out_scores)
    return (ranks[n_in:].sum() - n_out * (n_out + 1) / 2) / (n_in * n_out)


samples_in, samples_out = predict_samples(x_test), predict_samples(x_ood)
probs_in, probs_out = samples_in.mean(0), samples_out.mean(0)

# Averaging weight samples gives a more accurate prediction than a single sample.
ece = metrics.ExpectedCalibrationError(num_bins=15)
for name, probs in [("single sample", samples_in[0]), (f"average of {NUM_SAMPLES}", probs_in)]:
    ece.reset_state()
    ece.update_state(y_test, probs)
    accuracy = (probs.argmax(-1) == y_test).mean()
    print(f"{name:16s} accuracy {accuracy:.1%}, ECE {float(keras.ops.convert_to_numpy(ece.result())):.3f}")

# Total uncertainty (entropy of the average) = aleatoric (average entropy) + epistemic (mutual
# information: how much the samples disagree).
mutual_info_in = entropy(probs_in) - entropy(samples_in).mean(0)
mutual_info_out = entropy(probs_out) - entropy(samples_out).mean(0)
print("\nDetecting digits (AUROC, 0.5 = chance):")
print(f"  predictive entropy  {auroc(entropy(probs_in), entropy(probs_out)):.3f}")
print(f"  mutual information  {auroc(mutual_info_in, mutual_info_out):.3f}")
print(f"\nMean mutual information: clothing {mutual_info_in.mean():.3f}, digits {mutual_info_out.mean():.3f}")
