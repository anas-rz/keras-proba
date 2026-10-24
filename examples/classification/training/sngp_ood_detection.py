"""Out-of-distribution detection with SNGP.

Dataset: Fashion-MNIST (clothing images, 10 classes) for training and testing; MNIST (digits) as
out-of-distribution inputs the model has never seen.
Technique (training): the Spectral-normalized Neural Gaussian Process (SNGP). Spectral
normalization keeps distances between inputs in the hidden features, and a Gaussian process
output layer makes predictions less confident the further an input is from the training data.

A standard network is confidently wrong on digits; SNGP, with the same backbone and accuracy,
assigns them much lower confidence. The script compares how well the confidence separates
clothing from digits (AUROC, 0.5 = chance, 1.0 = perfect).

Run:
    KERAS_BACKEND=jax python examples/classification/training/sngp_ood_detection.py
"""

import os

import numpy as np

import keras
from keras_proba import callbacks, layers

QUICK = bool(os.environ.get("KERAS_PROBA_QUICK"))  # Small run for smoke tests.
EPOCHS = 1 if QUICK else 10
WIDTH, BLOCKS = 256, 3
keras.utils.set_random_seed(0)

# ---------------------------------------------------------------------------------------------
# Data: flattened 28x28 images in [0, 1].
# ---------------------------------------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
(_, _), (x_ood, _) = keras.datasets.mnist.load_data()
if QUICK:
    x_train, y_train = x_train[:5000], y_train[:5000]


def preprocess(images):
    return (images.reshape(len(images), -1) / 255.0).astype("float32")


x_train, x_test, x_ood = preprocess(x_train), preprocess(x_test), preprocess(x_ood)


# ---------------------------------------------------------------------------------------------
# Models: the same residual MLP backbone, with and without the SNGP changes.
# ---------------------------------------------------------------------------------------------
def build(sngp):
    inputs = keras.Input((784,))
    hidden = keras.layers.Dense(WIDTH)(inputs)
    for _ in range(BLOCKS):
        block = keras.layers.Dense(WIDTH, activation="relu")
        if sngp:
            # Bounding the spectral norm of each residual block keeps the backbone
            # distance-preserving. Reparameterization mode behaves the same on every backend.
            block = layers.SpectralNormalization(block, norm_multiplier=0.95, mode="reparameterization")
        hidden = hidden + keras.layers.Dropout(0.1)(block(hidden))
    if sngp:
        logits = layers.GaussianProcessClassificationHead(
            inner_dim=0,  # The backbone already pools: no extra hidden layer.
            num_classes=10,
            scale_random_features=True,
            temperature=np.pi / 8,  # The mean-field factor used at inference.
            gp_cov_momentum=-1,  # Exact covariance over each epoch, reset by the callback.
        )(hidden)
    else:
        logits = keras.layers.Dense(10)(hidden)
    model = keras.Model(inputs, logits)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


def auroc(in_scores, out_scores):
    """Probability that an out-of-distribution input scores higher than an in-distribution one."""
    ranks = np.concatenate([in_scores, out_scores]).argsort().argsort() + 1
    n_in, n_out = len(in_scores), len(out_scores)
    return (ranks[n_in:].sum() - n_out * (n_out + 1) / 2) / (n_in * n_out)


print(f"{'':10s} {'accuracy':>9s} {'confidence (clothing / digits)':>32s} {'OOD AUROC':>10s}")
for name, sngp in [("standard", False), ("SNGP", True)]:
    keras.utils.set_random_seed(0)
    model = build(sngp)
    # The GP covariance is accumulated over the training features; reset it each epoch so it
    # reflects the final features.
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=128,
        verbose=0,
        callbacks=[callbacks.ResetCovarianceCallback()] if sngp else [],
    )
    # At inference, the SNGP head returns uncertainty-adjusted (mean-field) logits.
    probs_in = keras.ops.convert_to_numpy(keras.ops.softmax(model.predict(x_test, batch_size=512, verbose=0)))
    probs_out = keras.ops.convert_to_numpy(keras.ops.softmax(model.predict(x_ood, batch_size=512, verbose=0)))
    accuracy = (probs_in.argmax(-1) == y_test).mean()
    confidence_in, confidence_out = probs_in.max(-1), probs_out.max(-1)
    score = auroc(1 - confidence_in, 1 - confidence_out)  # Low confidence = likely out of distribution.
    print(f"{name:10s} {accuracy:9.1%} {confidence_in.mean():19.2f} / {confidence_out.mean():.2f} {score:15.3f}")
