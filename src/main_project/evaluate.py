import torch
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

from main_project.train import train_classifier
from main_project.model import targetClassifier
from main_project.visualize import plot_transport_images
from main_project.utils import load
from main_project.environment import MODELS_DIM, INTERMEDIATE_FRACTIONS, MAX_POINTS


SEED = 5678
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)


# https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
def MMD(x: Array, y: Array, kernel):
    xx, yy, zz = jnp.matmul(x, x.T), jnp.matmul(y, y.T), jnp.matmul(x, y.T)

    rx = jnp.diag(xx)[jnp.newaxis, :]  # (1, N)
    ry = jnp.diag(yy)[jnp.newaxis, :]  # (1, M)

    # This is the expanded exponential kernel - XX corresponding to element x_i times x_i
    dxx = rx.T + rx - 2.0 * xx
    dyy = ry.T + ry - 2.0 * yy
    dxy = rx.T + ry - 2.0 * zz

    XX, YY, XY = (jnp.zeros_like(xx), jnp.zeros_like(xx), jnp.zeros_like(xx))

    # Turns distances into similarity scores betweem the distributions
    if kernel == "multiscale":
        # The standard devisation is unkown, so having multiple different bandwidth makes the test sentitive to multiple cases
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx) ** -1
            YY += a**2 * (a**2 + dyy) ** -1
            XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += jnp.exp(-0.5 * dxx / a)
            YY += jnp.exp(-0.5 * dyy / a)
            XY += jnp.exp(-0.5 * dxy / a)

    # MMD² = E[k(x,x')] + E[k(y,y')] − 2·E[k(x,y)]
    return jnp.mean(XX + YY - 2.0 * XY)


def classifier_confidence(transported_images, target_class=1):
    classifier = load(name="evaluate_classifier", path="models", model=targetClassifier(subkey))
    x = jnp.asarray(transported_images)

    log_probs = jax.vmap(classifier)(x)
    probs = jnp.exp(log_probs)  # Shape: [n, 10]

    p_target = probs[:, target_class]
    predictions = jnp.argmax(probs, axis=-1)
    classified = jnp.mean(predictions == target_class)

    # print(f"Mean P(class={target_class}):  {float(jnp.mean(p_target)):.20f}")
    # print(f"Fraction class {target_class}:  {float(classified):.20f}")

    return p_target


def evaluate_latent_space_knn(latent_array, labels):
    classifier = KNeighborsClassifier(n_neighbors=6)  # 5 by default
    knn_acc = cross_val_score(classifier, latent_array, labels, cv=5).mean()
    return knn_acc


columns = "Fraction of transport", "MDD", "Confidence of Classifier"


def evaluate_by_model(model):
    source_img, target_img, expected_target_img, intermediate_images = (
        np.load(f"data/{model}/original_images.npy"),
        np.load(f"data/{model}/target_images.npy"),
        np.load(f"data/{model}/expected_target_images.npy"),
        np.load(f"data/{model}/intermediate_images.npy"),
    )

    intermediate_images = intermediate_images.transpose(1, 0, 2)  # Corrects order for easier plotting

    data = []
    for frac, imgs in zip(INTERMEDIATE_FRACTIONS, intermediate_images):
        mmd = MMD(jnp.asarray(imgs), jnp.asarray(target_img), kernel="rbf")
        classifier_conf = classifier_confidence(imgs, 1)

        data.append([frac, mmd, classifier_conf[1]])

        # plot_transport_images(imgs, target_img, n=5, title=f"MMD score of {mmd.item()} and classification confidence of {classifier_conf[1]}")

    df = pd.DataFrame(data, columns=columns)

    print(df)
    print("\n\n\n")
    df.to_csv(f"data/{model}/evalution.csv")
    # print(df.to_latex())


def evaluate_sb_by_model(model):
    decoded = np.load(f"data/sb_{model}/decoded.npy")  # (n_samples, n_steps, 784)
    target_img = np.load(f"data/{model}/target_images.npy")  # reuse sinkhorn's target

    # decoded is already (n_samples, n_steps, 784) — transpose to (n_steps, n_samples, 784)
    # to match the same iteration pattern as intermediate_images
    decoded = decoded.transpose(1, 0, 2)  # (n_steps, n_samples, 784)

    n_steps = decoded.shape[0]
    t_values = np.linspace(0, 1, n_steps)

    data = []
    for t, imgs in zip(t_values, decoded):
        mmd = MMD(jnp.asarray(imgs), jnp.asarray(target_img), kernel="rbf")
        classifier_conf = classifier_confidence(imgs, 1)
        data.append([t, mmd, classifier_conf[1]])

    df = pd.DataFrame(data, columns=["t (time step)", "MMD", "Confidence of Classifier"])

    print(df)
    print("\n\n\n")
    df.to_csv(f"data/sb_{model}/evaluation.csv")


def run_evaluation():
    for dim in MODELS_DIM:
        model_name = f"ae_model_dim_{dim}"

        print("Evaluating Sinkhorn model", model_name, "\n")
        evaluate_by_model(model=model_name)

        print("Evaluating Schrödinger Bridge model", model_name, "\n")
        evaluate_sb_by_model(model=model_name)


if __name__ == "__main__":
    # Plotting
    run_evaluation()
