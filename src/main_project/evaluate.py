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
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

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
    classifier = KNeighborsClassifier(n_neighbors=3)  # 5 by default
    knn_acc = cross_val_score(classifier, latent_array, labels, cv=5).mean()
    return knn_acc




#Plotting
columns = "Fraction of transport","MDD", "Confidence of Classifier"
data = []

if __name__ == "__main__":

    source_img, target_img, intermediate_images = np.load("data/original_images.npy"), np.load("data/expected_target_images.npy"), np.load("data/intermediate_images.npy")
    intermediate_images = intermediate_images.transpose(1, 0, 2)  # Corrects order for easier plotting

    fractions = [0.25, 0.5, 0.75, 1.0]
    for frac, imgs in zip(fractions, intermediate_images):

   
        
        mmd = MMD(jnp.asarray(imgs), jnp.asarray(target_img), kernel="multiscale")
        classifier_conf = classifier_confidence(imgs, 1)

        data.append([frac, mmd,classifier_conf[1]])

        #plot_transport_images(imgs, target_img, n=5, title=f"MMD score of {mmd.item()} and classification confidence of {classifier_conf[1]}")

    df = pd.DataFrame(data, columns=columns)

    print("From digit 0 to digit 1 evaluation scores")
    print(df)
    print("\n\n\n")
    print(df.to_latex())
