import torch
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd



# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm


from main_project.train import train_classifier
from main_project.model import targetClassifier
from main_project.visualize import plot_transport_images
from main_project.optimal_transport import get_trajectory
from main_project.utils import load
from main_project.environment import MODELS_DIM, INTERMEDIATE_FRACTIONS, MAX_POINTS, GAMMA


SEED = 5678
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)


#calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid


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

    # # Turns distances into similarity scores betweem the distributions
    # if kernel == "multiscale":
    #     # The standard devisation is unkown, so having multiple different bandwidth makes the test sentitive to multiple cases
    #     bandwidth_range = [0.2, 0.5, 0.9, 1.3]
    #     for a in bandwidth_range:
    #         XX += a**2 * (a**2 + dxx) ** -1
    #         YY += a**2 * (a**2 + dyy) ** -1
    #         XY += a**2 * (a**2 + dxy) ** -1

    if kernel == "rbf":
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += jnp.exp(-0.5 * dxx / a**2)
            YY += jnp.exp(-0.5 * dyy / a**2)
            XY += jnp.exp(-0.5 * dxy / a**2)

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


columns = ["Fraction of transport", "MMD", "Confidence of Classifier", "FID"]


def evaluate_by_model_in_image_space(model):
    source_img, target_img, expected_target_img, intermediate_images = (
        np.load(f"data/{model}/original_images.npy"),
        np.load(f"data/{model}/target_images.npy"),
        np.load(f"data/{model}/expected_target_images.npy"),
        np.load(f"data/{model}/intermediate_images.npy"),
    )

    intermediate_images = intermediate_images.transpose(1, 0, 2)  # (n_fracs, n_points, 784)

    data = []
    for frac, imgs in zip(INTERMEDIATE_FRACTIONS, intermediate_images):
        mmd = MMD(jnp.asarray(imgs), jnp.asarray(target_img), kernel="rbf")
        classifier_conf = classifier_confidence(imgs, 1)
        fid = calculate_fid(np.asarray(imgs), np.asarray(target_img))
        data.append([frac, float(mmd), float(jnp.mean(classifier_conf)), fid])

    df = pd.DataFrame(data, columns=columns)

    print(df)
    print("\n\n\n")
    df.to_csv(f"data/{model}/evaluation.csv", index=False)
    # print(df.to_latex())

def evaluate_by_model_in_latent_space(model):
    y_original = np.load(f"data/{model}/y_original.npy")
    expected_target = np.load(f"data/{model}/expected_target.npy")
    
    mmd = MMD(jnp.asarray(y_original), jnp.asarray(expected_target), kernel="rbf")

    return mmd



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
        fid = calculate_fid(np.asarray(imgs), np.asarray(target_img))
        data.append([t, mmd, classifier_conf[1], fid])

    df = pd.DataFrame(data, columns=["t (time step)", "MMD", "Confidence of Classifier", "FID"])

    print(df)
    print("\n\n\n")
    df.to_csv(f"data/sb_{model}/evaluation.csv")


def run_evaluation():
    summary = []

    for dim in MODELS_DIM:
        model_name = f"ae_best_model_bo_{dim}"

        for gamma in GAMMA:
            folder_key = f"{model_name}_{gamma}"
            print(f"\n--- dim={dim}, gamma={gamma} ---")

            # Latent space MMD
            mmd_latent = evaluate_by_model_in_latent_space(model=folder_key)
            print(f"  MMD latent: {float(mmd_latent):.6f}")

            # Image space metrics (writes data/{folder_key}/evaluation.csv)
            evaluate_by_model_in_image_space(model=folder_key)
            img_df = pd.read_csv(f"data/{folder_key}/evaluation.csv")

            for _, row in img_df.iterrows():
                summary.append({
                    "latent_dim": dim,
                    "gamma": gamma,
                    "fraction": row["Fraction of transport"],
                    "mmd_latent": float(mmd_latent),
                    "mmd_image": row["MMD"],
                    "classifier_confidence": row["Confidence of Classifier"],
                    "fid": row["FID"],
                })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("data/evaluation_summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary_df.to_string())


if __name__ == "__main__":
    run_evaluation()
