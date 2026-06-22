import torch
import numpy as np
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
import pandas as pd
import pickle
import os
from itertools import permutations
from tqdm import tqdm

# example of calculating the frechet inception distance
import numpy
from numpy.random import random
from scipy.stats import wasserstein_distance_nd


# import ot  # POT library

# def sinkhorn_wasserstein(x, y, reg=0.1):
#     n, m = x.shape[0], y.shape[0]
#     a, b = np.ones(n)/n, np.ones(m)/m
#     M = ot.dist(x, y)  # cost matrix
#     return ot.sinkhorn2(a, b, M, reg)

from main_project.model import targetClassifier
from main_project.utils import load
from main_project.environment import MODELS_DIM, INTERMEDIATE_FRACTIONS, MAX_POINTS, GAMMA, LABELS

SEED = 5678
key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)


# #calculate frechet inception distance
# def calculate_fid(act1, act2):
#     # calculate mean and covariance statistics
#     mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
#     mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
#     # calculate sum squared difference between means
#     ssdiff = numpy.sum((mu1 - mu2)**2.0)
#     # calculate sqrt of product between cov
#     covmean = sqrtm(sigma1.dot(sigma2))
#     # check and correct imaginary numbers from sqrt
#     if iscomplexobj(covmean):
#         covmean = covmean.real
#     # calculate score
#     fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
#     return fid


@jax.jit
def median_bandwidth(x, y):
    z = jnp.concatenate([x, y], axis=0)
    sq_norms = jnp.sum(z**2, axis=-1)
    dists = sq_norms[:, None] + sq_norms[None, :] - 2 * (z @ z.T)
    n = dists.shape[0]
    mask = ~jnp.eye(n, dtype=bool)
    masked = jnp.where(mask, dists, jnp.nan)
    return jnp.nanmedian(masked)


# https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
from functools import partial


@partial(jax.jit, static_argnames=("kernel"))
def MMD(x: Array, y: Array, kernel):
    xx, yy, zz = jnp.matmul(x, x.T), jnp.matmul(y, y.T), jnp.matmul(x, y.T)

    rx = jnp.diag(xx)[jnp.newaxis, :]  # (1, N)
    ry = jnp.diag(yy)[jnp.newaxis, :]  # (1, M)

    # This is the expanded exponential kernel - XX corresponding to element x_i times x_i
    dxx = rx.T + rx - 2.0 * xx
    dyy = ry.T + ry - 2.0 * yy
    dxy = rx.T + ry - 2.0 * zz

    XX = jnp.zeros_like(dxx)
    YY = jnp.zeros_like(dyy)
    XY = jnp.zeros_like(dxy)

    if kernel == "rbf":
        
        med = median_bandwidth(x, y)  # Skriv hvorfor i teori
        bandwidth_range = [0.5 * med, med, 2 * med, 4 * med]

            #
        for a in bandwidth_range:
            XX += (jnp.exp(-0.5 * dxx / a)) /len(bandwidth_range)
            YY += (jnp.exp(-0.5 * dyy / a)) /len(bandwidth_range)
            XY += (jnp.exp(-0.5 * dxy / a))/len(bandwidth_range)

    # MMD² = E[k(x,x')] + E[k(y,y')] − 2·E[k(x,y)]
    #return jnp.mean(XX + YY - 2.0 * XY)
    return jnp.mean(XX) + jnp.mean(YY) - 2.0 * jnp.mean(XY)


classifier = load(name="evaluate_classifier", path="models", model=targetClassifier(subkey))


@jax.jit
def classifier_confidence(transported_images, target_class=1):
    x = jnp.asarray(transported_images)

    log_probs = jax.vmap(classifier)(x)
    probs = jnp.exp(log_probs)  # Shape: [n, 10]

    p_target = probs[:, target_class]
    # predictions = jnp.argmax(probs, axis=-1)
    # classified = jnp.mean(predictions == target_class)

    return jnp.mean(p_target)


def entropy_for_transport_plan(T: jnp.ndarray):
    T = T / (jnp.sum(T))
    mask = T > 0
    entropy = -jnp.sum(jnp.where(mask, T * jnp.log(T), 0.0))
    return entropy


columns = ["Fraction of transport", "MMD", "Confidence of Classifier", "FID"]


def evaluate_by_model_in_image_space(sinkhorn_data, target_label, save_dir, decoder):
    #target_img = jnp.asarray(sinkhorn_data["target_images"])

    target_img = jax.vmap(decoder)(sinkhorn_data["target_eval"])
    expected_target_img = jnp.asarray(sinkhorn_data["expected_target_images"])

    if "intermediate_images" in sinkhorn_data:
        intermediate_images = sinkhorn_data["intermediate_images"]

        # (n_points, n_fracs, 784) -> transpose to (n_fracs, n_points, 784)
        intermediate_images = intermediate_images.transpose(1, 0, 2)

        data = []
        for frac, imgs in zip(INTERMEDIATE_FRACTIONS, intermediate_images):
            mmd = MMD(jnp.asarray(imgs), jnp.asarray(target_img), kernel="rbf")
            classifier_conf = classifier_confidence(imgs, target_label)
            # fid = calculate_fid(np.asarray(imgs), np.asarray(target_img))
            data.append([frac, float(mmd), float(jnp.mean(classifier_conf))])  # fid

        df = pd.DataFrame(data, columns=columns)

        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(f"{save_dir}/evaluation.csv", index=False)

    mmd = MMD(jnp.asarray(expected_target_img), jnp.asarray(target_img), kernel="rbf")

    classifier_conf = classifier_confidence(expected_target_img, target_label)
    # fid = calculate_fid(np.asarray(expected_target_img), np.asarray(target_img))

    return (
        mmd,
        classifier_conf,
    )  # fid


def evaluate_by_model_in_latent_space(sinkhorn_data):
    # Use the held-out target split that was never seen by Sinkhorn.
    # Comparing against the in-sample target is not out-of-sample evaluation —
    # the plan is optimised to match those exact points.
    target_eval = sinkhorn_data["target_eval"]
    expected_target = sinkhorn_data["expected_target"]
   

    mmd = MMD(jnp.asarray(target_eval), jnp.asarray(expected_target), kernel="rbf")
    wasserstein_distance = wasserstein_distance_subsampled(np.asarray(expected_target), np.asarray(target_eval))

    return mmd, wasserstein_distance


def wasserstein_distance_subsampled(source, target, max_points=300, n_seeds=5):
    source = np.asarray(source)
    target = np.asarray(target)
    n = min(source.shape[0], target.shape[0], max_points)
    
    distances = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx1 = rng.choice(source.shape[0], n, replace=False)
        idx2 = rng.choice(target.shape[0], n, replace=False)
        distances.append(wasserstein_distance_nd(source[idx1], target[idx2]))
    
    return float(np.mean(distances))


# def wasserstein_distance_subsampled(source_imgs, target_imgs):

#     distances = []
#     for source_img, target_img in zip(source_imgs,target_imgs):
#         wasserstein_distance = wasserstein_distance_nd(np.asarray(source_img), np.asarray(target_img))
#         distances.append(wasserstein_distance)

#     return np.mean(distances)

from main_project.utils import load, load_with_hyperparams
def run_evaluation():
    summary = []

    for dim in MODELS_DIM:
        model_name = f"ae_best_model_bo_{dim}"
        pickle_filename = f"data/{model_name}_ot_data.pkl"

        model = load_with_hyperparams(name=model_name, path="models")

        if not os.path.exists(pickle_filename):
            print(f"Skipping {model_name} because {pickle_filename} does not exist.")
            continue

        with open(pickle_filename, "rb") as f:
            model_transport_data = pickle.load(f)

        for gamma_val in GAMMA:
            if gamma_val not in model_transport_data:
                continue

            for source_label, target_label in tqdm(
                permutations(LABELS, 2), f"evaluating numbers for gamma {gamma_val} and model {model_name}"
            ):
                label_key = f"source_{source_label}_target_{target_label}"

                if label_key not in model_transport_data[gamma_val]:
                    continue

                sinkhorn_data = model_transport_data[gamma_val][label_key]

                # Latent space MMD evaluation
                mmd_latent, wasserstein_distance_latent = evaluate_by_model_in_latent_space(sinkhorn_data)

                # Directory to mirror evaluation artifacts locally
                save_dir = f"data/{model_name}/{gamma_val}/{label_key}"

                # Image space metrics assessment
                mmd_img, classifier_conf_img = evaluate_by_model_in_image_space(
                    sinkhorn_data=sinkhorn_data, target_label=target_label, save_dir=save_dir, decoder = model.decoder
                )
                # fid_img =

                entropy = entropy_for_transport_plan(jnp.asarray(sinkhorn_data["P"]))


                summary.append(
                    {
                        "latent_dim": dim,
                        "gamma": gamma_val,
                        "source_label": source_label,
                        "target_label": target_label,
                        "mmd_latent": float(mmd_latent),
                        "wasserstein_distance_latent": float(wasserstein_distance_latent),
                        "mmd_image": mmd_img,
                        "classifier_confidence_image": classifier_conf_img,
                        # "fid_image": fid_img,
              
                        "entropy": entropy,
                        "running_time": sinkhorn_data["running_time"],
                        "iter_count": sinkhorn_data["iter_count"],
                    }
                )

    if summary:
        summary_df = pd.DataFrame(summary)

        os.makedirs("data", exist_ok=True)
        summary_df.to_csv("data/evaluation_summary.csv", index=False)
        print("\n=== Summary ===")
        print(summary_df.to_string())
    else:
        print("No evaluation data extracted. Verify dictionary keys and configuration matches.")


if __name__ == "__main__":
    run_evaluation()
