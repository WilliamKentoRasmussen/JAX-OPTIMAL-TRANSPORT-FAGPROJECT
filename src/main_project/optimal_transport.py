import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import os
import jax.random as jr
import re
import time
import pandas as pd
from main_project.model import AEv2
from main_project.utils import load, load_with_hyperparams
from main_project.data import getData, getDataloader
from main_project.environment import GAMMA, MODELS_DIM, INTERMEDIATE_FRACTIONS, MAX_POINTS, MAX_ITERATION,GAMMA

from main_project.sinkhorn import (
    run_sinkhorn_by_model,
    get_probability_y_given_x,
    cdist_euclidean,
    sinkhorn_log,
    load_source_and_target_arrays,
)

from main_project.schrodinger_bridge import SchrodingerBridge, density_weights, run_sb

stop_threshold = 1e-5
gamma = 1e-3

def load_model(model_name: str) -> AEv2:
    latent_dim = int(re.search(r"_(\d+)", model_name).group(1))

    return load(name=model_name, path="models", latent_dim=latent_dim)  #


def get_trajectory(source, target, P, decoder=None):
    """
    Computes all trajectory data in a single pass.
    Always returns both latent and image data.
    """
    decode   = decoder if decoder is not None else lambda z: z
    n_points = min(MAX_POINTS, len(source))

    source_images          = []
    target_images          = []
    expected_target_images = []
    intermediate_images    = []
    expected_target_latent = []
    original_target_latent = []

    for i in range(n_points):
        p_y_given_x    = get_probability_y_given_x(P, i)
        x_star         = jnp.array(source[i].squeeze())
        y_point        = jnp.array(target[i].squeeze())
        expected_target = jnp.array((p_y_given_x @ target).squeeze())

        # Latent
        expected_target_latent.append(expected_target)
        original_target_latent.append(y_point)

        # Images — computed in same pass
        source_images.append(np.array(decode(x_star)))
        target_images.append(np.array(decode(y_point)))
        expected_target_images.append(np.array(decode(expected_target)))
        intermediate_images.append([
            np.array(decode((1 - f) * x_star + f * expected_target))
            for f in INTERMEDIATE_FRACTIONS
        ])

    return {
        # Latent
        "y_original":       np.array(original_target_latent),
        "expected_target":  np.array(expected_target_latent),
        # Images
        "original_images":        np.array(source_images),
        "target_images":          np.array(target_images),
        "expected_target_images": np.array(expected_target_images),
        "intermediate_images":    np.array(intermediate_images),
    }


def save_sinkhorn_transformation(model_name, save=True, gamma=1e-3):
    model = load_with_hyperparams(name=model_name, path="models")

    t0 = time.perf_counter()
    latent_source, latent_target, P, u, v, iter = run_sinkhorn_by_model(model, gamma=gamma)
    t1 = time.perf_counter()

    # Single pass — always compute everything
    trajectory = get_trajectory(
        latent_source, latent_target,
        P=P,
        decoder=model.decoder,
    )
    t2 = time.perf_counter()
    print(f"  sinkhorn: {t1-t0:.2f}s  |  decoding: {t2-t1:.2f}s")

    if save:
        save_dir = f"data/{model_name}_{gamma}"
        os.makedirs(save_dir, exist_ok=True)

        # Choose which keys to save based on latent_evaluate flag
        keys_to_save = (
            ["y_original", "expected_target","original_images", "target_images",
                  "expected_target_images", "intermediate_images"]
        )

        for key in keys_to_save:
            np.save(f"{save_dir}/{key}.npy", trajectory[key])

        print(f"  saved {keys_to_save} to {save_dir}/")

    return iter




# def save_sinkhorn_transformation_without_ae(save=True):
#     source_arr, target_arr = load_source_and_target_arrays()
#     t0 = time.perf_counter()
#     C = cdist_euclidean(source_arr, target_arr)
#     s = jnp.ones(source_arr.shape[0]) / source_arr.shape[0]
#     d = jnp.ones(target_arr.shape[0]) / target_arr.shape[0]
#     P, u, v, iter = sinkhorn_log(
#         C=C, s=s, d=d, gamma=gamma, max_iters=MAX_ITERATION, stop_thresh=stop_threshold, verbose=True
#     )
#     t1 = time.perf_counter()
#     trajectory = get_trajectory(source_arr, target_arr, P=P, decoder=None)
#     t2 = time.perf_counter()
#     print(f"  sinkhorn: {t1-t0:.2f}s  |  decoding: {t2-t1:.2f}s")

#     if save:
#         save_dir = f"data/no_ae_{gamma}"
#         os.makedirs(save_dir, exist_ok=True)

#         for img_name, img_matrix in trajectory.items():
#             np.save(f"{save_dir}/{img_name}.npy", np.array(img_matrix))


def encode_with_model(model: AEv2, source_arr, target_arr):
    def recon(x):
        r, z = model(x)
        return r, z

    latent_source = jax.vmap(recon)(source_arr)[1]
    latent_target = jax.vmap(recon)(target_arr)[1]

    min_count = min(latent_source.shape[0], latent_target.shape[0])
    latent_source = latent_source[:min_count]
    latent_target = latent_target[:min_count]

    return latent_source, latent_target




# def save_sb_transformation(model_name, save=True):
#     model = load_model(model_name=model_name)
#     source_arr, target_arr = load_source_and_target_arrays()
#     latent_source, latent_target = encode_with_model(model, source_arr=source_arr, target_arr=target_arr)

#     t0 = time.perf_counter()
#     bridge, P = run_sb(latent_source, latent_target)
#     t1 = time.perf_counter()
#     result = bridge.sample_trajectories(P=P, model=model, n_samples=MAX_POINTS)
#     t2 = time.perf_counter()

#     print(f"  IPF fit: {t1-t0:.2f}s  |  trajectory sampling: {t2-t1:.2f}s")

#     if save:
#         save_dir = f"data/sb_{model_name}"
#         os.makedirs(save_dir, exist_ok=True)
#         np.save(f"{save_dir}/decoded.npy", result["decoded"])
#         np.save(f"{save_dir}/trajectories.npy", result["trajectories"])


def warmup_jax(n, dim):
    dummy_x = jnp.ones((n, dim))
    dummy_y = jnp.ones((n, dim))
    s = jnp.ones(n) / n
    d = jnp.ones(n) / n
    C = cdist_euclidean(dummy_x, dummy_y)
    P, *_ = sinkhorn_log(s, d, C, gamma=gamma, max_iters=2)
    jax.block_until_ready(P)


def save_transformations():
    data = []

    for dim in MODELS_DIM:
        model_name = f"ae_best_model_bo_{dim}"
        print(f"\n--- dim={dim} ---")
        for gamma in GAMMA:
            print(f"  gamma={gamma}")
            start = time.perf_counter()
            iter = save_sinkhorn_transformation(model_name=model_name, save=True, latent_evaluate=True, gamma=gamma)
            elapsed = time.perf_counter() - start
            data.append((dim, gamma, iter, elapsed))

    df = pd.DataFrame(data, columns=["dim", "gamma", "sinkhorn_iterations", "sinkhorn_time"])
    df.to_csv("sinkhorn_iterations_and_times.csv", index=False)
if __name__ == "__main__":
    save_transformations()
