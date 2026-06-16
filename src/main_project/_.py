import os
import re
import time
import pickle
import torch
from typing import Union
from itertools import combinations

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from tqdm import tqdm

from main_project.model import AEv2
from main_project.utils import load, load_with_hyperparams
from main_project.data import getData, getDataloader
from main_project.environment import (
    GAMMA,
    MODELS_DIM,
    INTERMEDIATE_FRACTIONS,
    MAX_POINTS,
    MAX_ITERATION,
    STOP_THRESHOLD,
    LABELS,
    SAVE_INTERMEDIATE,
    VERBOSE_OPTIMAL_TRANSPORT,
)

from main_project.sinkhorn import (
    run_sinkhorn_by_model,
    get_probability_y_given_x,
    cdist_euclidean,
    sinkhorn_log,
    load_source_and_target_arrays,
)

from main_project.schrodinger_bridge import SchrodingerBridge, density_weights, run_sb

stop_threshold = STOP_THRESHOLD
gamma = 1e-3


def get_trajectory(source, target, target_eval, P, decoder=None):
    decode = decoder if decoder is not None else lambda z: z
    n_points = min(MAX_POINTS, len(target_eval))

    source_images = []
    target_images = []
    expected_target_images = []
    intermediate_images = []
    expected_target_latent = []
    original_target_latent = []

    fractions = jnp.array(INTERMEDIATE_FRACTIONS)

    for i in range(n_points):
        p_y_given_x = get_probability_y_given_x(P, i)
        x_star = jnp.array(source[i].squeeze())
        #y_point = jnp.array(target[i].squeeze())
        y_point = jnp.array(target_eval[i].squeeze())

        expected_target = jnp.array((p_y_given_x @ target).squeeze())

        expected_target_latent.append(expected_target)
        original_target_latent.append(y_point)

        source_images.append(np.array(decode(x_star)))
        target_images.append(np.array(decode(y_point)))
        expected_target_images.append(np.array(decode(expected_target)))

        # Optimized Intermediate Latent & Image decoding
        if SAVE_INTERMEDIATE:
            # Vectorized calculation of intermediate mixtures: shape (F, latent_dim)
            # (1 - f) * x_star + f * expected_target
            inter_latents = (1.0 - fractions[:, None]) * x_star + fractions[:, None] * expected_target

            # Using vmap to decode all intermediate frames efficiently at once
            decoded_inter = jax.vmap(decode)(inter_latents)
            intermediate_images.append(np.array(decoded_inter))

    return {
        # Latent
        "target": np.array(original_target_latent),
        "expected_target": np.array(expected_target_latent),
        # Images
        "source_images": np.array(source_images),
        "target_images": np.array(target_images),
        "expected_target_images": np.array(expected_target_images),
        "intermediate_images": np.array(intermediate_images),
    }


def save_sinkhorn_transformation(model_name, gamma=1e-3, source_label=0, target_label=1):
    model = load_with_hyperparams(name=model_name, path="models")

    t0 = time.perf_counter()
    latent_source, latent_target, latent_target_eval, P, _, _, iter_count, running_time = run_sinkhorn_by_model(
        model, gamma=gamma, source_label=source_label, target_label=target_label
    )
    t1 = time.perf_counter()

    trajectory = get_trajectory(
        latent_source,
        latent_target,
        latent_target_eval
        P=P,
        decoder=model.decoder,
    )
    t2 = time.perf_counter()
    if VERBOSE_OPTIMAL_TRANSPORT:
        print(f"  sinkhorn: {t1-t0:.2f}s  |  decoding: {t2-t1:.2f}s")

    keys_to_save = ["target", "expected_target", "source_images", "target_images", "expected_target_images"]
    if SAVE_INTERMEDIATE:
        keys_to_save.append("intermediate_images")

    sinhorn_trajectory = {key: trajectory[key] for key in keys_to_save}
    sinhorn_trajectory["iter_count"] = iter_count
    sinhorn_trajectory["P"] = P
    sinhorn_trajectory["running_time"] = running_time
    # Held-out target latents — never seen by Sinkhorn, used for out-of-sample MMD
    sinhorn_trajectory["target_eval"] = np.array(latent_target_eval)

    return iter_count, sinhorn_trajectory


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

        model_transport_data = {}

        for gamma_val in GAMMA:
            print(f"\n--- gamma={gamma_val} ---")

            if gamma_val not in model_transport_data:
                model_transport_data[gamma_val] = {}

            for source_label, target_label in combinations(LABELS, 2):
                if VERBOSE_OPTIMAL_TRANSPORT:
                    print(f"\n--- source = {source_label} and target = {target_label} ---")
                start = time.perf_counter()

                iter_count, sinhorn_trajectory = save_sinkhorn_transformation(
                    model_name=model_name, gamma=gamma_val, source_label=source_label, target_label=target_label
                )

                elapsed = time.perf_counter() - start
                data.append((dim, gamma_val, iter_count, elapsed, source_label, target_label))

                # Assignment using structured string keys without nested initialization collision
                label_key = f"source_{source_label}_target_{target_label}"
                model_transport_data[gamma_val][label_key] = sinhorn_trajectory

        pickle_filename = f"data/{model_name}_ot_data.pkl"
        with open(pickle_filename, "wb") as f:
            pickle.dump(model_transport_data, f)
        print(f"Saved optimal transport data to {pickle_filename}")

    df = pd.DataFrame(
        data, columns=["dim", "gamma", "sinkhorn_iterations", "sinkhorn_time", "source_label", "target_label"]
    )
    df.to_csv("sinkhorn_iterations_and_times.csv", index=False)


if __name__ == "__main__":
    save_transformations()
