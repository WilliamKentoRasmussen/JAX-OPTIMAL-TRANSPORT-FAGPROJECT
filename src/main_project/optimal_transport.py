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

from main_project.model import AEv2
from main_project.utils import load
from main_project.data import getData, getDataloader
from main_project.environment import MODELS_DIM, INTERMEDIATE_FRACTIONS, MAX_POINTS, MAX_ITERATION

from main_project.sinkhorn import (
    run_sinkhorn_by_model,
    get_probability_y_given_x,
    cdist_euclidean,
    sinkhorn_log,
    load_source_and_target_arrays,
)

from main_project.schrodinger_bridge import SchrodingerBridge, density_weights, run_sb

gamma = 1e-3
stop_threshold = 1e-5


def load_model(model_name: str) -> AEv2:
    latent_dim = int(re.search(r"_(\d+)", model_name).group(1))

    return load(name=model_name, path="models", latent_dim=latent_dim)  #


def get_trajectory(source, target, P, decoder=None,latent=False):
    source_images = []
    target_images = []
    expected_target_images = []
    intermediate_images = []
    expected_target_latent = []
    original_target_latent = []

    n_points = min(MAX_POINTS, len(source))

    for i in range(n_points):
        p_y_given_x = get_probability_y_given_x(P, i)

        x_star = jnp.array(source[i].squeeze())
        y_point = jnp.array(target[i].squeeze())
        expected_target = jnp.array((p_y_given_x @ target).squeeze())

        decode = decoder if decoder is not None else lambda z: z

        intermediates = [np.array(decode((1 - f) * x_star + f * expected_target)) for f in INTERMEDIATE_FRACTIONS]



        expected_target_latent.append(expected_target)
        original_target_latent.append(y_point)
        source_images.append(np.array(decode(x_star)))
        target_images.append(np.array(decode(y_point)))
        expected_target_images.append(np.array(decode(expected_target)))
        intermediate_images.append(intermediates)
    if latent:
        return {"y_original": np.array(original_target_latent),
        "expected_target": np.array(expected_target_latent)}
    else:
        return {        
            "original_images": np.array(source_images),
            "target_images": np.array(target_images),
            "expected_target_images": np.array(expected_target_images),
            "intermediate_images": np.array(intermediate_images),
        }


def save_sinkhorn_transformation(model_name, save=True, latent=False):
    model = load_model(model_name=model_name)  # ae_best_model_lat2
    t0 = time.perf_counter()
    latent_source, latent_target, P, u, v, iter = run_sinkhorn_by_model(model)
    t1 = time.perf_counter()
    if latent:
        trajectory= get_trajectory(latent_source, latent_target, P=P, decoder=model.decoder, latent=latent)
        t2 = time.perf_counter()
        print(f"  sinkhorn: {t1-t0:.2f}s  |  decoding: {t2-t1:.2f}s")
        if save:
            save_dir = f"data/{model_name}"
            os.makedirs(save_dir, exist_ok=True)
            np.save(f"{save_dir}/y_original.npy", trajectory["y_original"])
            np.save(f"{save_dir}/expected_target.npy", trajectory["expected_target"])
            

    else:
        trajectory= get_trajectory(latent_source, latent_target, P=P, decoder=model.decoder)
        t2 = time.perf_counter()
        print(f"  sinkhorn: {t1-t0:.2f}s  |  decoding: {t2-t1:.2f}s")
        if save:
            save_dir = f"data/{model_name}"
            os.makedirs(save_dir, exist_ok=True)

            for img_name, img_matrix in trajectory.items():
                np.save(f"{save_dir}/{img_name}.npy", np.array(img_matrix))




def save_sinkhorn_transformation_without_ae(save=True):
    source_arr, target_arr = load_source_and_target_arrays()
    t0 = time.perf_counter()
    C = cdist_euclidean(source_arr, target_arr)
    s = jnp.ones(source_arr.shape[0]) / source_arr.shape[0]
    d = jnp.ones(target_arr.shape[0]) / target_arr.shape[0]
    P, u, v, iter = sinkhorn_log(
        C=C, s=s, d=d, gamma=gamma, max_iters=MAX_ITERATION, stop_thresh=stop_threshold, verbose=True
    )
    t1 = time.perf_counter()
    trajectory = get_trajectory(source_arr, target_arr, P=P, decoder=None)
    t2 = time.perf_counter()
    print(f"  sinkhorn: {t1-t0:.2f}s  |  decoding: {t2-t1:.2f}s")

    if save:
        save_dir = f"data/no_ae"
        os.makedirs(save_dir, exist_ok=True)

        for img_name, img_matrix in trajectory.items():
            np.save(f"{save_dir}/{img_name}.npy", np.array(img_matrix))


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




def save_sb_transformation(model_name, save=True):
    model = load_model(model_name=model_name)
    source_arr, target_arr = load_source_and_target_arrays()
    latent_source, latent_target = encode_with_model(model, source_arr=source_arr, target_arr=target_arr)

    t0 = time.perf_counter()
    bridge, P = run_sb(latent_source, latent_target)
    t1 = time.perf_counter()
    result = bridge.sample_trajectories(P=P, model=model, n_samples=MAX_POINTS)
    t2 = time.perf_counter()

    print(f"  IPF fit: {t1-t0:.2f}s  |  trajectory sampling: {t2-t1:.2f}s")

    if save:
        save_dir = f"data/sb_{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/decoded.npy", result["decoded"])
        np.save(f"{save_dir}/trajectories.npy", result["trajectories"])


def warmup_jax(n, dim):
    dummy_x = jnp.ones((n, dim))
    dummy_y = jnp.ones((n, dim))
    s = jnp.ones(n) / n
    d = jnp.ones(n) / n
    C = cdist_euclidean(dummy_x, dummy_y)
    T, *_ = sinkhorn_log(s, d, C, gamma=gamma, max_iters=2)
    jax.block_until_ready(T)


def save_transformations():
    running_times_sinkhorn = {}
    running_times_sb = {}

    # warmup_jax(n=MAX_POINTS, dim=MODELS_DIM[0])

    for dim in MODELS_DIM:
        model_name = f"ae_model_dim_{dim}"
        print(f"\n--- dim={dim} ---")

        start = time.perf_counter()
        save_sinkhorn_transformation(model_name=model_name, save=True, latent=True)
        running_times_sinkhorn[dim] = time.perf_counter() - start
    
    #     start = time.perf_counter()
    #     save_sb_transformation(model_name=model_name, save=True)
    #     running_times_sb[dim] = time.perf_counter() - start

    # print("\n--- Summary ---")
    # print(f"{'dim':>6}  {'sinkhorn':>12}  {'schr. bridge':>12}")
    # for dim in MODELS_DIM:
    #     print(f"{dim:>6}  {running_times_sinkhorn[dim]:>10.2f}s  {running_times_sb[dim]:>10.2f}s")


if __name__ == "__main__":
    save_transformations()
