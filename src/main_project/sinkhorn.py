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
from main_project.environment import GAMMA, MODELS_DIM, INTERMEDIATE_FRACTIONS, MAX_POINTS, MAX_ITERATION, TQDM_SINKHORN,VERBOSE_OPTIMAL_TRANSPORT, STOP_THRESSHOLD


gamma = 1e-3



@jax.jit
def cdist_euclidean_v0(x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes pairwise Euclidean distances between rows of x and y.

    Args:
        x: Array of shape (N, D)
        y: Array of shape (M, D)

    Returns:
        Distance matrix of shape (N, M)
    """
    return jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))

@jax.jit
def cosine_similarity(x,y):
    dot_product = jnp.dot(x,y)
    x_norm = jnp.linalg.norm(x)
    y_norm = jnp.linalg.norm(y)
    similarity = dot_product/(x_norm*y_norm)
    return similarity



@jax.jit
def cdist_euclidean(x, y):  # More efficient, since it written out
    x_sq = jnp.sum(x**2, axis=1)  # (N,)
    y_sq = jnp.sum(y**2, axis=1)  # (M,)
    cross = x @ y.T  # (N, M)  — efficient BLAS matmul
    sq_dists = x_sq[:, None] + y_sq[None, :] - 2 * cross
    sq_dists = jnp.clip(sq_dists, 0.0)  # numerical safety before sqrt
    return jnp.sqrt(sq_dists)


def sinkhorn_simple(
    s: jax.Array,
    d: jax.Array,
    C: jax.Array,
    gamma: float = 0.1,
    eps: float = 1e-3,
    max_iters: int = 100,
    stop_thresh: float = 1e-5,
    verbose: bool = False,
) -> jax.Array:
    """Sinkhorn algorithm for regularised optimal transport.

    Args:
        s:           Source marginal distribution, shape (N,)
        d:           Target marginal distribution, shape (M,)
        C:           Cost matrix, shape (N, M)
        gamma:       Entropic regularisation strength
        eps:         Unused (kept for API compatibility)
        max_iters:   Maximum number of Sinkhorn iterations
        stop_thresh: Early-stop threshold on change in u / v
        verbose:     Print iteration count on early stop

    Returns:
        Transport plan T of shape (N, M)
    """
    u, v = jnp.ones_like(s), jnp.ones_like(d)
    K = jnp.exp(-C / gamma)

    for i in range(max_iters):
        u_prev, v_prev = u, v
        u = s / (jnp.dot(K, v) + 1e-8)
        v = d / (jnp.dot(K.T, u) + 1e-8)

        if jnp.max(jnp.abs(u_prev - u)) < stop_thresh and jnp.max(jnp.abs(v_prev - v)) < stop_thresh:
            if verbose:
                print(f"Converged at iteration {i}")
            break

    # Outer product scaling: T[i,j] = u[i] * K[i,j] * v[j]
    P = u[:, None] * K * v[None, :]
    return P, u, v


def sinkhorn_log(s, d, C, gamma=0.1, max_iters=1000, stop_thresh=1e-7, verbose=False):
    log_s = jnp.log(s)
    log_d = jnp.log(d)
    u = jnp.zeros_like(s)
    v = jnp.zeros_like(d)

    for iter in tqdm(range(max_iters), desc="Sinkhorn iteration", disable=TQDM_SINKHORN):
        u_prev = u   # ← save before update
        v_prev = v   # ← save before update

        u = gamma * (log_s - jax.nn.logsumexp((v[None, :] - C) / gamma, axis=1))
        v = gamma * (log_d - jax.nn.logsumexp((u[:, None] - C) / gamma, axis=0))

        if (jnp.max(jnp.abs(u_prev - u)) < stop_thresh and
            jnp.max(jnp.abs(v_prev - v)) < stop_thresh):
            if verbose:
                print(f"Converged at iteration {iter}")
            break

    # Transport plan in log-space
    log_P = (u[:, None] + v[None, :] - C) / gamma
    P = jnp.exp(log_P)
    return P, u, v, iter


def load_source_and_target_arrays(source_label=0, target_label=1):
    _, test_data = getData()
    test_loader = getDataloader(test_data)

    source_data = []
    target_data = []

    for x, y in test_loader:
        source_mask = y == source_label
        target_mask = y == target_label
        source_data.extend(x[source_mask])
        target_data.extend(x[target_mask])

    source_arr = jnp.asarray(torch.stack(source_data).reshape(len(source_data), -1).numpy())
    target_arr = jnp.asarray(torch.stack(target_data).reshape(len(target_data), -1).numpy())

    min_count = min(source_arr.shape[0], target_arr.shape[0])
    source_arr = source_arr[:min_count]
    target_arr = target_arr[:min_count]

    return source_arr, target_arr


def run_sinkhorn_by_model(model, gamma,distance_metric="euclidean", source_label=0, target_label=1):
    source_arr, target_arr = load_source_and_target_arrays(source_label, target_label)

    def recon(x):
        recon, z = model(x)
        return recon, z

    # Gets latent representation of source and target distributions
    latent_source = jax.vmap(recon)(source_arr)[1]  # [n, latent_dim]
    latent_target = jax.vmap(recon)(target_arr)[1]  # [m, latent_dim]

    # Used to ensure no out of bounds error
    min_count = min(latent_source.shape[0], latent_target.shape[0])

    latent_source = latent_source[:min_count]
    latent_target = latent_target[:min_count]
    if distance_metric == "euclidean":
        C = cdist_euclidean(latent_source, latent_target)
    elif distance_metric == "cosine":
        C = cosine_similarity(latent_source, latent_target)
    s = jnp.ones(latent_source.shape[0]) / latent_source.shape[0]
    d = jnp.ones(latent_target.shape[0]) / latent_target.shape[0]

    # Now uniform weights work fine — equal n and m
    P, u, v, iter = sinkhorn_log(
        C=C, s=s, d=d, gamma=gamma, max_iters=MAX_ITERATION, stop_thresh=STOP_THRESSHOLD, verbose=True
    )

    # print(f"Sinkhorn converged in {iter} iterations with gamma={gamma} and threshold={stop_threshold}")

    return latent_source, latent_target, P, u, v, iter


def get_probability_y_given_x(P, index):
    p_y_given_x = P[index] / P[index].sum()
    return p_y_given_x


def main():
    data = []
    running_times = {}
    # warmup_jax(n=MAX_POINTS, dim=MODELS_DIM[0]) run only when interested in running time
    for dim in MODELS_DIM:
        model_name = f"ae_best_model_{dim}"
        for gamma in GAMMA:
            model = load(model_name)
            latent_source, latent_target, P, u, v, iter = run_sinkhorn_by_model(model=model,gamma=gamma)
            data.append((model_name, gamma, iter))
        start = time.perf_counter()
        print("Saving sinkhorn transformation for", model_name)
        # save_sinkhorn_transformation(model_name = model_name, save =True)
        running_times[dim] = time.perf_counter() - start

    # save_sinkhorn_transformation_without_ae()

    for dim, t in running_times.items():
        print(f"dim={dim:4d}  took {t:.2f}s")


if __name__ == "__main__":
    main()
