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


gamma = 1e-3
stop_threshold = 1e-5

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
def cdist_euclidean(x, y): #More efficient, since it written out
    x_sq = jnp.sum(x ** 2, axis=1)        # (N,)
    y_sq = jnp.sum(y ** 2, axis=1)        # (M,)
    cross = x @ y.T                        # (N, M)  — efficient BLAS matmul
    sq_dists = x_sq[:, None] + y_sq[None, :] - 2 * cross
    sq_dists = jnp.clip(sq_dists, 0.0)    # numerical safety before sqrt
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


def sinkhorn_log(s, d, C, gamma=0.1, max_iters=1000, stop_thresh=1e-5, verbose=False):
    log_s = jnp.log(s)
    log_d = jnp.log(d)

    u = jnp.zeros_like(s)
    v = jnp.zeros_like(d)
    iter = 0

    for iter in tqdm(range(max_iters), "Sinkhorn iteration"):
        u = gamma * (log_s - jax.nn.logsumexp((v[None, :] - C) / gamma, axis=1))
        v = gamma * (log_d - jax.nn.logsumexp((u[:, None] - C) / gamma, axis=0))
        iter += 1

    # transport plan in log-space
    log_P = (u[:, None] + v[None, :] - C) / gamma
    P = jnp.exp(log_P)

    return P, u, v, iter




def run_sinkhorn_by_model(model):
    training_data, _ = getData()
    train_loader = getDataloader(training_data)

    start_data = []
    target_data = []

    for x, y in train_loader:
        start_mask = y == 0
        target_mask = y == 1
        start_data.extend(x[start_mask])
        target_data.extend(x[target_mask])

    # Stack and flatten to [n, 784] as JAX arrays
    source_arr = jnp.asarray(torch.stack(start_data).reshape(len(start_data), -1).numpy())
    target_arr = jnp.asarray(torch.stack(target_data).reshape(len(target_data), -1).numpy())

    def recon(x):
        recon, z = model(x)
        return recon, z

    #Gets latent representation of source and target distributions
    latent_source = jax.vmap(recon)(source_arr)[1]  # [n, latent_dim]
    latent_target = jax.vmap(recon)(target_arr)[1]  # [m, latent_dim]

    # Used to ensure no out of bounds error
    min_count = min(latent_source.shape[0], latent_target.shape[0])

    latent_source = latent_source[:min_count]
    latent_target = latent_target[:min_count]

    C = cdist_euclidean(latent_source, latent_target)
    s = jnp.ones(latent_source.shape[0]) / latent_source.shape[0]
    d = jnp.ones(latent_target.shape[0]) / latent_target.shape[0]

    # Now uniform weights work fine — equal n and m
    T, u, v, iter = sinkhorn_log(C=C, s=s, d=d, gamma=gamma, max_iters=MAX_ITERATION, stop_thresh=stop_threshold, verbose=True)

    #print(f"Sinkhorn converged in {iter} iterations with gamma={gamma} and threshold={stop_threshold}")

    return latent_source, latent_target, T, u, v, iter




def get_probability_y_given_x(T, index):
    p_y_given_x = T[index] / T[index].sum()
    return p_y_given_x


def main():

    running_times = {}

    #warmup_jax(n=MAX_POINTS, dim=MODELS_DIM[0]) run only when interested in running time
    for dim in MODELS_DIM: 
        model_name = f"ae_model_dim_{dim}"
        start = time.perf_counter()
        print("Saving sinkhorn transformation for", model_name)
        #save_sinkhorn_transformation(model_name = model_name, save =True)

        running_times[dim] = time.perf_counter()-start

    #save_sinkhorn_transformation_without_ae()
    
    for dim, t in running_times.items():
        print(f"dim={dim:4d}  took {t:.2f}s")


if __name__ == "__main__":
    main()


