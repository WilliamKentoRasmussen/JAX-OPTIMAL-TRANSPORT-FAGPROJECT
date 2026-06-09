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

from main_project.old.sinkhornV2 import cdist_euclidean
from main_project.utils import load
from main_project.data import getData, getDataloader
from main_project.environment import MODELS_DIM, INTERMEDIATE_FRACTIONS, MAX_POINTS, MAX_ITERATION


gamma = 1e-3
stop_threshold = 1e-5

@jax.jit
def cdist_euclidean(x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes pairwise Euclidean distances between rows of x and y.

    Args:
        x: Array of shape (N, D)
        y: Array of shape (M, D)

    Returns:
        Distance matrix of shape (N, M)
    """
    return jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))


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



def save_sinkhorn_transformation(model_name, save = True):

    model = load(name=model_name, path="models") # ae_best_model_lat2
    latent_source, latent_target, T, u, v, iter = run_sinkhorn_by_model(model)

    source_images = []
    target_images = []
    expected_target_images = []
    intermediate_images = []

    n_points = min(MAX_POINTS, len(latent_source))
    for i in range(n_points):
        x_star = latent_source[i : i + 1] #New point
        y_point = latent_target[i : i + 1]
        p_y_given_x = get_probability_y_given_x(T, i)
        expected_target = p_y_given_x @ latent_target 


        x_star_flat = x_star.reshape(x_star.shape[0], -1)
        expected_target_flat = expected_target.reshape(expected_target.shape[0], -1)

        #Transfer points to 1D arrays
        y_point = jnp.array(y_point.squeeze())
        x_star = jnp.array(x_star_flat.squeeze())
        expected_target = jnp.array(expected_target_flat.squeeze())

        y_point_img = model.decoder(y_point)
        x_star_img = model.decoder(x_star)  
        expected_target_img = model.decoder(expected_target)
        
        
        intermediate_points = []
        # Save and decode specific intermediate points
        for fraction in INTERMEDIATE_FRACTIONS:

            # Calculate intermediate latent point
            intermediate_latent = (1 - fraction) * x_star + fraction * expected_target

            decoded_img = model.decoder(intermediate_latent)  # Model returns (reconstruction, latent)

            intermediate_points.append(np.array(decoded_img))


        #Store images
        source_images.append(np.array(x_star_img))
        target_images.append(np.array(y_point_img))
        expected_target_images.append(np.array(expected_target_img))
        intermediate_images.append(intermediate_points)

    if save:
        save_dir = f"data/{model_name}"
        os.makedirs(save_dir, exist_ok=True)  # creates the directory if it doesn't exist
        np.save(f"{save_dir}/intermediate_images.npy", np.array(intermediate_images))
        np.save(f"{save_dir}/original_images.npy", np.array(source_images))
        np.save(f"{save_dir}/expected_target_images.npy", np.array(expected_target_images))
        np.save(f"{save_dir}/target_images.npy", np.array(target_images))


def save_sinkhorn_transformation_without_ae():
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

    # Used to ensure no out of bounds error
    min_count = min(source_arr.shape[0], target_arr.shape[0])

    source_arr = source_arr[:min_count]
    target_arr = target_arr[:min_count]

    C = cdist_euclidean(source_arr, target_arr)
    s = jnp.ones(source_arr.shape[0]) / source_arr.shape[0]
    d = jnp.ones(target_arr.shape[0]) / target_arr.shape[0]

    T, u, v, iter = sinkhorn_log(C=C, s=s, d=d, gamma=gamma, max_iters=MAX_ITERATION, stop_thresh=stop_threshold, verbose=True)

    





def main():
    for dim in MODELS_DIM: 
        model_name = f"ae_model_dim_{dim}"
        print("Saving sinkhorn transformation for", model_name)
        save_sinkhorn_transformation(model_name = model_name, save =True)


if __name__ == "__main__":
    main()


