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

from main_project.sinkhorn import run_sinkhorn_by_model, get_probability_y_given_x, cdist_euclidean, sinkhorn_log

from main_project.schrodinger_bridge import SchrodingerBridge, density_weights

gamma = 1e-3
stop_threshold = 1e-5


def save_sinkhorn_transformation(model_name, save = True):
    latent_dim = int(re.search(r'_(\d+)', model_name).group(1))

    model = load(name=model_name, path="models", latent_dim=latent_dim) # ae_best_model_lat2

    t0 = time.perf_counter()
    latent_source, latent_target, T, u, v, iter = run_sinkhorn_by_model(model)
    t1 = time.perf_counter()

    source_images = []
    target_images = []
    expected_target_images = []
    intermediate_images = []

    n_points = min(MAX_POINTS, len(latent_source))

    #Decoding loop
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

    t2 = time.perf_counter()

    print(f"  sinkhorn: {t1-t0:.2f}s  |  decoding: {t2-t1:.2f}s")

    if save:
        save_dir = f"data/{model_name}"
        os.makedirs(save_dir, exist_ok=True)  # creates the directory if it doesn't exist
        np.save(f"{save_dir}/intermediate_images.npy", np.array(intermediate_images))
        np.save(f"{save_dir}/original_images.npy", np.array(source_images))
        np.save(f"{save_dir}/expected_target_images.npy", np.array(expected_target_images))
        np.save(f"{save_dir}/target_images.npy", np.array(target_images))


def save_sinkhorn_transformation_without_ae(save=True):
    training_data, _ = getData()
    train_loader = getDataloader(training_data)

    start_data = []
    target_data = []

    for x, y in train_loader:
        start_mask = y == 0
        target_mask = y == 1
        start_data.extend(x[start_mask])
        target_data.extend(x[target_mask])

    source_arr = jnp.asarray(torch.stack(start_data).reshape(len(start_data), -1).numpy())
    target_arr = jnp.asarray(torch.stack(target_data).reshape(len(target_data), -1).numpy())

    min_count = min(source_arr.shape[0], target_arr.shape[0])
    source_arr = source_arr[:min_count]
    target_arr = target_arr[:min_count]

    C = cdist_euclidean(source_arr, target_arr)
    s = jnp.ones(source_arr.shape[0]) / source_arr.shape[0]
    d = jnp.ones(target_arr.shape[0]) / target_arr.shape[0]

    T, u, v, iter = sinkhorn_log(C=C, s=s, d=d, gamma=gamma, max_iters=MAX_ITERATION, stop_thresh=stop_threshold, verbose=True)

    source_images = []
    target_images = []
    expected_target_images = []
    intermediate_images = []

    n_points = min(MAX_POINTS, len(source_arr))
    for i in range(n_points):
        p_y_given_x = get_probability_y_given_x(T, i)

        # All three are now consistently 1D vectors
        x_star = jnp.array(source_arr[i].squeeze())
        y_point = jnp.array(target_arr[i].squeeze())
        expected_target = jnp.array((p_y_given_x @ target_arr).squeeze())  # (784,)

        intermediate_points = []
        for fraction in INTERMEDIATE_FRACTIONS:
            decoded_img = (1 - fraction) * x_star + fraction * expected_target
            intermediate_points.append(np.array(decoded_img))

        source_images.append(np.array(x_star))
        target_images.append(np.array(y_point))
        expected_target_images.append(np.array(expected_target))
        intermediate_images.append(intermediate_points)

    if save:
        save_dir = f"data/no_ae"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/intermediate_images.npy", np.array(intermediate_images))
        np.save(f"{save_dir}/original_images.npy", np.array(source_images))
        np.save(f"{save_dir}/expected_target_images.npy", np.array(expected_target_images))
        np.save(f"{save_dir}/target_images.npy", np.array(target_images))




def save_sb_transformation(model_name, save=True):
    latent_dim = int(re.search(r'_(\d+)', model_name).group(1))
    model = load(name=model_name, path="models", model=AEv2(key=jr.PRNGKey(0), latent_dim=latent_dim))

    # --- Load data ---
    training_data, _ = getData()
    train_loader = getDataloader(training_data)

    source_data, target_data = [], []
    for x, y in train_loader:
        source_data.extend(x[y == 0])
        target_data.extend(x[y == 1])

    source_arr = jnp.asarray(torch.stack(source_data).reshape(len(source_data), -1).numpy())
    target_arr = jnp.asarray(torch.stack(target_data).reshape(len(target_data), -1).numpy())

    def recon(x):
        r, z = model(x)
        return r, z

    latent_source = jax.vmap(recon)(source_arr)[1]
    latent_target = jax.vmap(recon)(target_arr)[1]

    min_count = min(latent_source.shape[0], latent_target.shape[0])
    latent_source = latent_source[:min_count]
    latent_target = latent_target[:min_count]

    weights_x = jnp.array(density_weights(np.array(latent_source), k=5))
    weights_y = jnp.array(density_weights(np.array(latent_target), k=5))

    # --- IPF (equivalent to sinkhorn timing block) ---
    t0 = time.perf_counter()
    bridge = SchrodingerBridge(n_steps=20, sigma=0.5, max_iter=100, tol=1e-6)
    bridge.fit(latent_source=latent_source, latent_target=latent_target, weights_x=weights_x, weights_y=weights_y)
    P = bridge.get_transport_plan()
    jax.block_until_ready(P)
    t1 = time.perf_counter()

    # --- Decoding (equivalent to decoding loop timing block) ---
    result = bridge.sample_trajectories(P=P, model=model, n_samples=MAX_POINTS)
    t2 = time.perf_counter()

    print(f"  IPF fit: {t1-t0:.2f}s  |  trajectory sampling: {t2-t1:.2f}s")

    if save:
        save_dir = f"data/sb_{model_name}"
        os.makedirs(save_dir, exist_ok=True)
        np.save(f"{save_dir}/decoded.npy",      result["decoded"])
        np.save(f"{save_dir}/trajectories.npy", result["trajectories"])


def warmup_jax(n, dim):
    dummy_x = jnp.ones((n, dim))
    dummy_y = jnp.ones((n, dim))
    s = jnp.ones(n) / n
    d = jnp.ones(n) / n
    C = cdist_euclidean(dummy_x, dummy_y)
    T, *_ = sinkhorn_log(s, d, C, gamma=gamma, max_iters=2)
    jax.block_until_ready(T)



def main():
    running_times_sinkhorn = {}
    running_times_sb = {}

    #warmup_jax(n=MAX_POINTS, dim=MODELS_DIM[0])

    for dim in MODELS_DIM:
        model_name = f"ae_model_dim_{dim}"
        print(f"\n--- dim={dim} ---")

        start = time.perf_counter()
        save_sinkhorn_transformation(model_name=model_name, save=True)
        running_times_sinkhorn[dim] = time.perf_counter() - start

        start = time.perf_counter()
        save_sb_transformation(model_name=model_name, save=True)
        running_times_sb[dim] = time.perf_counter() - start

    print("\n--- Summary ---")
    print(f"{'dim':>6}  {'sinkhorn':>12}  {'schr. bridge':>12}")
    for dim in MODELS_DIM:
        print(f"{dim:>6}  {running_times_sinkhorn[dim]:>10.2f}s  {running_times_sb[dim]:>10.2f}s")


if __name__ == "__main__":
    main()
    