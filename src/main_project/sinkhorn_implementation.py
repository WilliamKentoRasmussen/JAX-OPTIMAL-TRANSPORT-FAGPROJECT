from typing import Union
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from main_project.sinkhornV2 import sinkhorn_simple as sinkhorn
from main_project.sinkhornV2 import sinkhorn_log as sinkhorn_log

from main_project.sinkhornV2 import cdist_euclidean
from main_project.utils import load
from main_project.data import getData, getDataloader

model = load(name="ae_best_model_lat2", path="models")
gamma = 1e-3
stop_threshold = 1e-5

<<<<<<< HEAD
=======
training_data, _ = getData()
train_loader = getDataloader(training_data)

source_data  = []
target_data = []

for x, y in train_loader:
    source_mask  = (y == 0)
    target_mask = (y == 1)
    source_data.extend(x[source_mask])
    target_data.extend(x[target_mask])

# Stack and flatten to [n, 784] as JAX arrays
source_arr  = jnp.array(np.stack([np.array(img).flatten() for img in source_data]))
target_arr = jnp.array(np.stack([np.array(img).flatten() for img in target_data]))
>>>>>>> kerem_3weeks


def get_latent_start_and_target_dist():
    training_data, _ = getData()
    train_loader = getDataloader(training_data)

    start_data  = []
    target_data = []

    for x, y in train_loader:
        start_mask  = (y == 0)
        target_mask = (y == 1)
        start_data.extend(x[start_mask])
        target_data.extend(x[target_mask])

    # Stack and flatten to [n, 784] as JAX arrays
    start_arr = jnp.asarray(
        torch.stack(start_data).reshape(len(start_data), -1).numpy()
    )

    target_arr = jnp.asarray(
        torch.stack(target_data).reshape(len(target_data), -1).numpy()
    )


<<<<<<< HEAD
    def recon(x):
        recon, z = model(x)
        return recon,z


    latent_start  = jax.vmap(recon)(start_arr)[1]   # [n, latent_dim]
    latent_target = jax.vmap(recon)(target_arr)[1]  # [m, latent_dim]


    #Used to ensure no out of bounds error
    min_count = min(latent_start.shape[0], latent_target.shape[0])

    latent_start  = latent_start[:min_count]
    latent_target = latent_target[:min_count]

    return latent_start, latent_target




def execute_sinkhorn(latent_start,latent_target):
    C = cdist_euclidean(latent_start, latent_target)
    s = jnp.ones(latent_start.shape[0]) / latent_start.shape[0]
    d = jnp.ones(latent_target.shape[0]) / latent_target.shape[0]

    # Now uniform weights work fine — equal n and m
    T,u,v,iter = sinkhorn_log(
        C = C, s=s, d=d, gamma=gamma, max_iters=1000, stop_thresh=stop_threshold, verbose=True
    )

    print(f"Sinkhorn converged in {iter} iterations with gamma={gamma} and threshold={stop_threshold}")
=======
latent_source  = jax.vmap(recon)(source_arr)[1]   # [n, latent_dim]
latent_target = jax.vmap(recon)(target_arr)[1]  # [m, latent_dim]



min_count = min(latent_source.shape[0], latent_target.shape[0])

latent_source  = latent_source[:min_count]
latent_target = latent_target[:min_count]

C = cdist_euclidean(latent_source, latent_target)
s = jnp.ones(latent_source.shape[0]) 
d = jnp.ones(latent_target.shape[0])


P,u,v,iter = sinkhorn_log(
    C = C, s=s, d=d, gamma=gamma, max_iters=1000, stop_thresh=stop_threshold, verbose=True
)
>>>>>>> kerem_3weeks

    return T,u,v,iter

def get_probability_y_given_x(T, index):
    p_y_given_x = T[index] / T[index].sum()
    return p_y_given_x


<<<<<<< HEAD

if __name__ == "__main__":

    save = True

    latent_start, latent_target = get_latent_start_and_target_dist()

    T,u,v,iter = execute_sinkhorn(latent_start, latent_target)

    n_points = 10

    fig, ax = plt.subplots(figsize=(10, 8))

    # Store for analysis
    intermediate_points = []
    intermediate_fractions = [0.25, 0.5, 0.75, 1.0]

    original_images = []
    expected_target_images = []
    intermediate_images = []

    for i in range(n_points):
        x_star = latent_start[i:i+1]
        
        p_y_given_x = get_probability_y_given_x(T,i)
        expected_target = p_y_given_x @ latent_target
        
        x_0_flat = x_star.reshape(x_star.shape[0], -1)
        x_1_flat = expected_target.reshape(expected_target.shape[0], -1)

        x0 = jnp.array(x_0_flat.squeeze())
        x1 = jnp.array(x_1_flat.squeeze())
    
        source_img = model.decoder(x0)  # This gives reconstructed image
        target_img = model.decoder(x1)
        
        original_images.append(np.array(source_img))
        expected_target_images.append(np.array(target_img))

        
        # Save intermediate points for this pair
        pair_intermediates = []
        
        # Plot interpolation path
        steps = np.linspace(0, 1, 20)
        path = np.array([(1-t)*x0 + t*x1 for t in steps])
        ax.plot(path[:, 0], path[:, 1],
                color='gray', alpha=0.4, linewidth=1.0, zorder=1)
        
        pair_images = []

        # Save and decode specific intermediate points
        for fraction in intermediate_fractions:
            # Calculate intermediate latent point
            intermediate_latent = (1 - fraction) * x0 + fraction * x1

            decoded_img = model.decoder(intermediate_latent)  # Model returns (reconstruction, latent)
            pair_images.append(np.array(decoded_img))

            # Store everything
            pair_intermediates.append({
                'fraction': fraction,
                'latent': intermediate_latent,
                'decoded_image': np.array(decoded_img),
                'source_idx': i,
                'source_point': x0,
                'target_point': x1
            })
            
            # Mark on plot
            ax.scatter(*intermediate_latent, color='green', s=30, alpha=0.6, 
                    zorder=2, marker='s')
        
        intermediate_points.append(pair_intermediates)
        intermediate_images.append(pair_images)
        
        
        # Plot source and target points
        ax.scatter(*x0, color='steelblue', edgecolors='black', s=60, zorder=3,
                label='Digit 0 (source)' if i == 0 else '')
        ax.scatter(*x1, color='coral', edgecolors='black', s=60, zorder=3,
                label='E[y|x*] digit 1 (target)' if i == 0 else '')
        
        ax.annotate('', xy=x1, xytext=x0,
                    arrowprops=dict(arrowstyle='->', color='black',
                                    alpha=0.5, lw=1.0))

    np.save(
            "data/intermediate_images.npy",
            np.array(intermediate_images)
        )
    np.save("data/original_images.npy", np.array(original_images))
    np.save("data/expected_target_images.npy", np.array(expected_target_images))




    # Plot all digit 0 and digit 1 points in background for context
    all_start  = np.array(latent_start)
    all_target = np.array(latent_target)
    ax.scatter(all_start[:, 0],  all_start[:, 1],
            color='steelblue', alpha=0.1, s=10, zorder=0)
    ax.scatter(all_target[:, 0], all_target[:, 1],
            color='coral',     alpha=0.1, s=10, zorder=0)
=======
def linear_bridge(x0, x1, steps=20, sigma=0.0, key=None):
    """
    Linear interpolation bridge between two latent points.
    
    x0, x1  : latent vectors [d]
    steps   : number of interpolation steps
    sigma   : noise level (0 = deterministic straight line)
    key     : jax random key (only needed if sigma > 0)
    
    Returns: [steps, d] trajectory
    """
    ts = jnp.linspace(0, 1, steps)
    def interp(t):
        z = (1 - t) * x0 + t * x1
        if sigma > 0.0 and key is not None:
            noise = sigma * jnp.sqrt(t * (1 - t)) * jax.random.normal(key, x0.shape)
            z = z + noise
        return z
    return jnp.stack([interp(t) for t in ts])   # [steps, d]


def plot_transport_paths(
    latent_start,
    latent_target,
    P,
    model,
    gamma,
    stop_threshold,
    n_points=10,
    intermediate_fractions=[0.25, 0.5, 0.75, 1.0],
    sigma=0,
    key=None,
):
    # ── Build paths and collect images ──────────────────────────────────────
    intermediate_points  = []
    original_images      = []
    expected_target_images = []

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(n_points):
        x_star           = latent_start[i:i+1]
        p_y_given_x      = P[i] / P[i].sum()
        expected_target  = p_y_given_x @ latent_target

        x0 = jnp.array(x_star.reshape(-1))           # [d]
        x1 = jnp.array(expected_target.reshape(-1))  # [d]

        # Decode source and expected target
        original_images.append(np.array(model.decoder(x0)))
        expected_target_images.append(np.array(model.decoder(x1)))

        # ── Full path via linear_bridge ──────────────────────────────────
        path = np.array(linear_bridge(x0, x1, steps=20, sigma=sigma, key=key))  # [20, d]
        ax.plot(path[:, 0], path[:, 1],
                color='gray', alpha=0.4, linewidth=1.0, zorder=1)

        # ── Intermediate points at requested fractions ───────────────────
        pair_intermediates = []
        for fraction in intermediate_fractions:
            # Reuse linear_bridge with a single step at t=fraction
            z_t       = (1 - fraction) * x0 + fraction * x1
            decoded   = model.decoder(z_t)
            ax.scatter(*z_t[:2], color='green', s=30, alpha=0.6, zorder=2, marker='s')
            pair_intermediates.append({
                'fraction':      fraction,
                'latent':        z_t,
                'decoded_image': np.array(decoded),
                'source_idx':    i,
                'source_point':  x0,
                'target_point':  x1,
            })
        intermediate_points.append(pair_intermediates)

        # ── Source / target markers and arrow ────────────────────────────
        ax.scatter(*x0[:2], color='steelblue', edgecolors='black', s=60, zorder=3,
                   label='Digit 0 (source)' if i == 0 else '')
        ax.scatter(*x1[:2], color='coral',     edgecolors='black', s=60, zorder=3,
                   label='E[y|x*] digit 1 (target)' if i == 0 else '')
        ax.annotate('', xy=x1[:2], xytext=x0[:2],
                    arrowprops=dict(arrowstyle='->', color='black', alpha=0.5, lw=1.0))

    # ── Background cloud of all points ──────────────────────────────────────
    all_start  = np.array(latent_start)
    all_target = np.array(latent_target)
    ax.scatter(all_start[:,  0], all_start[:,  1], color='steelblue', alpha=0.1, s=10, zorder=0)
    ax.scatter(all_target[:, 0], all_target[:, 1], color='coral',     alpha=0.1, s=10, zorder=0)
>>>>>>> kerem_3weeks

    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(f"OT Transport: digit 0 → digit 1, gamma={gamma}, threshold={stop_threshold}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/sinkhorn_transport_paths.png", dpi=150, bbox_inches='tight')
<<<<<<< HEAD


    # ---- VISUALIZE INTERMEDIATE RECONSTRUCTIONS ----
    n_display = min(n_points, 5)  # how many rows
    n_cols = 2 + len(intermediate_fractions)  # source + intermediates + target

    fig, axes = plt.subplots(n_display, n_cols, figsize=(2*n_cols, 2*n_display))

    for i in range(n_display):
        # Source image
        axes[i, 0].imshow(original_images[i].reshape(28, 28), cmap='gray')
        axes[i, 0].set_title("Source")
        axes[i, 0].axis('off')
        
        # Intermediate images
        for j, inter in enumerate(intermediate_points[i]):
            axes[i, j+1].imshow(inter['decoded_image'].reshape(28, 28), cmap='gray')
            axes[i, j+1].set_title(f"t={inter['fraction']}")
            axes[i, j+1].axis('off')
        
        # Target image
=======
    plt.show()

    # ── Reconstruction grid ──────────────────────────────────────────────────
    n_display = min(n_points, 5)
    n_cols    = 2 + len(intermediate_fractions)   # source + intermediates + target

    fig, axes = plt.subplots(n_display, n_cols, figsize=(2 * n_cols, 2 * n_display))

    for i in range(n_display):
        # Source
        axes[i, 0].imshow(original_images[i].reshape(28, 28), cmap='gray')
        axes[i, 0].set_title("Source")
        axes[i, 0].axis('off')

        # Intermediates
        for j, inter in enumerate(intermediate_points[i]):
            axes[i, j + 1].imshow(inter['decoded_image'].reshape(28, 28), cmap='gray')
            axes[i, j + 1].set_title(f"t={inter['fraction']}")
            axes[i, j + 1].axis('off')

        # Target
>>>>>>> kerem_3weeks
        axes[i, -1].imshow(expected_target_images[i].reshape(28, 28), cmap='gray')
        axes[i, -1].set_title("Target")
        axes[i, -1].axis('off')

    plt.tight_layout()
    plt.savefig("figures/interpolation_reconstructions.png", dpi=150)
<<<<<<< HEAD
    plt.show()
=======
    plt.show()

    return intermediate_points, original_images, expected_target_images




if __name__ == "__main__":
    intermediates, src_imgs, tgt_imgs = plot_transport_paths(
    latent_source, latent_target, P, model,
    gamma=gamma, stop_threshold=stop_threshold,
    n_points=10, sigma=0.0,
)
>>>>>>> kerem_3weeks
