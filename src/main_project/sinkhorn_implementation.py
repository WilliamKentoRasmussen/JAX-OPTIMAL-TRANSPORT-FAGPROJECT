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

model = load(name="ae_best_model", path="models")
gamma = 1e-3
stop_threshold = 1e-5



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

    return T,u,v,iter

def get_probability_y_given_x(T, index):
    p_y_given_x = T[index] / T[index].sum()
    return p_y_given_x



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

    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title(f"OT Transport: digit 0 → digit 1, gamma={gamma}, threshold={stop_threshold}")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/sinkhorn_transport_paths.png", dpi=150, bbox_inches='tight')


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
        axes[i, -1].imshow(expected_target_images[i].reshape(28, 28), cmap='gray')
        axes[i, -1].set_title("Target")
        axes[i, -1].axis('off')

    plt.tight_layout()
    plt.savefig("figures/interpolation_reconstructions.png", dpi=150)
    plt.show()