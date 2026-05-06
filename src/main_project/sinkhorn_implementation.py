from typing import Union
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from sinkhorn_algorithm.sinkhorn import sinkhorn as sinkhorn
from main_project.utils import load
from main_project.data import getData, getDataloader
from main_project.visualize import plot_latent_clusters

model = load(name="ae_best_model", path="models")
alpha = 1e-3
stop_threshold = 1e-5

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
start_arr  = jnp.array(np.stack([np.array(img).flatten() for img in start_data]))
target_arr = jnp.array(np.stack([np.array(img).flatten() for img in target_data]))


def encode(x):
    _, z = model(x)
    return z

latent_start  = jax.vmap(encode)(start_arr)   # [n, latent_dim]
latent_target = jax.vmap(encode)(target_arr)  # [m, latent_dim]

# Convert JAX arrays to torch tensors for the PyTorch sinkhorn
latent_start_torch  = torch.tensor(np.array(latent_start),  dtype=torch.float32)
latent_target_torch = torch.tensor(np.array(latent_target), dtype=torch.float32)
min_count = min(latent_start_torch.shape[0], latent_target_torch.shape[0])

latent_start_torch  = latent_start_torch[:min_count]
latent_target_torch = latent_target_torch[:min_count]

# Now uniform weights work fine — equal n and m
distance, approx_corr_1, approx_corr_2,u,v = sinkhorn(
    latent_start_torch, latent_target_torch,
    p=2, alpha=alpha, max_iters=1000, stop_thresh=stop_threshold, verbose=True
)


n_points = 10

fig, ax = plt.subplots(figsize=(10, 8))

for i in range(n_points):
    x_star = latent_start_torch[i:i+1]

    # Compute conditional for this point
    # diff        = x_star - latent_target_torch #distance measure
    # C_star      = (diff ** 2).sum(dim=-1) ** 0.5 
    # log_weights = (-C_star + v) / alpha # denominator but with log scaled
    # log_weights = log_weights - torch.logsumexp(log_weights, dim=0) # summation and normalization
    # p_y_given_x = torch.exp(log_weights)
    # expected_target = p_y_given_x @ latent_target_torch 

    diff   = x_star - latent_target_torch
    C_star = (diff ** 2).sum(dim=-1) ** 0.5       

    # Numerator: K(x*, y_j) * v(y_j)
    K_star   = torch.exp(-C_star / alpha)            
    v_scaled = torch.exp(v / alpha)                  
    weights  = K_star * v_scaled                   

    # Denominator: sum over all j
    p_y_given_x = weights / weights.sum()          #  P(y | x*)

    # E[y | x*]
    expected_target = p_y_given_x @ latent_target_torch   

    # Source and target coords
    x0 = x_star.squeeze().detach().numpy()
    x1 = expected_target.detach().numpy()



    # Plot interpolation path
    steps = np.linspace(0, 1, 20)
    path  = np.array([(1-t)*x0 + t*x1 for t in steps])
    ax.plot(path[:, 0], path[:, 1],
            color='gray', alpha=0.4, linewidth=1.0, zorder=1)

    # Start point (digit 0)
    ax.scatter(*x0, color='steelblue',edgecolors='black', s=60, zorder=3,
               label='Digit 0' if i == 0 else '')

    # End point (expected target)
    ax.scatter(*x1, color='coral', edgecolors='black', s=60, zorder=3,
               label='E[y|x*] digit 1' if i == 0 else '')

    # Arrow showing direction
    ax.annotate('', xy=x1, xytext=x0,
                arrowprops=dict(arrowstyle='->', color='black',
                                alpha=0.5, lw=1.0))

# Plot all digit 0 and digit 1 points in background for context
all_start  = latent_start_torch.detach().numpy()
all_target = latent_target_torch.detach().numpy()
ax.scatter(all_start[:, 0],  all_start[:, 1],
           color='steelblue', alpha=0.1, s=10, zorder=0)
ax.scatter(all_target[:, 0], all_target[:, 1],
           color='coral',     alpha=0.1, s=10, zorder=0)

ax.set_xlabel("Latent dim 1")
ax.set_ylabel("Latent dim 2")
ax.set_title(f"OT Transport: digit 0 → digit 1 (20 points), alpha = {alpha} , threshold = {stop_threshold}")
ax.legend()
plt.tight_layout()
plt.show()
plt.savefig("figures/sinkhorn_transport_paths.png")