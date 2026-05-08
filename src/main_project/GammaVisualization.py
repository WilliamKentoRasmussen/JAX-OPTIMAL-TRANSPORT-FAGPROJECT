import numpy as np 
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from main_project.utils import load
from main_project.data import getData, getDataloader
from main_project.sinkhornV2 import sinkhorn_log, sinkhorn_simple, cdist_euclidean
import torch
from matplotlib.gridspec import GridSpec
"""

def gaussian_1d(x, mean, std):
    return jnp.exp(-0.5 * ((x - mean) / std) ** 2)

x = jnp.linspace(0, 1, 120)
s_1d = 0.9 * gaussian_1d(x, mean=0.35, std=0.08)

d_1d = 0.75 * gaussian_1d(x, mean=0.25, std=0.08) + 0.35 * gaussian_1d(x, mean=0.75, std=0.16)

s_1d = s_1d / s_1d.sum()
d_1d = d_1d / d_1d.sum()

C = (x[:, None] - x[None, :]) ** 2

gammas = [0.0001, 0.001, 0.01, 0.1, 1]

fig, axes = plt.subplots(1, len(gammas), figsize=(20, 5))

fig = plt.figure(figsize=(20, 5))

for k, gamma in enumerate(gammas):

    # layout for each panel
    gs = GridSpec(
        2, 2,
        width_ratios=[1, 4],
        height_ratios=[1, 4],
        left=0.02 + k * 0.195,
        right=0.18 + k * 0.195,
        bottom=0.15,
        top=0.9,
        wspace=0.0,
        hspace=0.0,
        figure=fig
    )

    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])

    # transport plan
    T, _, _ = sinkhorn_simple(
        s_1d,
        d_1d,
        C,
        gamma=gamma
    )

    # top target distribution 
    ax_top.plot(x, d_1d, color="black", linewidth=1.5)
    ax_top.set_xlim(0, 1)
    ax_top.axis("off")

    # left soruce distribution 
    ax_left.plot(s_1d, x, color="black", linewidth=1.5)
    ax_left.set_ylim(0, 1)
    ax_left.invert_xaxis()
    ax_left.axis("off")

    # transport  
    im = ax_main.imshow(
        T,
        origin="lower",
        cmap="viridis",
        aspect="auto"
    )

    ax_main.set_xticks([])
    ax_main.set_yticks([])

    ax_main.set_xlabel(rf"$\gamma = {gamma}$")


plt.suptitle("Entropy-Regularized Transport Plans", fontsize=16)
plt.savefig("figures/gamma_visualization.png", dpi=150)
plt.show()
"""

    

##################################################################
##################################################################
model = load(name="ae_best_model", path="models") 

_, test_data = getData()  
test_dataloader = getDataloader(test_data)

z_ones = []
z_zeros = []

for x, y in test_dataloader:
    recon, z = jax.vmap(model)(x.numpy().reshape(x.shape[0], -1))
    for i in range(x.shape[0]):
        if y[i] == 0:
            z_zeros.append(z[i])
        elif y[i] == 1:
            z_ones.append(z[i])

z_zeros = jnp.array(z_zeros)
z_ones = jnp.array(z_ones)

C = cdist_euclidean(jnp.array(z_zeros), jnp.array(z_ones))
# uniform weights for both distributions
"""s = jnp.exp(-z_zeros**2)
s = jnp.array(s / s.sum(axis=0))  # Normalize to sum to 1 
d = jnp.exp(-z_ones**2) 
d = jnp.array(d / d.sum(axis=0))  # Normalize to sum to 1"""
s = jnp.ones(z_zeros.shape[0]) / z_zeros.shape[0] 
d = jnp.ones(z_ones.shape[0]) / z_ones.shape[0] 

gammas = [0.001, 0.005, 0.01, 0.05, 0.1]

fig, axes = plt.subplots(1, len(gammas), figsize=(20, 4))

for ax, gamma in zip(axes, gammas):
    T, _, _, _ = sinkhorn_log(s, d, C, gamma=gamma)
    
    vmin = jnp.min(T)
    vmax = jnp.max(T)

    im = ax.imshow(T, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title(f'Gamma = {gamma}')
    ax.set_xlabel('Target Distribution (Ones)')
    ax.set_ylabel('Source Distribution (Zeros)')

fig.colorbar(im, ax=axes.ravel().tolist())

plt.suptitle("Transport Plans for Different Gamma Values")
plt.tight_layout()

plt.show()


print("Cost matrix shape:", C.shape)



#




