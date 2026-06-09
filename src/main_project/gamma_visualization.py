import numpy as np 
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from main_project.utils import load
from main_project.data import getData, getDataloader
from main_project.sinkhorn import sinkhorn_log, sinkhorn_simple, cdist_euclidean
import torch
from matplotlib.gridspec import GridSpec

"""
Recration figure 3: Entropy-regularized transportaiton plans for different values of gamma. 
from J. Solomon, Computational Optima Transport
"""

def gaussian_1d(x, mean, std):
    return jnp.exp(-0.5 * ((x - mean) / std) ** 2)

x = jnp.linspace(0, 1, 120)
s_1d = 2 * gaussian_1d(x, mean=0.35, std=0.08)

d_1d = 0.75 * gaussian_1d(x, mean=0.25, std=0.08) + 0.35 * gaussian_1d(x, mean=0.75, std=0.16)

s_1d = s_1d / s_1d.sum()
d_1d = d_1d / d_1d.sum()

C = (x[:, None] - x[None, :]) ** 2

gammas = [0.0001, 0.001, 0.01, 0.1, 1]

fig = plt.figure(figsize=(20, 5))

for k, gamma in enumerate(gammas):

    # layout for each panel
    gs = GridSpec(
        2, 2,
        width_ratios=[1, 4],
        height_ratios=[1, 4],
        left=0.02 + k * 0.18,
        right=0.225 + k * 0.18,
        bottom=0.15,
        top=0.85,
        wspace=0.0,
        hspace=0.0,
        figure=fig
    )

    ax_top = fig.add_subplot(gs[0, 1])
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

    # left source distribution
    if k == 0:
        ax_left = fig.add_subplot(gs[1, 0])
        ax_left.plot(s_1d, x, color="black", linewidth=1.5)
        ax_left.set_ylim(0, 1)
        ax_left.invert_xaxis()
        ax_left.axis("off")

    # transport  
    im = ax_main.imshow(
        T,
        origin="lower",
        cmap="Greys",
        aspect="auto"
    )

    ax_main.set_xticks([])
    ax_main.set_yticks([])

    ax_main.set_xlabel(rf"$\gamma = {gamma}$")


plt.savefig("figures/gamma_visualization.png", dpi=150)
plt.show()
