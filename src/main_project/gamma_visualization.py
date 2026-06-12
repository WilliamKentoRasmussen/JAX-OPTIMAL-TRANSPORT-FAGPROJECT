import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from main_project.utils import load
from main_project.data import getData, getDataloader
from main_project.sinkhorn import sinkhorn_log, sinkhorn_simple, cdist_euclidean
import torch
import random
from matplotlib.gridspec import GridSpec

"""
Recration figure 3: Entropy-regularized transportaiton plans for different values of gamma. 
from J. Solomon, Computational Optima Transport
"""


def gaussian_1d(x, mean, std):
    return jnp.exp(-0.5 * ((x - mean) / std) ** 2)


def plot_gamma_1d():
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
            2,
            2,
            width_ratios=[1, 4],
            height_ratios=[1, 4],
            left=0.02 + k * 0.18,
            right=0.225 + k * 0.18,
            bottom=0.15,
            top=0.85,
            wspace=0.0,
            hspace=0.0,
            figure=fig,
        )

        ax_top = fig.add_subplot(gs[0, 1])
        ax_main = fig.add_subplot(gs[1, 1])

        # transport plan
        T, _, _ = sinkhorn_simple(s_1d, d_1d, C, gamma=gamma)

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
        im = ax_main.imshow(T, origin="lower", cmap="Greys", aspect="auto")

        ax_main.set_xticks([])
        ax_main.set_yticks([])

        ax_main.set_xlabel(rf"$\gamma = {gamma}$")

    plt.savefig("figures/gamma_visualization.png", dpi=150)
    plt.show()


###############################################
###############################################
"""
Recration figure 4.3: impact of gamma on coupling between two 2-D discrete empeircal densities with the sanme n = m points
where above a small thershold transport plan is displayed as segments between alpha and beta
from Gabreil Peyré and Marco Cuturi, Computational Optima Transport
"""


def sample_circle(n, r_max=0.25):
    r = np.sqrt(np.random.uniform(0, r_max**2, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def sample_ring(n, r_min=0.3, r_max=0.5):
    r = np.sqrt(np.random.uniform(r_min**2, r_max**2, n))
    theta = np.random.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def plot_transport_plan(ax, s, d, P, threshold=0.05):
    """Draw line segments between point pairs where P[i,j] exceeds threshold * P.max()."""
    p_max = P.max()
    for i in range(len(s)):
        for j in range(len(d)):
            w = P[i, j] / p_max
            if w > threshold:
                ax.plot(
                    [s[i, 0], d[j, 0]],
                    [s[i, 1], d[j, 1]],
                    color="black",
                    alpha=float(w) * 0.9,
                    linewidth=0.8,
                )


def plot_gamma_2d():
    np.random.seed(42)
    n = 30
    s = sample_circle(n=n, r_max=0.2)  # red — inner disk (α)
    d = sample_ring(n=n)  # blue — outer ring (β)

    a = jnp.ones(n) / n
    b = jnp.ones(n) / n
    C = cdist_euclidean(jnp.array(s), jnp.array(d))

    gammas = [0.0001, 0.001, 0.01, 0.1]

    fig, axes = plt.subplots(1, len(gammas), figsize=(18, 4))

    for ax, gamma in zip(axes, gammas):
        P, u, v, iters = sinkhorn_log(a, b, C, gamma=gamma, max_iters=2000)
        P = np.array(P)

        plot_transport_plan(ax, s, d, P, threshold=0.05)

        ax.scatter(*s.T, c="red", s=25, zorder=5)
        ax.scatter(*d.T, c="blue", s=25, zorder=5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(rf"$\gamma = {gamma}$", fontsize=13)

    # Label α and β on the first panel
    axes[0].annotate(r"$\alpha$", xy=(0.8, 0.45), xycoords="axes fraction", color="red", fontsize=20, fontweight="bold")
    axes[0].annotate(
        r"$\beta$", xy=(0.05, 0.85), xycoords="axes fraction", color="blue", fontsize=20, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("figures/gamma_visualization_2d.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_gamma_2d()
