import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def minmax(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def select_best_gamma_pareto(df, gammas, metric="mmd_image", w_mmd=0.5, w_iter=0.5):
    """
    Select optimal gamma via weighted composite of normalized MMD and iter_count.
    Returns best_gamma, mmd_per_gamma, iter_per_gamma, composite_per_gamma.
    """
    mmd_per_gamma = np.array([df.loc[df["gamma"] == g, metric].mean() for g in gammas])
    iter_per_gamma = np.array([df.loc[df["gamma"] == g, "iter_count"].mean() for g in gammas])

    composite = w_mmd * minmax(mmd_per_gamma) + w_iter * minmax(iter_per_gamma)
    best_idx = int(np.argmin(composite))
    best_gamma = gammas[best_idx]

    print(f"Selected γ={best_gamma:.4g}  "
          f"(MMD={mmd_per_gamma[best_idx]:.4f}, "
          f"iters={iter_per_gamma[best_idx]:.0f}, "
          f"composite={composite[best_idx]:.4f})")

    return best_gamma, mmd_per_gamma, iter_per_gamma, composite

def plot_pareto_gamma(
    df=None,
    csv_path="data/evaluation_summary.csv",
    save_dir="figures/plots",
    save=True,
    metric="mmd_image",
    w_mmd=0.5,
    w_iter=0.5,
):
    """
    Pareto scatter of mean MMD vs mean iter_count per gamma,
    with the weighted-composite-optimal gamma highlighted.
    """
    if df is None:
        df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)
    gammas = sorted(df["gamma"].unique())

    best_gamma, mmd_per_gamma, iter_per_gamma, composite = select_best_gamma_pareto(
        df, gammas, metric=metric, w_mmd=w_mmd, w_iter=w_iter
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.9, len(gammas)))

    for i, (g, mmd, iters, c) in enumerate(
        zip(gammas, mmd_per_gamma, iter_per_gamma, cmap)
    ):
        is_best = g == best_gamma
        ax.scatter(
            iters, mmd,
            color=c,
            s=180 if is_best else 90,
            zorder=4 if is_best else 3,
            edgecolors="black" if is_best else "none",
            linewidths=1.5,
            marker="*" if is_best else "o",
        )
        ax.annotate(
            f"γ={g:.4g}",
            xy=(iters, mmd),
            xytext=(6, 4), textcoords="offset points",
            fontsize=9,
            fontweight="bold" if is_best else "normal",
        )

    # add a single legend entry for the selected gamma
    ax.scatter([], [], marker="*", color="black", s=180,
               label=f"selected γ={best_gamma:.4g} (w_mmd={w_mmd}, w_iter={w_iter})")
    ax.scatter([], [], marker="o", color="gray", s=90, label="other γ values")

    ax.set_xlabel("Mean Sinkhorn iterations", fontsize=13)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=13)
    ax.set_title(r"MMD vs Sinkhorn iterations across $\gamma$ (Pareto view)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    fname = "pareto_gamma.png"
    if save:
        plt.savefig(f"{save_dir}/{fname}", dpi=300, bbox_inches="tight")
    plt.show()

    return best_gamma

df = pd.read_csv("data/evaluation_summary.csv")
best_gamma = plot_pareto_gamma(df=df, metric="mmd_image", w_mmd=0.9, w_iter=0.1)
print(best_gamma)