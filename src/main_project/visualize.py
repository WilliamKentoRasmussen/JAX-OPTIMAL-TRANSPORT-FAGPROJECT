import torch
import matplotlib.pyplot as plt
import jax
import equinox as eqx
from main_project.model import AEv2
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from main_project.utils import load_with_hyperparams
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

from main_project.utils import load, load_with_hyperparams
from main_project.data import getData  # fixed
from main_project.train import train_classifier
from main_project.model import targetClassifier
from main_project.environment import LABELS
from main_project.data import getData
import os

labels_map = {i: str(i) for i in range(10)}


def plot_transport_images(original_images, expected_target_images, n=5, title="Transport plot"):
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for i in range(n):
        axes[0, i].imshow(original_images[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")

        axes[1, i].imshow(expected_target_images[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Source (0)", fontsize=12)
    axes[1, 0].set_ylabel("Transported (1)", fontsize=12)

    plt.suptitle(f"OT Transport: digit 0 -> digit 1 - with {title} ")
    plt.tight_layout()
    plt.show()


def plot_random_samples(training_data):  # fixed: was using global
    figure = plt.figure(figsize=(10, 4))
    cols, rows = 10, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[int(label)])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def plot_latent_clusters(training_data, model, max_points=20000, point_size=4, alpha=0.6):
    xs, ys = [], []
    for img, label in training_data:
        xs.append(img.numpy())
        ys.append(label)
    x = jnp.array(xs).reshape(len(xs), -1)
    labels = jnp.array(ys)

    n = len(x)
    idx = np.random.choice(n, size=min(max_points, n), replace=False)
    x, labels = x[idx], labels[idx]

    _, z = jax.vmap(model)(x)
    z = np.array(z)

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(z[:, 0], z[:, 1], c=labels, cmap="tab10", s=point_size, alpha=alpha)
    ax.set_xlabel("Latent dimension $z_1$")
    ax.set_ylabel("Latent dimension $z_2$")
    ax.set_title("Latent Space Clusters")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax).set_label("Digit")
    plt.tight_layout()
    plt.show()


def plot_reconstruction(training_data, model, n_examples=10):
    xs = [img.numpy() for img, _ in list(training_data)[:n_examples]]
    x = jnp.array(xs)
    recon, _ = jax.vmap(model)(x.reshape(len(xs), -1))

    x = np.array(x).reshape(-1, 28, 28)
    recon = np.array(recon).reshape(-1, 28, 28)

    fig, axes = plt.subplots(2, n_examples, figsize=(1.5 * n_examples, 3))
    for i in range(n_examples):
        axes[0, i].imshow(x[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.suptitle("Reconstruction Results")
    plt.tight_layout()
    plt.show()


def pca_visualize_for_high_dimension(
    training_data,
    model,
    max_points=20000,
    point_size=4,
    alpha=0.6,
):
    # --- Load and prepare data ---
    xs, ys = [], []
    for img, label in training_data:
        xs.append(img.numpy())
        ys.append(label)

    x = jnp.array(xs)
    labels = np.array(ys)

    x = x.reshape(x.shape[0], -1)

    # --- Subsample ---
    n = len(x)
    idx = np.random.choice(n, size=min(max_points, n), replace=False)
    x = x[idx]
    labels = labels[idx]

    # --- Encode ---
    _, z = jax.vmap(model)(x)
    z = np.array(z)

    # --- PCA ---
    pca = PCA()
    z_pca = pca.fit_transform(z)
    cumvar = np.cumsum(pca.explained_variance_ratio_)  # how much do the k principal componentents explain the variance
    n_eff = (
        np.searchsorted(cumvar, 0.95) + 1
    )  # 0 based index so we want to find how many of the principal components reach 95% of variance

    # latent space
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    scatter = ax1.scatter(
        z_pca[:, 0],
        z_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=point_size,
        alpha=alpha,
    )
    plt.colorbar(scatter, ax=ax1).set_label("Digit")
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("Latent Space — PCA Projection")
    ax1.set_aspect("equal")
    plt.tight_layout()
    plt.show()

    # Cumulative variance
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(np.arange(1, len(cumvar) + 1), cumvar, marker=".", color="steelblue")
    ax2.axhline(0.95, linestyle="--", color="red", label="95% variance")
    ax2.axvline(n_eff, linestyle="--", color="green", label=f"{n_eff} dims needed")
    ax2.set_xlabel("Number of principal components")
    ax2.set_ylabel("Cumulative variance explained")
    ax2.set_title("Scree Plot")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Latent dim:               {z.shape[1]}")
    print(f"Effective dims (95% var): {n_eff}")
    print(f"Variance in PC1+PC2:      {cumvar[1]*100:.1f}%")


def plot_training_loss(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data["epoch"], data["train_loss"], label="Training Loss", color="blue")
    plt.plot(data["epoch"], data["val_loss"], label="Validation Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt


def figure_3_dim_vs_gamma_metrics_table(summary_df):
    latent_dims = sorted(summary_df["latent_dim"].unique())
    gammas = sorted(summary_df["gamma"].unique())

    rows = []
    for dim in latent_dims:
        row = {}
        for gamma in gammas:
            mask = (summary_df["latent_dim"] == dim) & (summary_df["gamma"] == gamma)
            mmd = summary_df.loc[mask, "mmd_image"].mean()
            wasserstein = summary_df.loc[mask, "wasserstein_distance"].mean()
            conf = summary_df.loc[mask, "classifier_confidence_image"].mean()
            row[gamma] = f"MMD: {mmd:.4f}, W-Dist: {wasserstein:.4f}, Conf: {conf:.4f}"
        rows.append(row)

    figure_3_df = pd.DataFrame(data=rows, index=latent_dims)
    figure_3_df.index.name = "latent_dim"

    print(figure_3_df.to_latex())
    figure_3_df.to_csv("data/figure_3.csv")
    return figure_3_df


def plot_gamma_vs_mmd(summary_df, save_dir="figures/plots"):
    os.makedirs(save_dir, exist_ok=True)

    latent_dims = sorted(summary_df["latent_dim"].unique())
    gammas = sorted(summary_df["gamma"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for dim in latent_dims:
        means = []
        lowers = []
        uppers = []

        for gamma in gammas:
            values = summary_df.loc[
                (summary_df["latent_dim"] == dim) & (summary_df["gamma"] == gamma), "mmd_image"
            ].values

            if len(values) == 0:
                means.append(np.nan)
                lowers.append(np.nan)
                uppers.append(np.nan)
                continue

            mean = np.mean(values)

            if len(values) > 1:
                std = np.std(values, ddof=1)
                ci = 1.96 * std / np.sqrt(len(values))
            else:
                ci = 0

            means.append(mean)
            lowers.append(mean - ci)
            uppers.append(mean + ci)

        means = np.array(means)
        lowers = np.array(lowers)
        uppers = np.array(uppers)

        ax.plot(gammas, means, marker="o", linewidth=2, label=f"dim={dim}")

        ax.fill_between(gammas, lowers, uppers, alpha=0.15)

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Average MMD Image")
    ax.set_title("Average MMD vs Gamma")

    ax.set_xscale("log")  # useful if gamma=[0.01,0.1,1,10]

    ax.grid(True, alpha=0.3)
    ax.legend(title="Latent Dim", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()

    plt.savefig(f"{save_dir}/gamma_vs_mmd.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_latent_dim_vs_average_mmd(summary_df, save_dir="figures/plots"):
    os.makedirs(save_dir, exist_ok=True)

    latent_dims = sorted(summary_df["latent_dim"].unique())

    means = []
    lower = []
    upper = []

    for dim in latent_dims:
        values = summary_df.loc[summary_df["latent_dim"] == dim, "mmd_image"].values

        mean = np.mean(values)
        std = np.std(values, ddof=1)

        n = len(values)

        # 95% confidence interval
        ci = 1.96 * std / np.sqrt(n)

        means.append(mean)
        lower.append(mean - ci)
        upper.append(mean + ci)

    means = np.array(means)
    lower = np.array(lower)
    upper = np.array(upper)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(latent_dims, means, marker="o", linewidth=2, label="Mean MMD")

    ax.fill_between(latent_dims, lower, upper, alpha=0.25, label="95% CI")

    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Average MMD Image")
    ax.set_title("Average MMD vs Latent Dimension")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    plt.savefig(f"{save_dir}/latent_dim_vs_average_mmd.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_mmd_image_heatmaps_full(summary_df, save=True, save_dir="figures/heatmaps"):
    os.makedirs(save_dir, exist_ok=True)

    labels = LABELS
    latent_dims = sorted(summary_df["latent_dim"].unique())

    # Shared color scale across all dims so they are visually comparable
    global_vmin = summary_df["mmd_image"].min()
    global_vmax = summary_df["mmd_image"].max()

    matrices = {}
    for latent_dim in latent_dims:
        model_df = summary_df[summary_df["latent_dim"] == latent_dim]
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        grouped = model_df.groupby(["source_label", "target_label"])["mmd_image"].mean()
        for (src, tgt), value in grouped.items():
            matrix.loc[src, tgt] = value
        matrices[latent_dim] = matrix

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(matrix, vmin=global_vmin, vmax=global_vmax, cmap="viridis_r")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Target Label")
        ax.set_ylabel("Source Label")
        ax.set_title(f"MMD Image Heatmap (avg over γ)\nLatent Dim = {latent_dim}  |  mean = {model_df['mmd_image'].mean():.3f}")
        plt.colorbar(im, ax=ax, label="MMD Image")

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

        plt.tight_layout()
        if save:
            plt.savefig(f"{save_dir}/mmd_image_heatmap_full_dim_{latent_dim}.png", dpi=300)
        plt.close()


def plot_mmd_latent_heatmaps_full(summary_df, save=True, save_dir="figures/heatmaps"):

    os.makedirs(save_dir, exist_ok=True)

    labels = LABELS
    latent_dims = sorted(summary_df["latent_dim"].unique())

    # Shared color scale across all dims
    global_vmin = summary_df["mmd_latent"].min()
    global_vmax = summary_df["mmd_latent"].max()

    matrices = {}
    for latent_dim in latent_dims:
        # Average over all gammas (was wrongly filtered to gamma=0.1 before)
        model_df = summary_df[summary_df["latent_dim"] == latent_dim]
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        grouped = model_df.groupby(["source_label", "target_label"])["mmd_latent"].mean()
        for (src, tgt), value in grouped.items():
            matrix.loc[src, tgt] = value
        matrices[latent_dim] = matrix

        _, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(matrix, vmin=global_vmin, vmax=global_vmax, cmap="viridis_r")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Target Label")
        ax.set_ylabel("Source Label")
        ax.set_title(f"MMD Latent Heatmap (avg over γ)\nLatent Dim = {latent_dim}  |  mean = {model_df['mmd_latent'].mean():.4f}")
        plt.colorbar(im, ax=ax, label="MMD Latent")

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7)

        plt.tight_layout()
        if save:
            plt.savefig(f"{save_dir}/mmd_latent_heatmap_full_dim_{latent_dim}.png", dpi=300)
        plt.close()

def plot_mmd_heatmaps_individual(summary_df, gamma=0.1, save=True, save_dir="figures/heatmaps"):
    """Per-dim figure: MMD Image (left) + MMD Latent (right) at a fixed gamma, each with its own colorbar."""
    os.makedirs(save_dir, exist_ok=True)
    labels = LABELS
    df = summary_df[summary_df["gamma"] == gamma]
    latent_dims = sorted(df["latent_dim"].unique())

    for latent_dim in latent_dims:
        model_df = df[df["latent_dim"] == latent_dim]

        img_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        lat_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

        for (src, tgt), val in model_df.groupby(["source_label", "target_label"])["mmd_image"].mean().items():
            img_matrix.loc[src, tgt] = val
        for (src, tgt), val in model_df.groupby(["source_label", "target_label"])["mmd_latent"].mean().items():
            lat_matrix.loc[src, tgt] = val

        fig, (ax_img, ax_lat) = plt.subplots(1, 2, figsize=(16, 7))

        im1 = ax_img.imshow(img_matrix, cmap="viridis_r")
        ax_img.set_xticks(range(len(labels)))
        ax_img.set_xticklabels(labels)
        ax_img.set_yticks(range(len(labels)))
        ax_img.set_yticklabels(labels)
        ax_img.set_xlabel("Target Label")
        ax_img.set_ylabel("Source Label")
        ax_img.set_title(f"MMD Image  |  mean = {model_df['mmd_image'].mean():.3f}")
        plt.colorbar(im1, ax=ax_img, label="MMD Image")
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = img_matrix.iloc[i, j]
                if not np.isnan(val):
                    ax_img.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)

        im2 = ax_lat.imshow(lat_matrix, cmap="viridis_r")
        ax_lat.set_xticks(range(len(labels)))
        ax_lat.set_xticklabels(labels)
        ax_lat.set_yticks(range(len(labels)))
        ax_lat.set_yticklabels(labels)
        ax_lat.set_xlabel("Target Label")
        ax_lat.set_ylabel("Source Label")
        ax_lat.set_title(f"MMD Latent  |  mean = {model_df['mmd_latent'].mean():.4f}")
        plt.colorbar(im2, ax=ax_lat, label="MMD Latent")
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = lat_matrix.iloc[i, j]
                if not np.isnan(val):
                    ax_lat.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=7)

        fig.suptitle(f"MMD Heatmaps — Latent Dim = {latent_dim}  (γ = {gamma})", fontsize=14)
        plt.tight_layout()
        if save:
            plt.savefig(f"{save_dir}/mmd_heatmap_combined_dim_{latent_dim}.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_latent_space_dim(model, dim, x_sub, labels_sub, point_size=4, alpha=0.6, save=True, save_dir="figures/plots"):
    """
    Latent-space scatter for a single model/dim.
    dim == 2  → direct scatter; dim > 2 → PCA projection to 2 components.
    """
    _, z = jax.vmap(model)(x_sub)
    z = np.array(z)

    if dim <= 2:
        z2 = z
        xlabel, ylabel = r"Latent $z_1$", r"Latent $z_2$"
        method = "direct"
    else:
        pca = PCA(n_components=2)
        z2 = pca.fit_transform(z)
        var = pca.explained_variance_ratio_
        xlabel = f"PC1 ({var[0]*100:.1f}%)"
        ylabel = f"PC2 ({var[1]*100:.1f}%)"
        method = "PCA"

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(z2[:, 0], z2[:, 1], c=labels_sub, cmap="tab10", s=point_size, alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Latent Space — Dim={dim}  ({method})")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax).set_label("Digit")
    plt.tight_layout()
    if save:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/latent_space_dim_{dim}.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_sinkhorn_iterations_vs_gamma(
    csv_path="sinkhorn_iterations_and_times.csv",
    save_dir="figures/plots",
    save=True,
    alpha=0.01,
):
    df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)
    gammas = sorted(df["gamma"].unique())

    samples = [
        df.loc[df["gamma"] == g, "sinkhorn_iterations"].values
        for g in gammas
    ]
    means = np.array([np.mean(s) for s in samples])
    stds  = np.array([np.std(s, ddof=1) if len(s) > 1 else 0.0 for s in samples])

    # Walk consecutive pairs: stop at first non-significant drop
    best_idx = 0
    for i in range(len(gammas) - 1):
        # One-sided: is samples[i] > samples[i+1]? (higher gamma → fewer iterations)
        _, p = stats.ttest_ind(samples[i], samples[i + 1], alternative="greater")
        if p >= alpha:
            best_idx = i  # no significant improvement beyond this gamma
            break
    else:
        best_idx = len(gammas) - 1  # all differences significant, take largest
    best_gamma = gammas[best_idx]

    _, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        gammas, means, yerr=[np.minimum(stds, means), stds],
        fmt="o-", linewidth=2, markersize=6,
        capsize=4, capthick=1.5,
        color="steelblue", ecolor="steelblue", elinewidth=1, alpha=0.9,
        label="mean ± std"
    )
    ax.axvline(best_gamma, color="tomato", linestyle="--", linewidth=1.5,
               label=f"best γ={best_gamma:.4f} (first non-significant step, α={alpha})")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$", fontsize=13)
    ax.set_ylabel("Sinkhorn Iterations", fontsize=13)
    ax.set_title(r"Sinkhorn Iterations vs $\gamma$", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if save:
        plt.savefig(f"{save_dir}/sinkhorn_iterations_vs_gamma.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Best γ: {best_gamma:.4f}  →  {means[best_idx]:.1f} ± {stds[best_idx]:.1f} iterations")
    print(f"\nPairwise t-tests (one-sided, α={alpha}):")
    for i in range(len(gammas) - 1):
        _, p = stats.ttest_ind(samples[i], samples[i + 1], alternative="greater")
        sig = "✓ significant" if p < alpha else "✗ not significant ← stop"
        print(f"  γ={gammas[i]:.4f} vs γ={gammas[i+1]:.4f}:  p={p:.4f}  {sig}")


def plot_interpolation_paths_across_dims(
    dims=None,
    gamma=0.1,
    source_label=0,
    target_label=1,
    n=6,
    seed=42,
    fractions=[0.25, 0.5, 0.75, 1.0],
    save=True,
    save_dir="figures/plots",
):
    """
    Show interpolation paths from source → transported latent for each dim.

    Layout (rows × cols):
      rows = latent dimensions
      cols = [source] [frac=0.25] [frac=0.5] [frac=0.75] [frac=1.0=transported] [target]

    Args:
        dims: list of latent dims, e.g. [2, 8, 10, 16, 32]
        gamma: regularisation value (must exist in pickle)
        source_label / target_label: digit pair to visualise
        n: number of example columns per row
        fractions: interpolation fractions to show
    """
    if dims is None:
        dims = [2, 8, 10, 16, 32]

    pair_key = f"source_{source_label}_target_{target_label}"
    rng = np.random.default_rng(seed)

    # Load models and data for each dim
    rows_data = []
    source_imgs = None
    target_imgs = None

    for dim in dims:
        pkl_path = f"data/ae_best_model_bo_{dim}_ot_data.pkl"
        model_path = f"ae_best_model_bo_{dim}"

        if not os.path.exists(pkl_path):
            print(f"Missing {pkl_path}, skipping dim={dim}")
            continue

        # Load OT data
        with open(pkl_path, "rb") as f:
            ot_data = pickle.load(f)

        if gamma not in ot_data or pair_key not in ot_data[gamma]:
            print(f"dim={dim}: gamma={gamma} or pair '{pair_key}' not found, skipping")
            continue

        entry = ot_data[gamma][pair_key]
        source_imgs_raw = entry["source_images"]  # (N, 784)
        target_latent = entry["target"]           # (N, dim)
        n_avail = target_latent.shape[0]

        # Fix random indices across dims for alignment
        if source_imgs is None:
            idx = rng.choice(n_avail, size=min(n, n_avail), replace=False)
            source_imgs = source_imgs_raw[idx]
            target_imgs = entry["target_images"][idx]

        source_imgs_batch = source_imgs_raw[idx]
        target_latent_batch = target_latent[idx]

        # Load model and encode source to get source_latent
        model = load_with_hyperparams(model_path)
        source_imgs_jnp = jnp.asarray(source_imgs_batch)
        _, source_latent_batch = jax.vmap(model)(source_imgs_jnp)
        source_latent_batch = np.array(source_latent_batch)

        # Interpolate and decode
        dim_interps = []
        for frac in fractions:
            # Linear interpolation: frac=0 → source_latent, frac=1 → target_latent
            interp_latent = (
                (1 - frac) * source_latent_batch + frac * target_latent_batch
            )
            interp_latent_jnp = jnp.asarray(interp_latent)
            # Decode the interpolated latent: use model.decoder not the full model
            interp_images = jax.vmap(model.decoder)(interp_latent_jnp)
            dim_interps.append(np.array(interp_images))

        # Row layout: [source] + [interpolations at each fraction]
        # Note: t=1.0 is the transported latent, so no separate target column needed
        row_data = [source_imgs] + dim_interps
        rows_data.append((dim, row_data))

    if not rows_data:
        print("No data loaded — check pickle paths and gamma/pair values.")
        return

    n_rows = len(rows_data)
    n_cols = len(fractions) + 1  # source + fractions (t=1.0 is already transported)
    actual_n = min(n, source_imgs.shape[0])

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.3))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.96, bottom=0.05)

    col_labels = (
        ["Source"]
        + [f"t={f}" for f in fractions]
    )

    for r, (dim, row_imgs) in enumerate(rows_data):
        for c in range(n_cols):
            ax = axes[r, c]
            # row_imgs[c] has shape (actual_n, 784); show first example
            ax.imshow(row_imgs[c][0].reshape(28, 28), cmap="gray")
            ax.axis("off")

    # Col labels on top
    for c, label in enumerate(col_labels):
        axes[0, c].set_title(label, fontsize=10, fontweight="bold", pad=6)

    # Add dimension labels as text on the left (outside subplots)
    for r, (dim, _) in enumerate(rows_data):
        fig.text(
            0.06,
            0.92 - r * (0.85 / n_rows),
            f"Dim={dim}",
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="right",
        )

    fig.suptitle(
        f"Interpolation Paths: {source_label}→{target_label}  (γ={gamma})",
        fontsize=12,
    )
    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        fname = f"{save_dir}/interp_paths_{source_label}_to_{target_label}_gamma_{gamma}.png"
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        print(f"Saved → {fname}")

    plt.show()


if __name__ == "__main__":

    # summary_df = pd.read_csv("data/evaluation_summary.csv")

    # plot_gamma_vs_mmd(summary_df)
    # plot_latent_dim_vs_average_mmd(summary_df)
    # plot_mmd_heatmaps_individual(summary_df=summary_df, gamma=0.1, save=True)

    # training_data, _ = getData()

    # xs, ys = [], []
    # for img, label in training_data:
    #     xs.append(img.numpy())
    #     ys.append(label)
    # x_full = jnp.array(xs).reshape(len(xs), -1)
    # labels_full = np.array(ys)
    # n = len(x_full)
    # idx = np.random.choice(n, size=min(20000, n), replace=False)
    # x_sub = x_full[idx]
    # labels_sub = labels_full[idx]

    # for dim in [2, 8, 10, 16, 32]:
    #     model = load_with_hyperparams(f"ae_best_model_bo_{dim}")
    #     plot_latent_space_dim(model, dim, x_sub, labels_sub, save=True, save_dir="figures/plots")

    plot_interpolation_paths_across_dims(source_label=5,target_label=7)


        if save:
            plt.savefig(f"{save_dir}/mmd_image_heatmap_full_dim_{latent_dim}.png", dpi=300)
        plt.close()

def evaluate_KNN_lantent_quality(model_name="ae_best_model_bo_2", number_neighbors = 5):
    training_data, test_data = getData()

    x_train = jnp.array(training_data.data.numpy()).reshape(-1, 784) / 255.0 
    y_train = jnp.array(training_data.targets.numpy())

    x_test = jnp.array(test_data.data.numpy()).reshape(-1, 784) / 255.0 
    y_test = jnp.array(test_data.targets.numpy())

    model = load_with_hyperparams(name=model_name, path="models")
    _, z_train = jax.vmap(model)(x_train)
    _, z_test = jax.vmap(model)(x_test)

    z_train = np.array(z_train)
    z_test = np.array(z_test)

    neigh = KNeighborsClassifier(n_neighbors=number_neighbors)
    neigh.fit(z_train, y_train)

    score = neigh.score(z_test, y_test)
    # ADD CI !!! 
    return score 

def evaluate_test_MSE(model_name="ae_best_model_bo_2"):
    training_data, test_data = getData()

    x_test = jnp.array(test_data.data.numpy()).reshape(-1, 784) / 255.0 

    model = load_with_hyperparams(name=model_name, path="models")

    x_hat_test, _ = jax.vmap(model)(x_test)

    # Reconstruction MSE with 95% CI via standard error
    per_sample_mse = np.mean((np.array(x_test) - np.array(x_hat_test)) ** 2, axis=1)
    mse = np.mean(per_sample_mse)
    # CI need to bee added !!! 
    mse_ci = 1.96 * np.std(per_sample_mse) / np.sqrt(len(per_sample_mse))
    
    return mse

def plot_reconstruction_for_all_dim(save = False):
    _, test_data = getData()

    # pick one example per digit class (0-9)
    class_examples = {}
    for img, label in test_data:
        label = int(label)
        if label not in class_examples:
            class_examples[label] = img.numpy()
        if len(class_examples) == 10:
            break
    labels = sorted(class_examples.keys())
    xs = [class_examples[c] for c in labels]

    x = jnp.array(xs).reshape(len(xs), -1)
    x_img = np.array(x).reshape(-1, 28, 28)

    recons = []
    for dim in MODELS_DIM:
        model = load_with_hyperparams(name=f"ae_best_model_bo_{dim}", path="models")
        recon, _ = jax.vmap(model)(x)
        recons.append(np.array(recon).reshape(-1, 28, 28))

    def hide_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    n_cols = len(labels)
    n_rows = 1 + len(MODELS_DIM)
    _, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))

    for i, label in enumerate(labels):
        axes[0, i].imshow(x_img[i], cmap="gray")
        hide_ticks(axes[0, i])
        # axes[0, i].set_title(f"class {label}", fontsize=8)
    axes[0, 0].set_ylabel("Original", fontsize=10)

    for row, (dim, recon) in enumerate(zip(MODELS_DIM, recons), start=1):
        for i in range(n_cols):
            axes[row, i].imshow(recon[i], cmap="gray")
            hide_ticks(axes[row, i])
        axes[row, 0].set_ylabel(f"dim={dim}", fontsize=10)

    plt.tight_layout()
    if save: 
        plt.savefig("figures/all_dim_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    summary_df = pd.read_csv("data/evaluation_summary.csv")
    figure_3_dim_vs_gamma_metrics_table(summary_df=summary_df)
    # plot_mmd_image_heatmaps_full(summary_df=summary_df, save=False)
    # training_data, test_data = getData()
    # loss_data = pd.read_csv("training_history_ae_best_model_bo_2.csv")
    # model = load_with_hyperparams(name="ae_best_model_bo_2", path="models")
    # plot_training_loss(loss_data)
    # plot_reconstruction(training_data, model)
    # plot_latent_clusters(training_data, model)
    # pca_visualize_for_high_dimension(training_data, model)
