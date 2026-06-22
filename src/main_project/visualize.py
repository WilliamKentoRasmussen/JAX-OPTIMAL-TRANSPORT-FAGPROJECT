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
from main_project.optimal_transport import get_trajectory
from main_project.utils import load_with_hyperparams
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier

from main_project.utils import load_with_hyperparams
from main_project.data import getData  # fixed
from main_project.environment import LABELS, MAX_POINTS, MODELS_DIM, OPTIMAL_GAMMA
from main_project.data import getData
import os

labels_map = {i: str(i) for i in range(10)}
# style.py
import matplotlib.pyplot as plt

def apply():
    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "xtick.labelsize":  10,
        "ytick.labelsize":  10,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.linestyle":   "--",
        "grid.alpha":       0.4,
        "figure.dpi":       150,
    })

apply()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
 
 
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

def plot_barycentric_geometry_vs_gamma_dim2(
    training_data,
    model,
    source_label,
    target_label,
    save_dir="figures/plots",
    save=True,
    point_size=12,
    alpha=0.6,
    n_highlight=35,         # how many source points to draw transport lines for
    highlight_size=80,      # marker size for highlighted points
    background_frac=1.0,    # fraction of background cloud points to plot (0-1)
    clip_quantile=0.1,     # trim this fraction from each tail when setting axis limits
    mean_marker_size=100,   # marker size for the target distribution mean star
    title_fontsize=16,
    label_fontsize=16,
    legend_fontsize=16,
    suptitle_fontsize=16,
    tick_fontsize=11
):
    model_name = "ae_best_model_bo_2"
    pickle_filename = f"data/{model_name}_ot_data.pkl"
    with open(pickle_filename, "rb") as f:
        model_transport_data = pickle.load(f)

    os.makedirs(save_dir, exist_ok=True)
    n_points = MAX_POINTS
    label = f"source_{source_label}_target_{target_label}"
    gammas = sorted(model_transport_data.keys())

    def encode_label(target_class, n_points=n_points, seed=34):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(training_data))
        xs = []
        for i in indices:
            img, lbl = training_data[i]
            if lbl == target_class:
                xs.append(img.numpy())
            if len(xs) >= n_points:
                break
        x = jnp.array(xs).reshape(len(xs), -1)
        _, z = jax.vmap(model)(x)
        return np.array(z)

    latent_source = encode_label(source_label)
    latent_target = encode_label(target_label)

    # --- mean of the full target distribution (computed once, fixed across gammas) ---
    target_mean = latent_target.mean(axis=0)

    # --- subsample background clouds for clarity ---
    rng = np.random.default_rng(0)
    if background_frac < 1.0:
        n_src_show = max(1, int(latent_source.shape[0] * background_frac))
        n_tgt_show = max(1, int(latent_target.shape[0] * background_frac))
        src_bg_idx = rng.choice(latent_source.shape[0], n_src_show, replace=False)
        tgt_bg_idx = rng.choice(latent_target.shape[0], n_tgt_show, replace=False)
    else:
        src_bg_idx = np.arange(latent_source.shape[0])
        tgt_bg_idx = np.arange(latent_target.shape[0])

    src_bg = latent_source[src_bg_idx]
    tgt_bg = latent_target[tgt_bg_idx]

    # --- compute robust axis limits (trim outliers) ---
    all_pts = np.vstack([latent_source, latent_target])
    lo = np.quantile(all_pts, clip_quantile, axis=0)
    hi = np.quantile(all_pts, 1 - clip_quantile, axis=0)
    pad = 0.1 * (hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])

    n_cols = len(gammas)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), sharex=True, sharey=True)
    if n_cols == 1:
        axes = [axes]

    # define colors/labels once so every subplot is consistent and we can
    # build a single shared legend afterwards
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
                    markersize=8, alpha=0.5, label=f"Digit {source_label} (source distribution)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:orange",
                    markersize=8, alpha=0.5, label=f"Digit {target_label} (target distribution)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
                    markeredgecolor="black", markersize=10, label=f"Source Points to be Transported"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="tab:green",
                    markeredgecolor="black", markersize=10, label="Transported Points in Target "),
        plt.Line2D([0], [0], color="gray", linewidth=1.2, label="Transport map"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                    markeredgecolor="black", markersize=16, label=f"Mean of Digit {target_label} distribution"),
    ]

    for ax, gamma in zip(axes, gammas):
        P = model_transport_data[gamma][label]["P"]
        n_tgt_P = P.shape[1]
        latent_target_P = latent_target[:n_tgt_P]

        p_y_given_x = P / P.sum(axis=1, keepdims=True)
        expected_target = p_y_given_x @ latent_target_P  # (n_src, 2)

        # --- background clouds (faint, subsampled) ---
        ax.scatter(
            src_bg[:, 0], src_bg[:, 1],
            c="tab:blue", marker="o", s=point_size, alpha=alpha * 0.25,
            edgecolors="none",
        )
        ax.scatter(
            tgt_bg[:, 0], tgt_bg[:, 1],
            c="tab:orange", marker="o", s=point_size, alpha=alpha * 0.25,
            edgecolors="none",
        )

        # --- highlighted subset + connecting lines ---
        n_hl = min(n_highlight, latent_source.shape[0])
        idx = np.arange(n_hl)

        src_hl = latent_source[idx]
        tgt_hl = expected_target[idx]

        for i in range(n_hl):
            ax.plot(
                [src_hl[i, 0], tgt_hl[i, 0]],
                [src_hl[i, 1], tgt_hl[i, 1]],
                color="gray", linewidth=0.8, alpha=0.8, zorder=2,
            )

        ax.scatter(
            src_hl[:, 0], src_hl[:, 1],
            c="tab:blue", marker="o", s=highlight_size,
            edgecolors="black", linewidths=0.8, zorder=3,
        )
        ax.scatter(
            tgt_hl[:, 0], tgt_hl[:, 1],
            c="tab:green", marker="^", s=highlight_size,
            edgecolors="black", linewidths=0.8, zorder=3,
        )

        # --- mean of the target distribution, as a star, on top of everything ---
        ax.scatter(
            target_mean[0], target_mean[1],
            c="red", marker="*", s=mean_marker_size,
            edgecolors="black", linewidths=1.0, zorder=4,
        )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"γ={gamma:.4f}", fontsize=title_fontsize)
        ax.set_xlabel("Latent dim 1", fontsize=label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    axes[0].set_ylabel("Latent dim 2", fontsize=label_fontsize)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(legend_handles) // 2 + 1,
        frameon=False,
        fontsize=legend_fontsize,
    )

    fig.suptitle(
        f"OT Transport: digit {source_label} → digit {target_label} "
        f"({n_highlight} points), across γ",
        y=1.15,
        fontsize=suptitle_fontsize,
    )

    if save:
        plt.savefig(
            f"{save_dir}/plot_barycentric_geometry_vs_gamma_dim2.png",
            dpi=300, bbox_inches="tight",
        )

    return fig



def plot_single_point_transport(
    training_data,
    model,
    source_label,
    target_label,
    gamma,
    point_index=0,          # which source point (by index, after encoding) to highlight
    save_dir="figures/plots",
    save=True,
    point_size=12,
    alpha=0.6,
    highlight_size=80,
    background_frac=1.0,
    clip_quantile=0.1,
    title_fontsize=14,
    label_fontsize=14,
    legend_fontsize=11,
    tick_fontsize=11,
    figsize=(6, 6),
):
    """
    Visualize the optimal-transport mapping for a single source point:
    background clouds for the source and target distributions, one
    highlighted source point, and its barycentric-projected destination
    E[y|x*], connected by a single transport line.
    """
    model_name = "ae_best_model_bo_2"
    pickle_filename = f"data/{model_name}_ot_data.pkl"
    with open(pickle_filename, "rb") as f:
        model_transport_data = pickle.load(f)
 
    os.makedirs(save_dir, exist_ok=True)
    n_points = MAX_POINTS
    label = f"source_{source_label}_target_{target_label}"
 
    if gamma not in model_transport_data:
        raise ValueError(
            f"gamma={gamma} not found. Available: {sorted(model_transport_data.keys())}"
        )
 
    def encode_label(target_class, n_points=n_points, seed=34):
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(training_data))
        xs = []
        for i in indices:
            img, lbl = training_data[i]
            if lbl == target_class:
                xs.append(img.numpy())
            if len(xs) >= n_points:
                break
        x = jnp.array(xs).reshape(len(xs), -1)
        _, z = jax.vmap(model)(x)
        return np.array(z)
 
    latent_source = encode_label(source_label)
    latent_target = encode_label(target_label)
 
    # subsample background clouds for clarity
    rng = np.random.default_rng(0)
    if background_frac < 1.0:
        n_src_show = max(1, int(latent_source.shape[0] * background_frac))
        n_tgt_show = max(1, int(latent_target.shape[0] * background_frac))
        src_bg_idx = rng.choice(latent_source.shape[0], n_src_show, replace=False)
        tgt_bg_idx = rng.choice(latent_target.shape[0], n_tgt_show, replace=False)
    else:
        src_bg_idx = np.arange(latent_source.shape[0])
        tgt_bg_idx = np.arange(latent_target.shape[0])
 
    src_bg = latent_source[src_bg_idx]
    tgt_bg = latent_target[tgt_bg_idx]
 
    # robust axis limits (trim outliers)
    all_pts = np.vstack([latent_source, latent_target])
    lo = np.quantile(all_pts, clip_quantile, axis=0)
    hi = np.quantile(all_pts, 1 - clip_quantile, axis=0)
    pad = 0.1 * (hi - lo)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])
 
    P = model_transport_data[gamma][label]["P"]
    n_tgt_P = P.shape[1]
    latent_target_P = latent_target[:n_tgt_P]
 
    p_y_given_x = P / P.sum(axis=1, keepdims=True)
    expected_target = p_y_given_x @ latent_target_P  # (n_src, 2)
 
    if point_index >= latent_source.shape[0]:
        raise ValueError(
            f"point_index={point_index} out of range (0 to {latent_source.shape[0] - 1})"
        )
 
    src_pt = latent_source[point_index]
    tgt_pt = expected_target[point_index]
 
    fig, ax = plt.subplots(figsize=figsize)
 
    # background clouds (faint, subsampled)
    ax.scatter(
        src_bg[:, 0], src_bg[:, 1],
        c="tab:blue", marker="o", s=point_size, alpha=alpha * 0.25,
        edgecolors="none",
    )
    ax.scatter(
        tgt_bg[:, 0], tgt_bg[:, 1],
        c="tab:orange", marker="o", s=point_size, alpha=alpha * 0.25,
        edgecolors="none",
    )
 
    # single transport line
    ax.plot(
        [src_pt[0], tgt_pt[0]],
        [src_pt[1], tgt_pt[1]],
        color="gray", linewidth=1.0, alpha=0.9, zorder=2,
    )
 
    # the one highlighted source point and its barycentric projection
    ax.scatter(
        src_pt[0], src_pt[1],
        c="tab:blue", marker="o", s=highlight_size,
        edgecolors="black", linewidths=1.0, zorder=3,
    )
    ax.scatter(
        tgt_pt[0], tgt_pt[1],
        c="tab:orange", marker="o", s=highlight_size,
        edgecolors="black", linewidths=1.0, zorder=3,
    )
 
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
 
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")
    fig.tight_layout(pad=0)
 
    if save:
        fname = f"transport_single_point_{source_label}_to_{target_label}_gamma_{gamma:.4f}.png"
        plt.savefig(
            f"{save_dir}/{fname}", dpi=300, bbox_inches="tight",
            pad_inches=0, facecolor="white",
        )
 
    return fig
 
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




def fig_2_dim_vs_gamma_metrics_table(summary_df):
    latent_dims = sorted(summary_df["latent_dim"].unique())
    
    evaluation_metrics =["entropy", "wasserstein_distance_latent", "mmd_latent", "mmd_image", "classifier_confidence_image", ]

    rows = []
    for dim in latent_dims:
        row = []
        for metric in evaluation_metrics:

            mask = (summary_df["latent_dim"] == dim) & (summary_df["gamma"] == OPTIMAL_GAMMA)
            metric_value = summary_df.loc[mask, metric].mean()
            row.append(f"{metric_value:.4f}")
        rows.append(row)

    figure_3_df = pd.DataFrame(data=rows, columns= ["Entropy Of P", "Latent Wasserstein Distance","Latent MMD" , "Image MMD","Image Classifier Probability"],  index=latent_dims)
    figure_3_df.index.name = "Latent Dimension"

    print(figure_3_df.to_latex())
    figure_3_df.to_csv("figures/rapport/fig_2_dim_vs_gamma_metrics_table.csv")
    return figure_3_df


def fig_4_plot_gamma_vs_mmd(summary_df, save_dir="figures/rapport"):
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

    plt.savefig(f"{save_dir}/fig_4_gamma_vs_mmd.png", dpi=300, bbox_inches="tight")

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

        
def fig_3_plot_mmd_heatmaps_individual(summary_df, save=True, save_dir="figures/rapport"):
    """Per-dim figure: MMD Latent (left) + Wasserstein Latent (right) at a fixed gamma,
    full 10×10 asymmetric heatmap of all 90 directed pairs."""
    os.makedirs(save_dir, exist_ok=True)
    labels = LABELS  # e.g. [0,1,...,9]
    df = summary_df[summary_df["gamma"] == OPTIMAL_GAMMA]
    latent_dims = sorted(df["latent_dim"].unique())

    for latent_dim in latent_dims:
        model_df = df[df["latent_dim"] == latent_dim]

        mmd_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        wass_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

        mmd_means = model_df.groupby(["source_label", "target_label"])["mmd_latent"].mean()
        wass_means = model_df.groupby(["source_label", "target_label"])["wasserstein_distance_latent"].mean()

        for (src, tgt), val in mmd_means.items():
            mmd_matrix.at[src, tgt] = val  # .at avoids label/loc ambiguity

        for (src, tgt), val in wass_means.items():
            wass_matrix.at[src, tgt] = val

        fig, (ax_mmd, ax_wass) = plt.subplots(1, 2, figsize=(18, 8))

        # --- MMD Latent ---
        mmd_vals = mmd_matrix.values.astype(float)
        im1 = ax_mmd.imshow(mmd_vals, cmap="viridis_r")
        ax_mmd.set_xticks(range(len(labels)))
        ax_mmd.set_xticklabels(labels)
        ax_mmd.set_yticks(range(len(labels)))
        ax_mmd.set_yticklabels(labels)
        ax_mmd.set_xlabel("Target Label")
        ax_mmd.set_ylabel("Source Label")
        ax_mmd.set_title(f"MMD Latent  |  mean = {np.nanmean(mmd_vals):.4f}")
        plt.colorbar(im1, ax=ax_mmd, label="MMD Latent")

        mmd_mean = np.nanmean(mmd_vals)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = mmd_matrix.iloc[i, j]
                if not np.isnan(val):
                    ax_mmd.text(j, i, f"{val:.4f}", ha="center", va="center",
                                fontsize=6, color="white" if val > mmd_mean else "black")

        # --- Wasserstein Latent ---
        wass_vals = wass_matrix.values.astype(float)
        im2 = ax_wass.imshow(wass_vals, cmap="viridis_r")
        ax_wass.set_xticks(range(len(labels)))
        ax_wass.set_xticklabels(labels)
        ax_wass.set_yticks(range(len(labels)))
        ax_wass.set_yticklabels(labels)
        ax_wass.set_xlabel("Target Label")
        ax_wass.set_ylabel("Source Label")
        ax_wass.set_title(f"Wasserstein Latent  |  mean = {np.nanmean(wass_vals):.4f}")
        plt.colorbar(im2, ax=ax_wass, label="Wasserstein Distance (Latent)")

        wass_mean = np.nanmean(wass_vals)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = wass_matrix.iloc[i, j]
                if not np.isnan(val):
                    ax_wass.text(j, i, f"{val:.4f}", ha="center", va="center",
                                 fontsize=6, color="white" if val > wass_mean else "black")

        fig.suptitle(
            f"Latent Space Distance Heatmaps — Latent Dim = {latent_dim}  (γ = {OPTIMAL_GAMMA})",
            fontsize=14
        )
        plt.tight_layout()

        if save:
            plt.savefig(
                f"{save_dir}/fig_3_latent_heatmaps_dim_{latent_dim}.png",
                dpi=300, bbox_inches="tight"
            )
        plt.close()
def plot_mmd_vs_dim(
    df: str,
    gamma_value: 0.001,
    save_path: str = "figures/plots/mmd_vs_dim_gamma.png",
    mode: str = "averaged",  # "averaged" or "per_pair"
):
 
    # filter to the requested gamma (use isclose to avoid float-equality issues)
    subset = df[abs(df["gamma"] - gamma_value) < 1e-12].copy()
    if subset.empty:
        raise ValueError(f"No rows found for gamma={gamma_value} in {csv_path}")
 
    fig, ax = plt.subplots(figsize=(6, 4.5))
 
    if mode == "averaged":
        # average mmd_latent / mmd_image across all source/target pairs per dim
        grouped = (
            subset.groupby("latent_dim")[["mmd_latent", "mmd_image"]]
            .mean()
            .sort_index()
        )
 
        ax.plot(
            grouped.index,
            grouped["mmd_latent"],
            marker="o",
            color="tab:blue",
            label="Latent MMD",
        )
        ax.plot(
            grouped.index,
            grouped["mmd_image"],
            marker="s",
            color="tab:orange",
            label="Image MMD",
        )
 
    elif mode == "per_pair":
        # one line per (source_label, target_label) pair, two metrics each
        pairs = subset[["source_label", "target_label"]].drop_duplicates()
        cmap_latent = plt.cm.Blues
        cmap_image = plt.cm.Oranges
        n = len(pairs)
 
        for i, (_, row) in enumerate(pairs.iterrows()):
            s, t = row["source_label"], row["target_label"]
            pair_data = subset[
                (subset["source_label"] == s) & (subset["target_label"] == t)
            ].sort_values("latent_dim")
 
            shade = 0.4 + 0.5 * (i / max(n - 1, 1))  # vary shade across pairs
            ax.plot(
                pair_data["latent_dim"],
                pair_data["mmd_latent"],
                marker="o",
                color=cmap_latent(shade),
                label=f"Latent MMD ({s}→{t})",
            )
            ax.plot(
                pair_data["latent_dim"],
                pair_data["mmd_image"],
                marker="s",
                linestyle="--",
                color=cmap_image(shade),
                label=f"Image MMD ({s}→{t})",
            )
    else:
        raise ValueError("mode must be 'averaged' or 'per_pair'")
 
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("MMD")
    ax.set_title(f"MMD vs. latent dimension (γ = {gamma_value})")
    ax.set_xticks(sorted(subset["latent_dim"].unique()))
    ax.legend(loc="best", fontsize=8 if mode == "per_pair" else 10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
 
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


   
def fig_3_plot_mmd_heatmaps_individual_selfmade(summary_df, save=True, save_dir="figures/rapport"):
    """Per-dim figure: MMD Latent (left) + Wasserstein Latent (right) at a fixed gamma,
    full 10×10 asymmetric heatmap of all 90 directed pairs."""
    os.makedirs(save_dir, exist_ok=True)
    labels = LABELS  # e.g. [0,1,...,9]
    df = summary_df[summary_df["gamma"] == OPTIMAL_GAMMA]
    latent_dims = sorted(df["latent_dim"].unique())

    for latent_dim in latent_dims:
        model_df = df[df["latent_dim"] == latent_dim]

        mmd_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
        wass_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

        mmd_means = model_df.groupby(["source_label", "target_label"])["mmd_latent"].mean()
        wass_means = model_df.groupby(["source_label", "target_label"])["wasserstein_distance_latent"].mean()

        for (src, tgt), val in mmd_means.items():
            mmd_matrix.at[src, tgt] = val  # .at avoids label/loc ambiguity

        for (src, tgt), val in wass_means.items():
            wass_matrix.at[src, tgt] = val

        fig, (ax_mmd, ax_wass) = plt.subplots(1, 2, figsize=(18, 8))

        # --- MMD Latent ---
        mmd_vals = mmd_matrix.values.astype(float)

        im1 = ax_mmd.imshow(mmd_vals, cmap="viridis_r")
        ax_mmd.set_xticks(range(len(labels)))
        ax_mmd.set_xticklabels(labels)
        ax_mmd.set_yticks(range(len(labels)))
        ax_mmd.set_yticklabels(labels)
        ax_mmd.set_xlabel("Target Label")
        ax_mmd.set_ylabel("Source Label")
        ax_mmd.set_title(f"MMD Latent")
        plt.colorbar(im1, ax=ax_mmd, label="MMD Latent")

        mmd_mean = np.nanmean(mmd_vals)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = mmd_matrix.iloc[i, j]
                if not np.isnan(val):
                    ax_mmd.text(j, i, f"{val:.4f}", ha="center", va="center",
                                fontsize=6, color="white" if val > mmd_mean else "black")

        # --- Wasserstein Latent ---
        wass_vals = wass_matrix.values.astype(float)
        im2 = ax_wass.imshow(wass_vals, cmap="viridis_r")
        ax_wass.set_xticks(range(len(labels)))
        ax_wass.set_xticklabels(labels)
        ax_wass.set_yticks(range(len(labels)))
        ax_wass.set_yticklabels(labels)
        ax_wass.set_xlabel("Target Label")
        ax_wass.set_ylabel("Source Label")
        ax_wass.set_title(f"Wasserstein Latent")
        plt.colorbar(im2, ax=ax_wass, label="Wasserstein Distance (Latent)")

        wass_mean = np.nanmean(wass_vals)
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = wass_matrix.iloc[i, j]
                if not np.isnan(val):
                    ax_wass.text(j, i, f"{val:.4f}", ha="center", va="center",
                                 fontsize=6, color="white" if val > wass_mean else "black")

        fig.suptitle(
            f"Latent Space Distance Heatmaps — Latent Dim = {latent_dim}  (γ = {OPTIMAL_GAMMA})",
            fontsize=14
        )
        plt.tight_layout()

        if save:
            plt.savefig(
                f"{save_dir}/fig_3_latent_heatmaps_dim_{latent_dim}_selfmade.png",
                dpi=300, bbox_inches="tight"
            )
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


def select_best_gamma_ttest(df, gammas, alpha=0.05, verbose=True):
    """
    Select the preferred gamma using sequential Welch's t-tests on pooled
    per-row mmd_latent values (pooled across all rows -- source/target pairs
    and latent dimensions -- at each gamma).
 
    Walk gammas in ascending order. Start with `selected = gammas[0]`.
    For each subsequent candidate gamma g:
        - Run Welch's t-test between selected's pooled mmd_latent values
          and candidate's pooled mmd_latent values.
        - If NOT significant (p >= alpha): prefer the larger gamma
          (no meaningful cost in MMD, so take the cheaper/larger one) ->
          selected = g.
        - If significant: only move to g if g's mean MMD is actually lower
          (a real improvement) -> selected = g. Otherwise keep `selected`
          and continue comparing future gammas against it.
 
    Returns
    -------
    selected_gamma : float
    log : list of dict
        One entry per pairwise comparison made, for inspection/reporting.
    """
    gammas = sorted(gammas)
    samples = {g: df.loc[df["gamma"] == g, "mmd_image"].dropna().values for g in gammas}
 
    selected = gammas[0]
    log = []
 
    for g in gammas[1:]:
        a = samples[selected]
        b = samples[g]
 
        if len(a) < 2 or len(b) < 2:
            t_stat, p_val = np.nan, np.nan
            significant = False
        else:
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test
            significant = p_val < alpha
 
        mean_selected = np.mean(a) if len(a) else np.nan
        mean_candidate = np.mean(b) if len(b) else np.nan
 
        if not significant:
            decision = f"not significant (p={p_val:.4f}) -> prefer larger gamma {g}"
            selected = g
        else:
            if mean_candidate < mean_selected:
                decision = (
                    f"significant (p={p_val:.4f}), candidate mean lower "
                    f"({mean_candidate:.4f} < {mean_selected:.4f}) -> move to {g}"
                )
                selected = g
            else:
                decision = (
                    f"significant (p={p_val:.4f}), candidate mean not lower "
                    f"({mean_candidate:.4f} >= {mean_selected:.4f}) -> keep {selected}"
                )
 
        log.append({
            "candidate": g,
            "t_stat": t_stat,
            "p_value": p_val,
            "significant": significant,
            "mean_selected_before": mean_selected,
            "mean_candidate": mean_candidate,
            "decision": decision,
        })
 
        if verbose:
            print(f"vs γ={g}: {decision}")
 
    return selected, log
 
 
def plot_mmd_vs_gamma(
    df=None,
    csv_path="data/evaluation_summary.csv",
    save_dir="figures/plots",
    save=True,
    alpha=0.05,
):
    """
    Plot latent MMD vs gamma, one line per latent dimension, with a single
    vertical line marking the globally selected optimal gamma (via sequential
    Welch's t-tests pooled across all dimensions and source/target pairs).
    """
    if df is None:
        df = pd.read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)
    gammas = sorted(df["gamma"].unique())
    dims = sorted(df["latent_dim"].unique())
 
    best_gamma, ttest_log = select_best_gamma_ttest(df, gammas, alpha=alpha)
    best_mmd_pooled = df.loc[df["gamma"] == best_gamma, "mmd_image"].mean()
 
    _, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(dims)))
 
    for color, d in zip(cmap, dims):
        sub = df[df["latent_dim"] == d]
        means = [sub.loc[sub["gamma"] == g, "mmd_image"].mean() for g in gammas]
        ax.plot(
            gammas, means,
            "o-", linewidth=2, markersize=5,
            color=color, alpha=0.9,
            label=f"latent_dim={d}",
        )
 
    ax.axvline(
        best_gamma, color="tomato", linestyle="--", linewidth=1.8, zorder=4,
        label=f"selected γ={best_gamma:.4g} (t-test, α={alpha})",
    )
 
    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$", fontsize=13)
    ax.set_ylabel("Image MMD", fontsize=13)
    ax.set_title(r"Image MMD vs $\gamma$ by Latent Dimension", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
 
    fname = "mmd_vs_gamma_by_dim.png"
    print(f"Selected γ: {best_gamma:.4g}  ->  pooled mean MMD = {best_mmd_pooled:.4f}")
 
    if save:
        plt.savefig(f"{save_dir}/{fname}", dpi=300, bbox_inches="tight")
    plt.show()
 

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


import matplotlib.ticker as ticker

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def plot_boxplot_time_iteration_per_latent_dim_log(summary_df, save_dir="figures/boxplots"):
    """
    Plot boxplots of running time and iteration count per latent dimension.

    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with columns: 'latent_dim', 'running_time', 'iter_count'.
    save_dir : str
        Directory to save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    latent_dims = sorted(summary_df["latent_dim"].unique())
    distributions = {"running_time": [], "iter_count": []}
    
    for dim in latent_dims:
        mask = summary_df["latent_dim"] == dim and summary_df["gamma"] == 0.1
        distributions["running_time"].append(
            summary_df.loc[mask, "running_time"].values
        )
        distributions["iter_count"].append(
            summary_df.loc[mask, "iter_count"].values
        )



    FLIER_PROPS = dict(marker="o", markerfacecolor="none",
                       markeredgecolor="#555", markersize=4, linestyle="none")
    BOX_PROPS   = dict(facecolor="#d9e8f5", color="#2c5f8a")
    MEDIAN_PROPS = dict(color="#c0392b", linewidth=1.8)
    WHISKER_PROPS = dict(color="#2c5f8a", linewidth=1.2)
    CAP_PROPS    = dict(color="#2c5f8a", linewidth=1.2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35)

    tick_labels = [str(d) for d in latent_dims]

    # --- Panel A: Running Time -------------------------------------------
    ax = axes[0]
    bp = ax.boxplot(
        distributions["running_time"],
        patch_artist=True,
        flierprops=FLIER_PROPS,
        boxprops=BOX_PROPS,
        medianprops=MEDIAN_PROPS,
        whiskerprops=WHISKER_PROPS,
        capprops=CAP_PROPS,
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if v >= 1 else f"{v:.2f}"
    ))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks(range(1, len(latent_dims) + 1))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Latent dimension $d$")
    ax.set_ylabel("Running time (s) — log scale")
    ax.set_title("(A) Running time per latent dimension")

    # --- Panel B: Iteration Count ----------------------------------------
    ax = axes[1]
    bp2 = ax.boxplot(
        distributions["iter_count"],
        patch_artist=True,
        flierprops=FLIER_PROPS,
        boxprops=BOX_PROPS,
        medianprops=MEDIAN_PROPS,
        whiskerprops=WHISKER_PROPS,
        capprops=CAP_PROPS,
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.set_xticks(range(1, len(latent_dims) + 1))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Latent dimension $d$")
    ax.set_ylabel("Iteration count (log scale)")
    ax.set_title("(B) Iteration count per latent dimension")

    # Shared legend element (median line)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#c0392b", linewidth=1.8, label="Median"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#d9e8f5",
                       edgecolor="#2c5f8a", label="IQR (box)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
    )

    save_path = os.path.join(save_dir, "boxplot_time_iteration_per_latent_dim_log.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {save_path}")

def plot_boxplot_time_iteration_per_latent_dim(summary_df, save_dir="figures/boxplots"):
    """
    Plot boxplots of running time and iteration count per latent dimension.
 
    Parameters
    ----------
    summary_df : pd.DataFrame
        DataFrame with columns: 'latent_dim', 'running_time', 'iter_count'.
    save_dir : str
        Directory to save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
 
    latent_dims = sorted(summary_df["latent_dim"].unique())
    distributions = {"running_time": [], "iter_count": []}
 
    for dim in latent_dims:
        mask = summary_df["latent_dim"] == dim
        distributions["running_time"].append(
            summary_df.loc[mask, "running_time"].values
        )
        distributions["iter_count"].append(
            summary_df.loc[mask, "iter_count"].values
        )
 

 
    FLIER_PROPS = dict(marker="o", markerfacecolor="none",
                       markeredgecolor="#555", markersize=4, linestyle="none")
    BOX_PROPS   = dict(facecolor="#d9e8f5", color="#2c5f8a")
    MEDIAN_PROPS = dict(color="#c0392b", linewidth=1.8)
    WHISKER_PROPS = dict(color="#2c5f8a", linewidth=1.2)
    CAP_PROPS    = dict(color="#2c5f8a", linewidth=1.2)
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.35)
 
    tick_labels = [str(d) for d in latent_dims]
 
    # --- Panel A: Running Time -------------------------------------------
    ax = axes[0]
    bp = ax.boxplot(
        distributions["running_time"],
        patch_artist=True,
        flierprops=FLIER_PROPS,
        boxprops=BOX_PROPS,
        medianprops=MEDIAN_PROPS,
        whiskerprops=WHISKER_PROPS,
        capprops=CAP_PROPS,
    )
    ax.set_xticks(range(1, len(latent_dims) + 1))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Latent dimension $d$")
    ax.set_ylabel("Running time (s)")
    ax.set_title("(A) Running time per latent dimension")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
 
    # --- Panel B: Iteration Count ----------------------------------------
    ax = axes[1]
    bp2 = ax.boxplot(
        distributions["iter_count"],
        patch_artist=True,
        flierprops=FLIER_PROPS,
        boxprops=BOX_PROPS,
        medianprops=MEDIAN_PROPS,
        whiskerprops=WHISKER_PROPS,
        capprops=CAP_PROPS,
    )
    ax.set_xticks(range(1, len(latent_dims) + 1))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Latent dimension $d$")
    ax.set_ylabel("Iteration count")
    ax.set_title("(B) Iteration count per latent dimension")
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
 
    # Shared legend element (median line)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#c0392b", linewidth=1.8, label="Median"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#d9e8f5",
                       edgecolor="#2c5f8a", label="IQR (box)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
    )
 
    save_path = os.path.join(save_dir, "boxplot_time_iteration_per_latent_dim.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {save_path}")

def fig_0_plot_barycentric_blurring_effect(summary_df, save_dir="figures/rapport", save=True):
    os.makedirs(save_dir, exist_ok=True)

    gammas = sorted(summary_df["gamma"].unique())
    latent_dims = sorted(summary_df["latent_dim"].unique())

    fig, ax = plt.subplots(figsize=(6, 5))

    def _ci(vals):
        vals = np.array([float(v) for v in vals])
        mean = np.mean(vals)
        ci = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return mean, mean - ci, mean + ci

    for dim in latent_dims:
        means, lowers, uppers = [], [], []
        for g in gammas:
            vals = summary_df.loc[
                (summary_df["latent_dim"] == dim) & (summary_df["gamma"] == g), "entropy"
            ].values
            m, lo, hi = _ci(vals)
            means.append(m); lowers.append(lo); uppers.append(hi)
        means, lowers, uppers = np.array(means), np.array(lowers), np.array(uppers)
        ax.plot(gammas, means, marker="o", linewidth=2, label=f"dim={dim}")
        ax.fill_between(gammas, lowers, uppers, alpha=0.12)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\gamma$", fontsize=12)
    ax.set_ylabel(r"Transport plan entropy  $H(P)$", fontsize=11)
    ax.set_title("Barycentric Blurring Effect", fontsize=13)
    ax.legend(title="Latent dim", fontsize=9, frameon=False)

    plt.tight_layout()
    if save:
        path = os.path.join(save_dir, "fig_0_barycentric_blurring_effect.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()

# def fig_0_plot_barycentric_blurring_effect(summary_df, save_dir="figures/plots", save=True):
   
#     os.makedirs(save_dir, exist_ok=True)

#     gammas = sorted(summary_df["gamma"].unique())
#     latent_dims = sorted(summary_df["latent_dim"].unique())
#     gamma_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(gammas)))

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     fig.suptitle(
#         "Barycentric Blurring Effect: how entropic regularisation degrades the transport map",
#         fontsize=13,
#         y=1.02,
#     )

#     def _ci(vals):
#         vals = np.array([float(v) for v in vals])
#         mean = np.mean(vals)
#         ci = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
#         return mean, mean - ci, mean + ci

#     # ── Panel A: H(P) vs gamma ─────────────────────────────────────────────────
#     ax = axes[0]
#     for dim in latent_dims:
#         means, lowers, uppers = [], [], []
#         for g in gammas:
#             vals = summary_df.loc[
#                 (summary_df["latent_dim"] == dim) & (summary_df["gamma"] == g), "entropy"
#             ].values
#             m, lo, hi = _ci(vals)
#             means.append(m); lowers.append(lo); uppers.append(hi)
#         means, lowers, uppers = np.array(means), np.array(lowers), np.array(uppers)
#         ax.plot(gammas, means, marker="o", linewidth=2, label=f"dim={dim}")
#         ax.fill_between(gammas, lowers, uppers, alpha=0.12)

#     ax.set_xscale("log")
#     ax.set_xlabel(r"$\gamma$", fontsize=12)
#     ax.set_ylabel(r"Transport plan entropy  $H(P)$", fontsize=11)
#     ax.set_title(
#         "(A)  Plan entropy increases with $\\gamma$\n"
#         r"$\rightarrow$ mapping becomes more diffuse",
#         fontsize=10,
#     )
#     ax.legend(title="Latent dim", fontsize=9, frameon=False)

#     # ── Panel B: Classifier confidence vs gamma ────────────────────────────────
#     ax = axes[1]
#     for dim in latent_dims:
#         means, lowers, uppers = [], [], []
#         for g in gammas:
#             vals = summary_df.loc[
#                 (summary_df["latent_dim"] == dim) & (summary_df["gamma"] == g),
#                 "classifier_confidence_image",
#             ].values
#             m, lo, hi = _ci(vals)
#             means.append(m); lowers.append(lo); uppers.append(hi)
#         means, lowers, uppers = np.array(means), np.array(lowers), np.array(uppers)
#         ax.plot(gammas, means, marker="o", linewidth=2, label=f"dim={dim}")
#         ax.fill_between(gammas, lowers, uppers, alpha=0.12)

#     ax.set_xscale("log")
#     ax.set_xlabel(r"$\gamma$", fontsize=12)
#     ax.set_ylabel("Classifier confidence (target class)", fontsize=11)
#     ax.set_title(
#         "(B)  Confidence stays high at large $\\gamma$\n"
#         r"$\rightarrow$ blurred centroid fools the metric",
#         fontsize=10,
#     )
#     ax.legend(title="Latent dim", fontsize=9, frameon=False)

#     # ── Panel C: Entropy vs classifier confidence scatter coloured by gamma ────
#     ax = axes[2]
#     for g, color in zip(gammas, gamma_colors):
#         sub = summary_df[summary_df["gamma"] == g]
#         entropies = np.array([float(v) for v in sub["entropy"].values])
#         confs = np.array([float(v) for v in sub["classifier_confidence_image"].values])
#         ax.scatter(entropies, confs, color=color, alpha=0.5, s=18, label=f"$\\gamma$={g}")

#     ax.set_xlabel(r"Transport plan entropy  $H(P)$", fontsize=11)
#     ax.set_ylabel("Classifier confidence (target class)", fontsize=11)
#     ax.set_title(
#         "(C)  High entropy correlates with high confidence\n"
#         r"$\rightarrow$ degenerate plans inflate the metric",
#         fontsize=10,
#     )
#     ax.legend(title=r"$\gamma$", fontsize=9, frameon=False, markerscale=1.8)

#     plt.tight_layout()
#     if save:
#         path = os.path.join(save_dir, "barycentric_blurring_effect.png")
#         plt.savefig(path, dpi=300, bbox_inches="tight")
#         print(f"Saved → {path}")
#     plt.show()


from matplotlib.lines import Line2D
def fig_1_plot_boxplot_mmd_per_gamma(summary_df, save_dir="figures/rapport"):
    os.makedirs(save_dir, exist_ok=True)

    gammas = sorted(summary_df["gamma"].unique())

    distributions = []
    for gamma in gammas:
        mask = summary_df["gamma"] == gamma
        distributions.append(
            summary_df.loc[mask, "mmd_image"].values  # full pipeline MMD
        )

    FLIER_PROPS   = dict(marker="o", markerfacecolor="none",
                         markeredgecolor="#555", markersize=4, linestyle="none")
    BOX_PROPS     = dict(facecolor="#d9e8f5", color="#2c5f8a")
    MEDIAN_PROPS  = dict(color="#c0392b", linewidth=1.8)
    WHISKER_PROPS = dict(color="#2c5f8a", linewidth=1.2)
    CAP_PROPS     = dict(color="#2c5f8a", linewidth=1.2)

    fig = plt.figure(figsize=(10, 7))
    tick_labels = [str(gamma) for gamma in gammas]

    ax = fig.add_axes([0.1, 0.15, 0.85, 0.75])  # leave room for legend

    bp = ax.boxplot(                   # noqa: F841
        distributions,                 # ← was distributions["running_time"]
        patch_artist=True,
        flierprops=FLIER_PROPS,
        boxprops=BOX_PROPS,
        medianprops=MEDIAN_PROPS,
        whiskerprops=WHISKER_PROPS,
        capprops=CAP_PROPS,
    )

    ax.set_xticks(range(1, len(gammas) + 1))
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Regularisation $\\gamma$")        # ← was "Latent dimension $d$"
    ax.set_ylabel("MMD (image space)")               # ← was "Running time (s)"
    ax.set_title("MMD per $\\gamma$ (image space)")  # ← was Running time title
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    legend_elements = [
        Line2D([0], [0], color="#c0392b", linewidth=1.8, label="Median"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#d9e8f5",
                      edgecolor="#2c5f8a", label="IQR (box)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.04),
    )

    save_path = os.path.join(save_dir, "fig_1_boxplot_mmd_per_gamma.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {save_path}")

if __name__ == "__main__":
    summary_df = pd.read_csv("data/evaluation_summary.csv")



    training_data,_ = getData()
    model = load_with_hyperparams("ae_best_model_bo_2")

    plot_single_point_transport(
    training_data,
    model,
    2,
    6,
    0.001)
    plot_barycentric_geometry_vs_gamma_dim2(training_data,model=model,source_label=0,target_label=1)
    # pca_visualize_for_high_dimension(training_data,model=load_with_hyperparams("ae_best_model_bo_4"))
    # plot_mmd_vs_gamma(df=summary_df)
    # fig_0_plot_barycentric_blurring_effect(summary_df=summary_df)
    # fig_1_plot_boxplot_mmd_per_gamma(summary_df=summary_df)
    # fig_2_dim_vs_gamma_metrics_table(summary_df=summary_df)
    fig_3_plot_mmd_heatmaps_individual(summary_df=summary_df)
    # fig_4_plot_gamma_vs_mmd(summary_df=summary_df)
    # fig_2_dim_vs_gamma_metrics_table(summary_df=summary_df)
    