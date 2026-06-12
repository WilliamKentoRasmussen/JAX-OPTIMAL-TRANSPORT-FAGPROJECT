import torch
import matplotlib.pyplot as plt
import jax
import equinox as eqx
from main_project.model import AEv2
import jax.random as jr
import jax.numpy as jnp
import numpy as np
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from main_project.utils import load, load_with_hyperparams
from main_project.data import getData  # fixed

from main_project.train import train_classifier
from main_project.model import targetClassifier
from main_project.utils import load
from main_project.environment import MODELS_DIM, INTERMEDIATE_FRACTIONS, MAX_POINTS, LABELS
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


def plot_gamma_vs_mmd(summary_df,
                      save_dir="figures/plots"):


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
                (summary_df["latent_dim"] == dim)
                & (summary_df["gamma"] == gamma),
                "mmd_image"
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

        ax.plot(
            gammas,
            means,
            marker="o",
            linewidth=2,
            label=f"dim={dim}"
        )

        ax.fill_between(
            gammas,
            lowers,
            uppers,
            alpha=0.15
        )

    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel("Average MMD Image")
    ax.set_title("Average MMD vs Gamma")

    ax.set_xscale("log")  # useful if gamma=[0.01,0.1,1,10]

    ax.grid(True, alpha=0.3)
    ax.legend(title="Latent Dim", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()

    plt.savefig(
        f"{save_dir}/gamma_vs_mmd.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_latent_dim_vs_average_mmd(summary_df,
                                   save_dir="figures/plots"):

    os.makedirs(save_dir, exist_ok=True)

    latent_dims = sorted(summary_df["latent_dim"].unique())

    means = []
    lower = []
    upper = []

    for dim in latent_dims:

        values = summary_df.loc[
            summary_df["latent_dim"] == dim,
            "mmd_image"
        ].values

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

    ax.plot(
        latent_dims,
        means,
        marker="o",
        linewidth=2,
        label="Mean MMD"
    )

    ax.fill_between(
        latent_dims,
        lower,
        upper,
        alpha=0.25,
        label="95% CI"
    )

    ax.set_xlabel("Latent Dimension")
    ax.set_ylabel("Average MMD Image")
    ax.set_title("Average MMD vs Latent Dimension")

    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    plt.savefig(
        f"{save_dir}/latent_dim_vs_average_mmd.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def plot_mmd_image_heatmaps_full(summary_df, save_dir="figures/heatmaps"):

    os.makedirs(save_dir, exist_ok=True)

    labels = LABELS

    #Laver et heatmap for hvert dimension
    for latent_dim in sorted(summary_df["latent_dim"].unique()):

        model_df = summary_df[
            summary_df["latent_dim"] == latent_dim
        ]

        matrix = pd.DataFrame(
            np.nan,
            index=labels,
            columns=labels
        )

        grouped = (
            model_df
            .groupby(["source_label", "target_label"])["mmd_image"]
            .mean()
        )

        for (src, tgt), value in grouped.items():
            matrix.loc[src, tgt] = value

        fig, ax = plt.subplots(figsize=(8, 8))

        im = ax.imshow(matrix)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

        ax.set_xlabel("Target Label")
        ax.set_ylabel("Source Label")
        ax.set_title(
            f"MMD Image Heatmap (avg γ)\nLatent Dim = {latent_dim}"
        )

        plt.colorbar(im, ax=ax, label="MMD Image")

        for i in range(len(labels)):
            for j in range(len(labels)):
                val = matrix.iloc[i, j]

                if not np.isnan(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7
                    )

        plt.tight_layout()

        plt.savefig(
            f"{save_dir}/mmd_image_heatmap_full_dim_{latent_dim}.png",
            dpi=300
        )
        plt.close()
    
    
if __name__ == "__main__":
    training_data, test_data = getData()
    loss_data = pd.read_csv("training_history_ae_best_model_bo_2.csv")
    model = load_with_hyperparams(name="ae_best_model_bo_2", path="models")
    plot_training_loss(loss_data)
    plot_reconstruction(training_data, model)
    plot_latent_clusters(training_data, model)
    pca_visualize_for_high_dimension(training_data, model)
 

