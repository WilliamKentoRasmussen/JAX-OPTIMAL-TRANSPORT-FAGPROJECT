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
from main_project.environment import MODELS_DIM, INTERMEDIATE_FRACTIONS, MAX_POINTS

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


# if __name__ == "__main__":
#     training_data, test_data = getData()
#     loss_data = pd.read_csv("data/training_history.csv")
#     model = load(name="ae_best_model_lat2", path="models")
#     plot_training_loss(loss_data)
#     plot_reconstruction(training_data, model)
#     plot_latent_clusters(training_data, model)
#     pca_visualize_for_high_dimension(training_data, model)



def _load_eval_csv(model_name, sb=False):
    """Load evaluation CSV for a given model, Sinkhorn or SB variant."""
    if sb:
        return pd.read_csv(f"data/sb_{model_name}/evaluation.csv")
    return pd.read_csv(f"data/{model_name}/evalution.csv")
 
 
def _minmax_normalize(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series * 0.0
    return (series - lo) / (hi - lo)
 
 
# ── 1. Metrics vs. transport fraction, one line per latent dim ────────────────
 
def plot_metrics_vs_fraction_by_dim(dims=MODELS_DIM):
    """
    Three stacked subplots (MMD, FID, Classifier Confidence) with one line per
    latent dimension.  Covers both Sinkhorn and Schrödinger Bridge as separate
    figures so the trajectories are easy to compare.
    """
    metrics_sinkhorn = [("MDD", "MMD"), ("FID", "FID"), ("Confidence of Classifier", "Classifier Confidence")]
    metrics_sb       = [("MMD", "MMD"), ("FID", "FID"), ("Confidence of Classifier", "Classifier Confidence")]
 
    for method, metric_cols, loader_flag in [
        ("Sinkhorn",           metrics_sinkhorn, False),
        ("Schrödinger Bridge", metrics_sb,       True),
    ]:
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        fig.suptitle(f"Metrics vs. Transport Fraction — {method}", fontsize=14, y=1.01)
 
        for dim in dims:
            model_name = f"ae_model_dim_{dim}"
            try:
                df = _load_eval_csv(model_name, sb=loader_flag)
            except FileNotFoundError:
                continue
 
            x_col = df.columns[1]  # fraction or t column
            for ax, (col, label) in zip(axes, metric_cols):
                if col not in df.columns:
                    # Sinkhorn CSV has a typo: "MDD" instead of "MMD"
                    col = "MDD" if col == "MMD" and "MDD" in df.columns else col
                ax.plot(df[x_col], df[col], marker="o", markersize=3, label=f"dim={dim}")
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, loc="upper right")
 
        axes[-1].set_xlabel("Transport fraction / time step")
        plt.tight_layout()
        plt.savefig(f"data/plots/metrics_vs_fraction_{method.replace(' ', '_').lower()}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
 
 
# ── 2. Normalised overlay of all three metrics on one plot ────────────────────
 
def plot_normalized_metrics_overlay(dims=MODELS_DIM):
    """
    For each (method, dim) combination: MMD, FID, and confidence normalised to
    [0, 1] and drawn on a single axes so agreement / divergence is visible.
    """
    for method, loader_flag in [("Sinkhorn", False), ("Schrödinger Bridge", True)]:
        n_cols = min(3, len(dims))
        n_rows = int(np.ceil(len(dims) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
        fig.suptitle(f"Normalised Metrics Overlay — {method}", fontsize=13)
 
        for idx, dim in enumerate(dims):
            ax = axes[idx // n_cols][idx % n_cols]
            model_name = f"ae_model_dim_{dim}"
            try:
                df = _load_eval_csv(model_name, sb=loader_flag)
            except FileNotFoundError:
                ax.set_visible(False)
                continue
 
            x_col = df.columns[1]
            mmd_col  = "MDD" if "MDD" in df.columns else "MMD"
            fid_col  = "FID"
            conf_col = "Confidence of Classifier"
 
            for col, color, label in [
                (mmd_col,  "steelblue", "MMD"),
                (fid_col,  "tomato",    "FID"),
                (conf_col, "seagreen",  "Confidence"),
            ]:
                if col not in df.columns:
                    continue
                ax.plot(df[x_col], _minmax_normalize(df[col]), color=color, label=label,
                        marker=".", markersize=3)
 
            ax.set_title(f"dim = {dim}")
            ax.set_xlabel("Fraction / t")
            ax.set_ylabel("Normalised value")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
 
        # hide unused subplots
        for idx in range(len(dims), n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)
 
        plt.tight_layout()
        plt.savefig(f"data/plots/normalised_overlay_{method.replace(' ', '_').lower()}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
 
 
# ── 3. Final-step paired bar chart: Sinkhorn vs. SB ──────────────────────────
 
def plot_final_step_bar_comparison(dims=MODELS_DIM):
    """
    At t=1.0 (last row of the CSV), compare Sinkhorn vs. Schrödinger Bridge for
    each metric across latent dimensions.
    """
    metric_pairs = [
        ("MDD",  "MMD",  "MMD at t=1"),
        ("FID",  "FID",  "FID at t=1"),
        ("Confidence of Classifier", "Confidence of Classifier", "Confidence at t=1"),
    ]
 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Final Transport Quality: Sinkhorn vs. Schrödinger Bridge", fontsize=13)
 
    x = np.arange(len(dims))
    width = 0.35
 
    for ax, (sink_col, sb_col, title) in zip(axes, metric_pairs):
        sink_vals, sb_vals = [], []
        for dim in dims:
            model_name = f"ae_model_dim_{dim}"
            try:
                df_s = _load_eval_csv(model_name, sb=False)
                sink_val = df_s[sink_col].iloc[-1] if sink_col in df_s.columns else np.nan
            except FileNotFoundError:
                sink_val = np.nan
            try:
                df_b = _load_eval_csv(model_name, sb=True)
                sb_val = df_b[sb_col].iloc[-1] if sb_col in df_b.columns else np.nan
            except FileNotFoundError:
                sb_val = np.nan
            sink_vals.append(sink_val)
            sb_vals.append(sb_val)
 
        bars1 = ax.bar(x - width / 2, sink_vals, width, label="Sinkhorn",           color="steelblue", alpha=0.85)
        bars2 = ax.bar(x + width / 2, sb_vals,   width, label="Schrödinger Bridge", color="tomato",    alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"dim={d}" for d in dims], rotation=30)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("data/plots/final_step_bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ── 4. Shaded gap — full trajectory difference ────────────────────────────────
 
def plot_shaded_trajectory_gap(dims=MODELS_DIM):
    """
    For each metric, draw both method trajectories and shade the gap between
    them so the divergence over time is immediately visible.
    """
    metric_triples = [
        ("MDD",  "MMD",  "MMD"),
        ("FID",  "FID",  "FID"),
        ("Confidence of Classifier", "Confidence of Classifier", "Classifier Confidence"),
    ]
 
    for dim in dims:
        model_name = f"ae_model_dim_{dim}"
        try:
            df_s = _load_eval_csv(model_name, sb=False)
            df_b = _load_eval_csv(model_name, sb=True)
        except FileNotFoundError:
            continue
 
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f"Sinkhorn vs. Schrödinger Bridge trajectory gap — dim={dim}", fontsize=12)
 
        # Align on a common grid via interpolation
        t_common = np.linspace(0, 1, 100)
        x_s = df_s.iloc[:, 1].values
        x_b = df_b.iloc[:, 1].values
 
        for ax, (sink_col, sb_col, label) in zip(axes, metric_triples):
            if sink_col not in df_s.columns or sb_col not in df_b.columns:
                ax.set_visible(False)
                continue
 
            y_s = np.interp(t_common, x_s / x_s.max(), df_s[sink_col].values)
            y_b = np.interp(t_common, x_b / x_b.max(), df_b[sb_col].values)
 
            ax.plot(t_common, y_s, color="steelblue", label="Sinkhorn")
            ax.plot(t_common, y_b, color="tomato",    label="Schrödinger Bridge")
            ax.fill_between(t_common, y_s, y_b, alpha=0.18, color="purple", label="Gap")
            ax.set_title(label)
            ax.set_xlabel("Normalised transport fraction")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
 
        plt.tight_layout()
        plt.savefig(f"data/plots/shaded_gap_dim_{dim}.png", dpi=150, bbox_inches="tight")
        plt.show()
 
 
# ── 5. Transport image grids at multiple checkpoints ─────────────────────────
 
def plot_transport_checkpoints(model_name, n_images=5, checkpoint_fracs=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """
    Grid of source images → intermediate steps → target images sampled at
    specific transport fractions.  Each row is one fraction level.
    """
    intermediate_images = np.load(f"data/{model_name}/intermediate_images.npy")
    source_img          = np.load(f"data/{model_name}/original_images.npy")
    target_img          = np.load(f"data/{model_name}/target_images.npy")
 
    # intermediate_images: shape (n_samples, n_steps, 784) → (n_steps, n_samples, 784)
    intermediate_images = intermediate_images.transpose(1, 0, 2)
    n_steps = intermediate_images.shape[0]
 
    frac_indices = [int(f * (n_steps - 1)) for f in checkpoint_fracs]
    n_rows = len(frac_indices) + 2  # source row + fraction rows + target row
 
    fig, axes = plt.subplots(n_rows, n_images, figsize=(2 * n_images, 2 * n_rows))
 
    def _show_row(row_idx, images, row_label):
        for col in range(n_images):
            axes[row_idx, col].imshow(images[col].reshape(28, 28), cmap="gray")
            axes[row_idx, col].axis("off")
        axes[row_idx, 0].set_ylabel(row_label, fontsize=9)
 
    _show_row(0, source_img, "Source (t=0)")
    for row_idx, step in enumerate(frac_indices, start=1):
        frac = checkpoint_fracs[row_idx - 1]
        _show_row(row_idx, intermediate_images[step], f"t={frac:.2f}")
    _show_row(n_rows - 1, target_img, "Target (t=1)")
 
    plt.suptitle(f"Transport checkpoints — {model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"data/plots/transport_checkpoints_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ── 6. Side-by-side Sinkhorn vs. SB transport paths ──────────────────────────
 
def plot_sinkhorn_vs_sb_side_by_side(model_name, n_images=5):
    """
    Two rows of image grids, one for Sinkhorn and one for Schrödinger Bridge,
    both showing the same source images transported at evenly spaced steps.
    """
    intermediate_sink = np.load(f"data/{model_name}/intermediate_images.npy").transpose(1, 0, 2)
    decoded_sb        = np.load(f"data/sb_{model_name}/decoded.npy").transpose(1, 0, 2)
    source_img        = np.load(f"data/{model_name}/original_images.npy")
 
    n_steps_sink = intermediate_sink.shape[0]
    n_steps_sb   = decoded_sb.shape[0]
 
    # Sample 5 evenly spaced steps from each
    sink_steps = [intermediate_sink[i] for i in np.linspace(0, n_steps_sink - 1, 5, dtype=int)]
    sb_steps   = [decoded_sb[i]        for i in np.linspace(0, n_steps_sb   - 1, 5, dtype=int)]
 
    fig, axes = plt.subplots(2, n_images, figsize=(2 * n_images, 5))
 
    for col, step_imgs in enumerate(sink_steps):
        axes[0, col].imshow(step_imgs[0].reshape(28, 28), cmap="gray")
        axes[0, col].axis("off")
        t_label = f"t={col / (n_images - 1):.2f}"
        axes[0, col].set_title(t_label, fontsize=8)
 
    for col, step_imgs in enumerate(sb_steps):
        axes[1, col].imshow(step_imgs[0].reshape(28, 28), cmap="gray")
        axes[1, col].axis("off")
 
    axes[0, 0].set_ylabel("Sinkhorn", fontsize=10)
    axes[1, 0].set_ylabel("Schrödinger Bridge", fontsize=10)
 
    plt.suptitle(f"Transport path comparison — {model_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"data/plots/sinkhorn_vs_sb_{model_name}.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ── 7. FID vs. MMD scatter across all (model, fraction) points ───────────────
 
def plot_fid_vs_mmd_scatter(dims=MODELS_DIM):
    """
    Scatter of FID vs. MMD for every (model, fraction) data point.  Points are
    coloured by latent dimension so clusters and outliers are easy to spot.
    If the two metrics agree, points should lie roughly on a line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("tab10")
 
    for ax, (method, loader_flag, mmd_col) in zip(axes, [
        ("Sinkhorn",           False, "MDD"),
        ("Schrödinger Bridge", True,  "MMD"),
    ]):
        for i, dim in enumerate(dims):
            model_name = f"ae_model_dim_{dim}"
            try:
                df = _load_eval_csv(model_name, sb=loader_flag)
            except FileNotFoundError:
                continue
            if mmd_col not in df.columns or "FID" not in df.columns:
                continue
            ax.scatter(df[mmd_col], df["FID"], color=cmap(i % 10),
                       label=f"dim={dim}", alpha=0.75, s=25)
 
        ax.set_xlabel("MMD")
        ax.set_ylabel("FID")
        ax.set_title(f"FID vs. MMD — {method}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig("data/plots/fid_vs_mmd_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ── 8. Final-step metrics vs. latent dimension (scaling ablation) ─────────────
 
def plot_metrics_vs_latent_dim(dims=MODELS_DIM):
    """
    Line plots of final-step FID, MMD, and confidence vs. latent dimension
    size — answers whether a larger latent space produces better transport.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Transport Quality vs. Latent Dimension Size (at t=1)", fontsize=13)
 
    metric_triples = [
        ("MDD",  "MMD",  "MMD",                "steelblue"),
        ("FID",  "FID",  "FID",                "tomato"),
        ("Confidence of Classifier", "Confidence of Classifier", "Classifier Confidence", "seagreen"),
    ]
 
    for ax, (sink_col, sb_col, label, color) in zip(axes, metric_triples):
        sink_vals, sb_vals = [], []
        for dim in dims:
            model_name = f"ae_model_dim_{dim}"
            try:
                df_s = _load_eval_csv(model_name, sb=False)
                sink_vals.append(df_s[sink_col].iloc[-1] if sink_col in df_s.columns else np.nan)
            except FileNotFoundError:
                sink_vals.append(np.nan)
            try:
                df_b = _load_eval_csv(model_name, sb=True)
                sb_vals.append(df_b[sb_col].iloc[-1] if sb_col in df_b.columns else np.nan)
            except FileNotFoundError:
                sb_vals.append(np.nan)
 
        ax.plot(dims, sink_vals, marker="o", color=color,        linestyle="-",  label="Sinkhorn")
        ax.plot(dims, sb_vals,   marker="s", color=color,        linestyle="--", label="Schrödinger Bridge", alpha=0.7)
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(dims)
 
    plt.tight_layout()
    plt.savefig("data/plots/metrics_vs_latent_dim.png", dpi=150, bbox_inches="tight")
    plt.show()
 
 
# ─────────────────────────────────────────────
# Main runners
# ─────────────────────────────────────────────
 
def run_all_eval_plots(dims=MODELS_DIM):
    import os
    os.makedirs("data/plots", exist_ok=True)
 
    print("1/8  Metrics vs. transport fraction by dim…")
    plot_metrics_vs_fraction_by_dim(dims)
 
    print("2/8  Normalised metrics overlay…")
    plot_normalized_metrics_overlay(dims)
 
    print("3/8  Final-step bar comparison…")
    plot_final_step_bar_comparison(dims)
 
    print("4/8  Shaded trajectory gap…")
    plot_shaded_trajectory_gap(dims)
 
    print("5/8  Transport image checkpoints…")
    for dim in dims:
        plot_transport_checkpoints(f"ae_model_dim_{dim}")
 
    print("6/8  Sinkhorn vs. SB side-by-side…")
    for dim in dims:
        plot_sinkhorn_vs_sb_side_by_side(f"ae_model_dim_{dim}")
 
    print("7/8  FID vs. MMD scatter…")
    plot_fid_vs_mmd_scatter(dims)
 
    print("8/8  Metrics vs. latent dimension…")
    plot_metrics_vs_latent_dim(dims)
 
    print("Done.")
 
 
if __name__ == "__main__":
    training_data, test_data = getData()
    loss_data = pd.read_csv("training_history_ae_best_model_bo_2.csv")
    model = load_with_hyperparams(name="ae_best_model_bo_2", path="models")
    plot_training_loss(loss_data)
    plot_reconstruction(training_data, model)
    plot_latent_clusters(training_data, model)
    pca_visualize_for_high_dimension(training_data, model)
 
    run_all_eval_plots()
