import torch
import matplotlib.pyplot as plt
import jax
import equinox as eqx
from main_project.model import AEv2
import jax.random as jr
import jax.numpy as jnp
import numpy as np
from data import getData
from main_project.train import train,train_step,loss_fn
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import sklearn
from main_project.utils import load

training_data, test_data = getData()

model = load(name="model", path="models")



def plot_random_samples():
    figure = plt.figure(figsize=(10, 4))
    cols, rows = 10, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def plot_latent_clusters(
    training_data,
    model=model,
    max_points=5000,   # allow many more points now
    point_size=8,
    alpha=0.6
):
    # --- Load data ---
    xs, ys = [], []
    for img, label in training_data:
        xs.append(img.numpy())
        ys.append(label)

    x = jnp.array(xs)
    labels = jnp.array(ys)

    # Flatten images
    x = x.reshape(x.shape[0], -1)

    # --- Subsample (but allow more points than before) ---
    n = len(x)
    idx = np.random.choice(n, size=min(max_points, n), replace=False)

    x = x[idx]
    labels = labels[idx]

    # --- Model forward pass ---
    _, z = jax.vmap(model)(x)
    z = np.array(z)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        z[:, 0],
        z[:, 1],
        c=labels,
        cmap="tab10",
        s=point_size,
        alpha=alpha
    )

    # --- Styling ---
    ax.set_xlabel("Latent dimension $z_1$")
    ax.set_ylabel("Latent dimension $z_2$")
    ax.set_title("Latent Space Clusters (Colored by Digit)")

    ax.set_aspect('equal')

    # Colorbar instead of legend (cleaner for many points)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Digit")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_latent_clusters(training_data)
