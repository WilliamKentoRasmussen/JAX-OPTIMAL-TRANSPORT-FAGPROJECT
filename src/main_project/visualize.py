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
from main_project.utils import load
from main_project.data import getData  # fixed

labels_map = {i: str(i) for i in range(10)}

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
    ax.set_aspect('equal')
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
        axes[0, i].imshow(x[i], cmap="gray"); axes[0, i].axis("off")
        axes[1, i].imshow(recon[i], cmap="gray"); axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.suptitle("Reconstruction Results")
    plt.tight_layout()
    plt.show()

def plot_training_loss(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['epoch'], data['train_loss'], label='Training Loss', color='blue')
    plt.plot(data['epoch'], data['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend(); plt.grid(True)
    plt.show()

if __name__ == "__main__":
    training_data, test_data = getData()
    loss_data = pd.read_csv("data/training_history.csv")
    model = load(name="ae_best_model_lat2", path="models")
    plot_training_loss(loss_data)
    plot_reconstruction(training_data, model)
    plot_latent_clusters(training_data, model)
