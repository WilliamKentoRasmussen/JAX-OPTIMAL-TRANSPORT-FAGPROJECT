from typing import Union
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from main_project.utils import load
from main_project.data import getData, getDataloader
from main_project.visualize import plot_latent_clusters


model = load(name="ae_best_model_lat2", path="models")


_, test_data = getData()
test_loader = getDataloader(test_data)


def recon(x):
    recon, z = model(x)
    return recon, z


all_imgs = jnp.array(np.stack([np.array(img).flatten() for img, _ in test_data]))
all_labels = np.array([label for _, label in test_data])


recons, z = jax.vmap(recon)(all_imgs)
z_array = jnp.array(z)
recon_array = jnp.array(recons)


def evaluate_latent_space_knn(latent_array, labels):
    classifier = KNeighborsClassifier(n_neighbors=3)  # 5 by default
    knn_acc = cross_val_score(classifier, latent_array, labels, cv=5).mean()
    return knn_acc


if __name__ == "__main__":
    knn_acc = evaluate_latent_space_knn(z_array, all_labels)
    print(f"KNN Split accuracy {knn_acc}")
