import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd

from main_project.sinkhorn import run_sinkhorn_by_model, cdist_euclidean
from main_project.utils import load_with_hyperparams
from main_project.environment import GAMMA, MODELS_DIM

if __name__ == "__main__":
    W_distances = []
    
    for dim in MODELS_DIM:
        model = load_with_hyperparams(name=f"ae_best_model_bo_{dim}")
        W_distances = []
        for gamma_val in GAMMA:
            latent_source, latent_target , P, _, _, _ =run_sinkhorn_by_model(model, gamma = gamma_val)
            C = cdist_euclidean(latent_source, latent_target)
            H = -jnp.sum(P * jnp.log(P + 1e-10))
            W_gamma = jnp.sum(C * P) - gamma_val * H
            W_distances.append(W_gamma)
        print(W_distances)

