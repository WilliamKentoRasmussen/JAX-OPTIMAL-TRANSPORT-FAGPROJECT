import optuna
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import optax
from optuna import trial
import torch
from torch.utils.data import Subset
import pandas as pd

from main_project.model import AEv2
from main_project.data import getData, getDataloader
from main_project.trainnew import AETrainer
from main_project.evaluate import MMD, classifier_confidence, evaluate_latent_space_knn
from main_project.environment import INTERMEDIATE_FRACTIONS
from environment import MODELS_DIM
from main_project.utils import save_with_hyperparams
import os
# https://medium.com/@vikakbary/the-first-step-to-optuna-understanding-766e50488c67

if os.path.exists("ae_bo.db"):
    os.remove("ae_bo.db")


def objective(trial, latent_dim=2):
    arch_name = trial.suggest_categorical(
        "arch", ["small2", "small", "medium", "large", "large2", "medium2", "large3", "large4"]
    )  # try different architectures
    hidden_dim = arch_presets[arch_name]
    lr = trial.suggest_float(
        "lr", 1e-4, 1e-2, log=True
    )  # try different learning rates in the log domain within the domain
    lambda_l2 = trial.suggest_float(
        "lambda_l2", 1e-5, 1e-1, log=True
    )  # try regulariztion rates in the log domain within the domain

    # loss_name = trial.suggest_categorical("loss", ["mse", "mae", "huber", "bce"])

    key = jr.PRNGKey(trial.number)
    model = AEv2(latent_dim=latent_dim, hidden_dims=hidden_dim, key=key)
    trainer = AETrainer(
        model=model,
        learning_rate=lr,
        lambda_l2=lambda_l2,
    )

    # ── 3. Train using your existing Trainer ─────────────────────────────────
    trained_model, history, test_loss = trainer.train(
        epochs=30, val_split=0.2, model=model, model_name=f"trial_{trial.number}"
    )

    # Prune bad trials after training
    val_loss = history[-1]["val_loss"]
    trial.report(val_loss, step=30)
    if trial.should_prune():
        raise optuna.TrialPruned()

    training_data, _ = getData()
    n = len(training_data)
    split = int(n * 0.8)
    val_subset = Subset(training_data, range(split, n))
    val_loader = getDataloader(val_subset)

    trial.set_user_attr("val_loss", float(val_loss))  # save the attributes for each trial

    return val_loss


arch_presets = {
    "small2": [64, 32, 16, 16, 8],
    "small": [128, 64, 32, 16, 8],
    "medium": [256, 128, 64, 32, 16],
    "medium2": [256, 256, 128, 64, 32],
    "large": [512, 256, 128, 64, 32],
    "large2": [512, 512, 256, 128, 64],
    "large3": [512, 512, 512, 256, 128],
    "large4": [512, 512, 512, 512, 256],
}

best_parameters = []
# ── Run ───────────────────────────────────────────────────────────────────────
for latent_dim in MODELS_DIM:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=5
        ),  # stop early if val loss is below median of previous trials likely not optimum
        study_name=f"ae_full_optimization_{latent_dim}",
        load_if_exists=True,
    )

    study.optimize(lambda trial: objective(trial, latent_dim=latent_dim), n_trials=20)

    best = study.best_trial
    print(f"\nBest trial for latent dimension {latent_dim}:")

    print(f"\nBest objective:  {best.value:.4f}")
    print(f"\nBest hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k:15s}: {v}")

    print(f"\nValidation metrics:")
    print(f"  val_loss:           {best.user_attrs['val_loss']:.4f}")

    print(f"\nRetraining best configuration for latent dimension {latent_dim} on full training data...")
    best_params = best.params
    key = jr.PRNGKey(999)
    best_model = AEv2(
        latent_dim=latent_dim,
        hidden_dims=arch_presets[best_params["arch"]],
        key=key,
    )
    final_trainer = AETrainer(
        model=best_model,
        learning_rate=best_params["lr"],
        lambda_l2=best_params["lambda_l2"],
    )
    final_model, _, final_test_loss = final_trainer.train(
        epochs=500, val_split=0.2, model=best_model, model_name=f"ae_best_model_bo_{latent_dim}"
    )
    print(f"Final test loss: {final_test_loss:.4f}")
    save_with_hyperparams(
        model=final_model,
        filename=f"ae_best_model_bo_{latent_dim}",
        hidden_dims=arch_presets[best_params["arch"]],
        latent_dim=latent_dim,
    )
    best_parameters.append((latent_dim, best.params, final_test_loss))

df = pd.DataFrame(best_parameters, columns=["latent_dim", "best_params", "final_test_loss"])
df.to_csv("best_hyperparameters.csv", index=False)
