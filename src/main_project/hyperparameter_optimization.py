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
from main_project.trainnew import Trainer
from main_project.evaluate import MMD, classifier_confidence, evaluate_latent_space_knn
from main_project.environment import INTERMEDIATE_FRACTIONS

import os

if os.path.exists("ae_bo.db"):
    os.remove("ae_bo.db")


def compute_bo_metrics(model, val_loader, source_class=0, target_class=1):
    all_z, all_labels = [], []

    for imgs, labels in val_loader:
        imgs_flat = jnp.array(imgs.numpy()).reshape(imgs.shape[0], -1)
        recon, z = jax.vmap(model)(imgs_flat)

        labels_np = jnp.array(labels)
        z_np = jnp.array(z)
        imgs_np = jnp.array(imgs_flat)

        all_z.append(z_np)
        all_labels.append(labels_np)

    z = jnp.vstack(all_z)
    labels = jnp.concatenate(all_labels)

    # how separated clusters are
    knn_acc = evaluate_latent_space_knn(z, labels)

    # mmd in latent space between source and target class
    source_z = z[labels == source_class]
    target_z = z[labels == target_class]

    n_min = min(len(source_z), len(target_z), 1000)  # limit to 1000 samples for MMD computation

    mmd_latent = float(MMD(jnp.array(source_z[:n_min]), jnp.array(target_z[:n_min]), kernel="rbf"))

    return {
        "knn_acc": knn_acc,
        "mmd_latent": mmd_latent,
    }


def objective(trial):
    arch_presets = {
        "small2": [64, 32, 16, 16],
        "small": [128, 64, 32, 16],
        "medium": [256, 128, 64, 32],
        "medium2": [256, 256, 128, 64],
        "large": [512, 256, 128, 64],
        "large2": [512, 512, 256, 128],
    }

    arch_name = trial.suggest_categorical(
        "arch", ["small2", "small", "medium", "large", "large2", "medium2"]
    )  # try different architectures
    hidden_dim = arch_presets[arch_name]
    lr = trial.suggest_float(
        "lr", 1e-4, 1e-2, log=True
    )  # try different learning rates in the log domain within the domain
    lambda_l2 = trial.suggest_float(
        "lambda_l2", 1e-5, 1e-1, log=True
    )  # try regulariztion rates in the log domain within the domain

    key = jr.PRNGKey(trial.number)
    model = AEv2(latent_dim=2, hidden_dims=hidden_dim, key=key)
    optimizer = optax.adam(lr)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
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

    metrics = compute_bo_metrics(trained_model, val_loader, source_class=0, target_class=1)

    trial.set_user_attr("val_loss", float(val_loss))  # save the attributes for each trial
    trial.set_user_attr("knn_acc", float(metrics["knn_acc"]))
    trial.set_user_attr("mmd_latent", float(metrics["mmd_latent"]))

    return 0.4 * val_loss + 0.4 * metrics["mmd_latent"] + 0.2 * metrics["knn_acc"]


# ── Run ───────────────────────────────────────────────────────────────────────

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),  # stop early if val loss is below median of previous trials
    study_name="ae_full_optimization",
    storage="sqlite:///ae_bo.db",
    load_if_exists=True,
)

study.optimize(objective, n_trials=30)


best = study.best_trial

print(f"\nBest objective:  {best.value:.4f}")
print(f"\nBest hyperparameters:")
for k, v in best.params.items():
    print(f"  {k:15s}: {v}")

print(f"\nValidation metrics:")
print(f"  val_loss:           {best.user_attrs['val_loss']:.4f}")
print(f"  knn_acc:            {best.user_attrs['knn_acc']:.4f}")
print(f"  mmd_latent:         {best.user_attrs['mmd_latent']:.4f}")


print("\nRetraining best configuration on full training data...")
best_params = best.params
key = jr.PRNGKey(999)
best_model = AEv2(
    latent_dim=2,
    hidden_dim=best_params["hidden_dim"],
    key=key,
)
final_trainer = Trainer(
    model=best_model,
    optimizer=optax.adam(best_params["lr"]),
    lambda_l2=best_params["lambda_l2"],
)
final_model, _, final_test_loss = final_trainer.train(
    epochs=best_params["n_epochs"],
    val_split=0.0,
    model=best_model,
    model_name="ae_best_model_bo",
    learning_rate=best_params["lr"],
)
print(f"Final test loss: {final_test_loss:.4f}")
