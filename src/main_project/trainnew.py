from pyexpat import model
import re

from main_project.model import AEv2
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from main_project.data import getDataloader, getData
import optax
import pandas as pd
import jax.random as jr
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from torch.utils.data import DataLoader, Subset
from main_project.train import classifier_loss_fn, val_step
from main_project.utils import save
from tqdm import tqdm
import re


class AETrainer:
    def __init__(self, model, learning_rate, lambda_l2):
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.lambda_l2 = lambda_l2
        # self.loss_name = loss_name

    # def loss_fn(self,model, x):
    #     recon, z = jax.vmap(model)(x)

    #     if self.loss_name == "mse":
    #         # standard — penalizes large errors heavily
    #         recon_loss = jnp.mean((recon - x)**2)

    #     elif self.loss_name == "mae":
    #         # more robust to outliers — treats all errors equally
    #         recon_loss = jnp.mean(jnp.abs(recon - x))

    #     elif self.loss_name == "bce":
    #         # binary cross entropy — natural for pixel values in [0,1]
    #         # your decoder uses sigmoid so this is the correct paired loss
    #         recon_loss = jnp.mean(
    #             -x * jnp.log(recon + 1e-8) - (1 - x) * jnp.log(1 - recon + 1e-8)
    #         )

    #     latent_l2 = jnp.mean(jnp.sum(z**2, axis=-1))
    #     return recon_loss + self.lambda_l2 * latent_l2
    @eqx.filter_jit
    def loss_fn(self, model, x):
        recon, z = jax.vmap(model)(x)
        recon_loss = jnp.mean((recon - x) ** 2)
        latent_l2 = jnp.mean(jnp.sum(z**2, axis=-1))
        return recon_loss + self.lambda_l2 * latent_l2

    @eqx.filter_jit
    def train_step(self, model, opt_state, x):
        loss, grads = eqx.filter_value_and_grad(self.loss_fn)(model, x)
        updates, opt_state = self.optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    @eqx.filter_jit
    def val_step(self, model, x):
        return self.loss_fn(model, x)

    def evaluate(self, model, dataloader):
        total_loss = 0
        num_batches = 0
        for imgs, _ in dataloader:
            imgs = jnp.array(imgs.numpy())
            imgs = imgs.reshape(imgs.shape[0], -1)
            loss = self.loss_fn(model, imgs)
            total_loss += float(loss)
            num_batches += 1
        return total_loss / num_batches

    def train(self, epochs=20, val_split=0.2, model=AEv2(key=jr.PRNGKey(0)), model_name="ae_best_model",
              patience=20, batch_size=64):
        training_data, test_data = getData()
        print(f"training data is {type(training_data)}")
        num_train = len(training_data)
        indices = np.arange(num_train)
        split = int(num_train * (1 - val_split))
        train_idx, val_idx = indices[:split], indices[split:]

        train_subset = Subset(training_data, train_idx)
        val_subset = Subset(training_data, val_idx)

        train_loader = getDataloader(train_subset, batch_size=batch_size)
        val_loader = getDataloader(val_subset, batch_size=batch_size)
        test_loader = getDataloader(test_data, batch_size=batch_size)

        # Initialize model and optimizer
        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))

        history = []  # store all epoch info

        best_val_loss = float("inf")
        best_model = model
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # --- Training ---
            epoch_loss = 0
            num_batches = 0
            for imgs, _ in train_loader:
                imgs = jnp.array(imgs.numpy()).reshape(imgs.shape[0], -1)
                model, opt_state, loss = self.train_step(model, opt_state, imgs)
                epoch_loss += float(loss)
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches

            # --- Validation ---
            val_loss_total = 0
            val_batches = 0
            for imgs, _ in val_loader:
                imgs = jnp.array(imgs.numpy()).reshape(imgs.shape[0], -1)
                loss = self.val_step(model, imgs)
                val_loss_total += float(loss)
                val_batches += 1

            avg_val_loss = val_loss_total / val_batches

            # --- Save history ---
            history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

            print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

            # --- Early stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

        model = best_model

        # --- Evaluate on test set ---
        test_loss = self.evaluate(model, test_loader)
        print(f"Test loss: {test_loss:.4f}")

        if re.match(r"ae_best_model_bo_\d+", model_name):
            df = pd.DataFrame(history)
            df.to_csv(f"training_history_{model_name}.csv", index=False)

        return model, history, test_loss


class ClassifierTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    @eqx.filter_jit
    def classifier_loss_fn(self, model, x, labels):
        # model outputs log_softmax, so we use nll loss
        log_probs = jax.vmap(model)(x)  # [B, 10]
        loss = optax.losses.softmax_cross_entropy_with_integer_labels(log_probs, labels)
        return jnp.mean(loss)

    @eqx.filter_jit
    def classifier_train_steps(self, model, opt_state, x, labels):
        loss, grads = eqx.filter_value_and_grad(self.classifier_loss_fn)(model, x, labels)
        updates, opt_state = self.optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    def train_classifier(
        self,
        epochs=20,
        val_split=0.2,
        model=None,
        model_name="classifier",
    ):
        training_data, test_data = getData()

        num_train = len(training_data)
        indices = np.arange(num_train)
        split = int(num_train * (1 - val_split))
        train_loader = getDataloader(Subset(training_data, indices[:split]))
        val_loader = getDataloader(Subset(training_data, indices[split:]))
        test_loader = getDataloader(test_data)

        opt_state = self.optimizer.init(eqx.filter(model, eqx.is_array))
        history = []

        for epoch in tqdm(range(epochs)):
            epoch_loss, num_batches = 0.0, 0
            for imgs, labels in train_loader:
                imgs = jnp.array(imgs.numpy())
                labels = jnp.array(labels.numpy())
                model, opt_state, loss = self.classifier_train_steps(model, opt_state, imgs, labels)
                epoch_loss += float(loss)
                num_batches += 1
            avg_train_loss = epoch_loss / num_batches

            val_loss_total, val_batches = 0.0, 0
            for imgs, labels in val_loader:
                imgs = jnp.array(imgs.numpy())
                labels = jnp.array(labels.numpy())
                val_loss_total += float(self.classifier_loss_fn(model, imgs, labels))
                val_batches += 1
            avg_val_loss = val_loss_total / val_batches

            history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
            print(f"Epoch {epoch+1}/{epochs} — train: {avg_train_loss:.4f}, val: {avg_val_loss:.4f}")

        test_loss, test_batches = 0.0, 0
        for imgs, labels in test_loader:
            imgs = jnp.array(imgs.numpy())
            labels = jnp.array(labels.numpy())
            test_loss += float(self.classifier_loss_fn(model, imgs, labels))
            test_batches += 1
        print(f"Test loss: {test_loss / test_batches:.4f}")

        save(model=model, name=model_name)
        pd.DataFrame(history).to_csv(model_name + "_training_history.csv", index=False)
        return model, history, test_loss / test_batches
