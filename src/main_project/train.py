from main_project.model import AEv2
import jax
import jax.numpy as jnp
import equinox as eqx
from main_project.data import getDataloader,getData
import optax
import jax.random as jr
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from torch.utils.data import DataLoader, Subset
from main_project.utils import save

seed = 3456
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key, 2)

epochs = 20
lambda_l2 = 0.5  # [0, 0.01, 10]
optimizer = optax.adam(learning_rate=1e-4)


def loss_fn(model, x):
    recon, z = jax.vmap(model)(x)

    recon_loss = jnp.mean(optax.losses.squared_error(recon, x))

    # z shape: (batch, 784)
    latent_l2 = jnp.mean(jnp.sum(z**2, axis=-1))
    # print(recon_loss)
    loss = recon_loss + lambda_l2 * latent_l2
    return loss


# @eqx.filter_jit
def train_step(model, opt_state, x):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x)
    updates, opt_state = optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
def val_step(model, x):
    return loss_fn(model, x)
def evaluate(model, dataloader):
    total_loss = 0
    num_batches = 0

    for imgs, _ in dataloader:
        imgs = jnp.array(imgs.numpy())
        imgs = imgs.reshape(imgs.shape[0], -1)

        loss = loss_fn(model, imgs)
        total_loss += float(loss)
        num_batches += 1

    return total_loss / num_batches





import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd
from torch.utils.data import Subset
from sklearn.model_selection import KFold

def train(epochs=20, k=10):
    training_data, test_data = getData()
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    best_model = None
    best_score = float("inf")
    test_loader = getDataloader(test_data)

    fold_results = []
    test_results = []
    history = []  

    for fold, (train_idx, val_idx) in enumerate(cv.split(training_data)):
        print(f"=== Fold {fold+1}/{k} ===")
        
        # Initialize model and optimizer
        model = AEv2(key=jr.PRNGKey(0))
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        
        # Create subsets and dataloaders
        train_subset = Subset(training_data, train_idx)
        val_subset   = Subset(training_data, val_idx)
        train_loader = getDataloader(train_subset)
        val_loader   = getDataloader(val_subset)
        
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # --- Training ---
            epoch_loss = 0
            num_batches = 0
            for imgs, _ in train_loader:
                imgs = jnp.array(imgs.numpy())
                imgs = imgs.reshape(imgs.shape[0], -1)
                
                model, opt_state, loss = train_step(model, opt_state, imgs)
                
                epoch_loss += float(loss)
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # --- Validation ---
            val_loss = 0
            num_val_batches = 0
            for imgs, _ in val_loader:
                imgs = jnp.array(imgs.numpy())
                imgs = imgs.reshape(imgs.shape[0], -1)
                
                loss = val_step(model, imgs)  
                val_loss += float(loss)
                num_val_batches += 1
            
            avg_val_loss = val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            
            # --- Log epoch for DataFrame ---
            history.append({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })
            
            print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
        
        # --- Store fold-level results ---
        fold_results.append(avg_val_loss)
        test_loss = evaluate(model, test_loader)
        test_results.append(test_loss)
        if test_loss < best_score:
            best_score = test_loss
            best_model = model

    # Save best model
    save(model=best_model, name="ae_best_model")

    # Convert history to DataFrame and save
    df = pd.DataFrame(history)
    df.to_csv("training_history.csv", index=False)

    print(f"Average validation loss across {k} folds: {sum(fold_results)/k:.4f}")
    
    return fold_results, test_results
    


if __name__ == "__main__":
    fold_results,test_results = train(epochs=200,k=5)

    plt.figure(figsize=(8, 5))
    plt.plot(fold_results, label="Train loss")
    plt.xlabel("Fold")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
