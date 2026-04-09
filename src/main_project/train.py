from main_project.model import AEv2
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from main_project.data import getDataloader,getData
import optax
import pandas as pd
import jax.random as jr
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from torch.utils.data import DataLoader, Subset
from main_project.utils import save

seed = 3456
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key, 2)

lambda_l2 = 0.001  # [0, 0.01, 10]
optimizer = optax.adam(learning_rate=1e-4)

@eqx.filter_jit
def loss_fn(model, x):
    recon, z = jax.vmap(model)(x)

    recon_loss = jnp.mean(optax.losses.squared_error(recon, x))

    # z shape: (batch, 784)
    latent_l2 = jnp.mean(jnp.sum(z**2, axis=-1))
    # print(recon_loss)
    loss = recon_loss + lambda_l2 * latent_l2
    return loss


@eqx.filter_jit
def train_step(model, opt_state, x):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x)
    updates, opt_state = optimizer.update(grads, opt_state, params=eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss
@eqx.filter_jit
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




def train(epochs=20, val_split=0.2):
    # Load data
    training_data, test_data = getData()
    print(f"training data is {type(training_data)}")
    num_train = len(training_data)
    indices = np.arange(num_train)
    split = int(num_train * (1 - val_split))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_subset = Subset(training_data, train_idx)
    val_subset   = Subset(training_data, val_idx)
    
    train_loader = getDataloader(train_subset)
    val_loader   = getDataloader(val_subset)
    test_loader  = getDataloader(test_data)

    # Initialize model and optimizer
    model = AEv2(key=jr.PRNGKey(0))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    history = []  # store all epoch info
    
    
    for epoch in range(epochs):
        # --- Training ---
        epoch_loss = 0
        num_batches = 0
        for imgs, _ in train_loader:
            imgs = jnp.array(imgs.numpy()).reshape(imgs.shape[0], -1)
            model, opt_state, loss = train_step(model, opt_state, imgs)
            epoch_loss += float(loss)
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        
        # --- Validation ---
        val_loss_total = 0
        val_batches = 0
        for imgs, _ in val_loader:
            imgs = jnp.array(imgs.numpy()).reshape(imgs.shape[0], -1)
            loss = val_step(model, imgs)
            val_loss_total += float(loss)
            val_batches += 1
        
        avg_val_loss = val_loss_total / val_batches
        
        # --- Save history ---
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })
        
        print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
        
        # --- Save best model ---
        
    
    # --- Evaluate on test set ---
    test_loss = evaluate(model, test_loader)
    print(f"Test loss: {test_loss:.4f}")
    
    # Save best model and training history
    save(model=model, name="ae_best_model")
    df = pd.DataFrame(history)
    df.to_csv("training_history.csv", index=False)
    
    return model, history, test_loss
    


if __name__ == "__main__":
    model,fold_results,test_results = train(epochs=1000,val_split=0.2)
