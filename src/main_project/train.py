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
def val_step(model,x):
    loss = loss_fn(model,x)
    return loss





def train(epochs=20, k=10):
    training_data, test_data = getData()
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(training_data)):
        print(f"=== Fold {fold+1}/{k} ===")
        
        # Reset model and optimizer per fold
        model = AEv2(key=jr.PRNGKey(0))
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        
        # Subsets and dataloaders
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
                
                loss = val_step(model, imgs)  # implement val_step similar to train_step but without optimizer
                val_loss += float(loss)
                num_val_batches += 1
            
            avg_val_loss = val_loss / num_val_batches
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
        
        # Save model for this fold
        eqx.tree_serialise_leaves(f"ae_model_fold{fold+1}.eqx", model)
        
        # Store fold validation result
        fold_results.append(avg_val_loss)
    
    # Print cross-validated performance
    print(f"Average validation loss across {k} folds: {sum(fold_results)/k:.4f}")


if __name__ == "__main__":
    model, train_losses = train(epochs=10)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
