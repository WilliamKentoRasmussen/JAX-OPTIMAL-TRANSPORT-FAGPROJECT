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
from tqdm import tqdm

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




def train(epochs=20, val_split=0.2, model = AEv2(key=jr.PRNGKey(0)), model_name="ae_best_model"):
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
    save(model=model, name=model_name)
    df = pd.DataFrame(history)
    df.to_csv("training_history.csv", index=False)
    
    return model, history, test_loss


@eqx.filter_jit
def classifier_loss_fn(model, x, labels):
    # model outputs log_softmax, so we use nll loss
    log_probs = jax.vmap(model)(x)           # [B, 10]
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(log_probs, labels)
    return jnp.mean(loss)



@eqx.filter_jit
def classifier_train_step(model, opt_state, x, labels):
    loss, grads = eqx.filter_value_and_grad(classifier_loss_fn)(model, x, labels)
    updates, opt_state = optimizer.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_array)
    )
    return eqx.apply_updates(model, updates), opt_state, loss

def train_classifier(
    epochs=20,
    val_split=0.2,
    model=None,
    model_name="classifier",
):
    training_data, test_data = getData()

    num_train = len(training_data)
    indices   = np.arange(num_train)
    split     = int(num_train * (1 - val_split))
    train_loader = getDataloader(Subset(training_data, indices[:split]))
    val_loader   = getDataloader(Subset(training_data, indices[split:]))
    test_loader  = getDataloader(test_data)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    history   = []

    for epoch in tqdm(range(epochs)):

        epoch_loss, num_batches = 0.0, 0
        for imgs, labels in train_loader:
            imgs   = jnp.array(imgs.numpy())
            labels = jnp.array(labels.numpy())
            model, opt_state, loss = classifier_train_step(model, opt_state, imgs, labels)
            epoch_loss += float(loss)
            num_batches += 1
        avg_train_loss = epoch_loss / num_batches

        val_loss_total, val_batches = 0.0, 0
        for imgs, labels in val_loader:
            imgs   = jnp.array(imgs.numpy())
            labels = jnp.array(labels.numpy())
            val_loss_total += float(classifier_loss_fn(model, imgs, labels))
            val_batches += 1
        avg_val_loss = val_loss_total / val_batches

        history.append({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f"Epoch {epoch+1}/{epochs} — train: {avg_train_loss:.4f}, val: {avg_val_loss:.4f}")

    test_loss, test_batches = 0.0, 0
    for imgs, labels in test_loader:
        imgs   = jnp.array(imgs.numpy())
        labels = jnp.array(labels.numpy())
        test_loss += float(classifier_loss_fn(model, imgs, labels))
        test_batches += 1
    print(f"Test loss: {test_loss / test_batches:.4f}")

    save(model=model, name=model_name)
    pd.DataFrame(history).to_csv(model_name + "_training_history.csv", index=False)
    return model, history, test_loss / test_batches


if __name__ == "__main__":
    model,fold_results,test_results = train(epochs=1000,val_split=0.2)

    # print("starting training evaluation classifier")
    # model, history, test_loss = train_classifier(
    #     epochs=50,
    #     val_split=0.2,
    #     model=model,
    #     model_name="evaluate_classifier",
    # )
