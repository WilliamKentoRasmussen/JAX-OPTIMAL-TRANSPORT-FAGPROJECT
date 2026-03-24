from main_project.model import AEv2
import jax
import jax.numpy as jnp
import equinox as eqx
from main_project.data import getDataloaders
import optax
import jax.random as jr
import matplotlib.pyplot as plt

seed = 3456 
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key, 2)

epochs = 20
lambda_l2 = 0.5 # [0, 0.01, 10]
optimizer = optax.adam(learning_rate=1e-4)



def loss_fn(model, x):
    recon, z = jax.vmap(model)(x)
    
    recon_loss = jnp.mean(optax.losses.squared_error(recon, x))

    # z shape: (batch, 784)
    latent_l2 = jnp.mean(jnp.sum(z**2, axis=-1))
    # print(recon_loss)
    loss =  recon_loss + lambda_l2 * latent_l2
    return loss


# @eqx.filter_jit
def train_step(model, opt_state, x):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x)
    updates, opt_state = optimizer.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_array)
    ) 
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train(epochs = 20):

    train_dataloader, test_dataloader = getDataloaders()
    model = AEv2(key=jr.PRNGKey(0)) #(subkey)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    train_losses = []

    for epoch in range(epochs):
        epoch_loss = 0 
        num_batches = 0

        for imgs, _ in train_dataloader: 
            imgs = jnp.array(imgs.numpy())

            # img shape: (batch, 1, 28, 28)
            imgs = imgs.reshape(imgs.shape[0], -1)

            model, opt_state, loss = train_step(model, opt_state, imgs)
            
            epoch_loss += float(loss)
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches 
        train_losses.append(avg_loss)#epoch_loss)
        print(avg_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss = {epoch_loss}")


    return model, train_losses
    


if __name__ == "__main__":
    model, train_losses = print(train(5)[1])

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()

