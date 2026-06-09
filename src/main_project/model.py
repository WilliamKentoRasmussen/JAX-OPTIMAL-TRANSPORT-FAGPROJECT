import equinox as eqx
import jax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping


class AEv2(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(self, key, latent_dim=2):
        super().__init__()
        key_splt = jax.random.split(key, 10)

        self.encoder = eqx.nn.Sequential(
            (
                eqx.nn.Linear(28 * 28, 256, key=key_splt[0]),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                eqx.nn.Linear(256, 128, key=key_splt[1]),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                eqx.nn.Linear(128, 64, key=key_splt[2]),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                eqx.nn.Linear(64, 32, key=key_splt[3]),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                eqx.nn.Linear(32, latent_dim, key=key_splt[4]),
            )
        )

        self.decoder = eqx.nn.Sequential(
            (
                eqx.nn.Linear(latent_dim, 32, key=key_splt[5]),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                eqx.nn.Linear(32, 64, key=key_splt[6]),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                eqx.nn.Linear(64, 128, key=key_splt[7]),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                eqx.nn.Linear(128, 256, key=key_splt[8]),
                eqx.nn.Lambda(jax.nn.leaky_relu),
                eqx.nn.Linear(256, 28 * 28, key=key_splt[9]),
                eqx.nn.Lambda(jax.nn.sigmoid),
            )
        )

    def __call__(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 300
PRINT_EVERY = 30


class targetClassifier(eqx.Module):
    CNN: eqx.Module

    def __init__(self, key):

        super().__init__()

        key_split = jax.random.split(key, 4)

        self.CNN = eqx.nn.Sequential(
            (
                eqx.nn.Conv2d(1, 3, kernel_size=4, key=key_split[0]),
                eqx.nn.MaxPool2d(kernel_size=2),
                eqx.nn.Lambda(jax.nn.relu),      
                eqx.nn.Lambda(jnp.ravel),        
                eqx.nn.Linear(1728, 512, key=key_split[1]),
                eqx.nn.Lambda(jax.nn.sigmoid),   
                eqx.nn.Linear(512, 64, key=key_split[2]),
                eqx.nn.Lambda(jax.nn.relu),      
                eqx.nn.Linear(64, 10, key=key_split[3]),
                eqx.nn.Lambda(jax.nn.log_softmax),  
            )
        )

    def __call__(self,x):
        #Train function flattens images, hence we have to restore the shape for the CNN.
        x = x.reshape(1, 28, 28)
        return self.CNN(x)

if __name__ == "__main__":
    seed = 3456
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key, 2)
    model = AEv2(subkey)
    x = jax.random.randint(subkey, (784,), 10, 10)
    output = model(x)[0].numpy()
    print(f"Output shape of model: {output.shape}")
