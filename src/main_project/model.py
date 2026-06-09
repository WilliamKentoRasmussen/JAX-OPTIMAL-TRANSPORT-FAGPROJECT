import equinox as eqx
import jax
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping


class AEv2(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module

    def __init__(
        self,
        key,
        latent_dim: int = 2,
        hidden_dims: list = [256, 128, 64, 32],   # encoder dims, decoder is reversed
        activation = jax.nn.leaky_relu,
        output_activation = jax.nn.sigmoid,
    ):
        key_splits = jax.random.split(key, 2 * (len(hidden_dims) + 1))

        # ── Encoder: 784 → hidden_dims → latent_dim ──────────────────────────
        encoder_layers = []
        in_dim = 28 * 28
        for i, out_dim in enumerate(hidden_dims):
            encoder_layers.append(eqx.nn.Linear(in_dim, out_dim, key=key_splits[i]))
            encoder_layers.append(eqx.nn.Lambda(activation))
            in_dim = out_dim
        
        encoder_layers.append(
            eqx.nn.Linear(in_dim, latent_dim, key=key_splits[len(hidden_dims)])
        )
        self.encoder = eqx.nn.Sequential(tuple(encoder_layers))

        # ── Decoder: latent_dim → hidden_dims reversed → 784 ─────────────────
        decoder_dims = list(reversed(hidden_dims))
        decoder_layers = []
        in_dim = latent_dim
        offset = len(hidden_dims) + 1
        for i, out_dim in enumerate(decoder_dims):
            decoder_layers.append(eqx.nn.Linear(in_dim, out_dim, key=key_splits[offset + i]))
            decoder_layers.append(eqx.nn.Lambda(activation))
            in_dim = out_dim
        # final projection to 784
        decoder_layers.append(
            eqx.nn.Linear(in_dim, 28 * 28, key=key_splits[offset + len(decoder_dims)])
        )
        decoder_layers.append(eqx.nn.Lambda(output_activation))
        self.decoder = eqx.nn.Sequential(tuple(decoder_layers))

    def __call__(self, x):
        z   = self.encoder(x)
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

    def __call__(self, x):
        # Train function flattens images, hence we have to restore the shape for the CNN.
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
