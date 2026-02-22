#%%
import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
from data import get_dataloaders
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


SEED = 5678

key = jax.random.PRNGKey(SEED)


class CNN(eqx.Module):
    layers: list

    def __init__(self, key):
        key1, key2, key3, key4 = jax.random.split(key, 4)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.
        self.layers = [
            eqx.nn.Conv2d(1, 3, kernel_size=4, key=key1),
            eqx.nn.MaxPool2d(kernel_size=2),
            jax.nn.relu,
            jnp.ravel,
            eqx.nn.Linear(1728, 512, key=key2),
            jax.nn.sigmoid,
            eqx.nn.Linear(512, 64, key=key3),
            jax.nn.relu,
            eqx.nn.Linear(64, 10, key=key4),
            jax.nn.log_softmax,
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x

#https://www.geeksforgeeks.org/deep-learning/implementing-an-autoencoder-in-pytorch/
class AE(eqx.Module):
    encoder_layers: list
    decoder_layers: list

    def __init__(self,key):

        key1, key2, key3, key4, key5, key6, key7, key8, key9, key10, = jax.random.split(key, 10)
        
        self.encoder_layers = [
            jnp.ravel,
            eqx.nn.Linear(28*28, 128, key=key1),
            jax.nn.relu,
            eqx.nn.Linear(128, 64, key=key2),
            jax.nn.relu,
            eqx.nn.Linear(64, 36, key =key3),
            jax.nn.relu,
            eqx.nn.Linear(36, 18, key =key4),
            jax.nn.relu,
            eqx.nn.Linear(18, 9, key=key5),
        ]

        self.decoder_layers = [
            eqx.nn.Linear(9, 18, key=key6),
            jax.nn.relu,
            eqx.nn.Linear(18, 36, key=key7),
            jax.nn.relu,
            eqx.nn.Linear(36, 64, key =key8),
            jax.nn.relu,
            eqx.nn.Linear(64, 128, key =key9),
            jax.nn.relu,
            eqx.nn.Linear(128, 28*28, key=key10),
            #jax.nn.sigmoid,
        ]

    def __call__(self, x: Float[Array, "784"]) -> Float[Array, "784"]:
        
        for layer in self.encoder_layers:
            x = layer(x)

        for layer in self.decoder_layers:
            x = layer(x)
        
        return x





key, subkey = jax.random.split(key, 2) # 2 number of new keys
model = CNN(subkey)