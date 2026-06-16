import json
from main_project.model import AEv2
import equinox as eqx
import jax
import jax.random as jr
import os

_ACTIVATION_FNS = {
    "leaky_relu": jax.nn.leaky_relu,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "elu": jax.nn.elu,
}

hyperparamters = {}


def save(model, name="model", path="models"):
    filename = path + "/" + name + ".eqx"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(name="model", path="models", model=None, latent_dim=2):
    if model is None:
        model = AEv2(key=jr.PRNGKey(0), latent_dim=latent_dim)
    filename = path + "/" + name + ".eqx"
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


def save_with_hyperparams(model, hidden_dims, latent_dim, activation_name="leaky_relu", filename="model", path="models"):
    filepath = path + "/" + filename + ".eqx"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        hyperparam_str = json.dumps({"hidden_dims": hidden_dims, "latent_dim": latent_dim, "activation": activation_name})
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_with_hyperparams(name="model", path="models"):
    filename = path + "/" + name + ".eqx"
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        activation = _ACTIVATION_FNS[hyperparams.get("activation", "leaky_relu")]
        model = AEv2(
            key=jr.PRNGKey(0),
            latent_dim=hyperparams["latent_dim"],
            hidden_dims=hyperparams["hidden_dims"],
            activation=activation,
        )
        return eqx.tree_deserialise_leaves(f, model)


if __name__ == "__main__":
    model, train_losses = train(1)
    save("models/model.eqx", model)
    newmodel = load("models/model.eqx")
    assert model.encoder.layers[0].weight[2, 2] == newmodel.encoder.layers[0].weight[2, 2]
