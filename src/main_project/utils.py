import json
from main_project.model import AEv2
from main_project.train import train
import equinox as eqx
import jax.random as jr
import os
hyperparamters = {}

def save(model, name = "model", path = "models"):
    filename = path + "/" + name + ".eqx"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(name = "model",path="models"):
    filename = path + "/" + name + ".eqx"
    with open(filename, "rb") as f:
        
        model = AEv2(key=jr.PRNGKey(0))
        return eqx.tree_deserialise_leaves(f, model)

def save_with_hyperparams(filename, hyperparams, model):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load_with_hyperparams(filename):
    with open(filename, "rb") as f:
        hyperparams = json.loads(f.readline().decode())
        model = AEv2(key=jr.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)

if __name__ == "__main__":

    model, train_losses = train(1)
    save("models/model.eqx", model)
    newmodel = load("models/model.eqx")
    assert model.encoder.layers[0].weight[2, 2] == newmodel.encoder.layers[0].weight[2, 2]