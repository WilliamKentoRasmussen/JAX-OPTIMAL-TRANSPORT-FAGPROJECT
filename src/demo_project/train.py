#%%
import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree, Bool  # https://github.com/google/jaxtyping
from data import get_dataloaders
from model import CNN,AE
from utils import loss, cross_entropy, loss_AE
from evaluate import compute_accuracy, evaluate, evaluate_AE
import matplotlib.pyplot as plt 



# Hyperparameters

BATCH_SIZE = 64
LEARNING_RATE = 1e-3#3e-4
STEPS = 1000
PRINT_EVERY = 100
SEED = 5678

key = jax.random.PRNGKey(SEED)
key, subkey = jax.random.split(key, 2)

"""
dummy_x, dummy_y = next(iter(trainloader))
loss_value = loss(model, dummy_x, dummy_y)
print(loss_value.shape)  # scalar loss
# Example inference
output = jax.vmap(model)(dummy_x)
print(output.shape)  # batch of predictions
"""

#%%
optim = optax.adamw(LEARNING_RATE)

def train(
    model: CNN,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> CNN:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: CNN,
        opt_state: PyTree,
        x: Float[Array, "batch 1 28 28"],
        y: Int[Array, " batch"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss, test_accuracy = evaluate(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}, test_accuracy={test_accuracy.item()}"
            )
    return model

#%%

def train_AE(
    model: AE,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    steps: int,
    print_every: int,
) -> AE:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: AE,
        opt_state: PyTree,
        x: Float[Array, "784"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_AE)(model, x)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    # Loop over our training dataset as many times as we need.
    def infinite_trainloader():
        while True:
            yield from trainloader

    for step, (x, y) in zip(range(steps), infinite_trainloader()):
        # PyTorch dataloaders give PyTorch tensors by default,
        # so convert them to NumPy arrays.
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = make_step(model, opt_state, x)
        if (step % print_every) == 0 or (step == steps - 1):
            test_loss = evaluate_AE(model, testloader)
            print(
                f"{step=}, train_loss={train_loss.item()}, "
                f"test_loss={test_loss.item()}"
            )
    return model



USE_CNN: bool = False

if __name__ == "__main__":
    trainloader, testloader = get_dataloaders(batch_size=64)

    if USE_CNN:
        model = CNN(subkey)
        model = train(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)
        test_loss, test_accuracy = evaluate(model, testloader)
        print(f"Final test loss = {test_loss}, accuracy = {test_accuracy}")
    else:
        model = AE(subkey)
        model = train_AE(model, trainloader, testloader, optim, STEPS, PRINT_EVERY)
        test_loss = evaluate_AE(model, testloader)

        images, _ = next(iter(testloader))
        images_np = images.numpy().reshape(-1, 784)
        reconstructed = jax.vmap(model)(images_np)

        fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10, 3))
        for i in range(10):
            axes[0, i].imshow(images_np[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(jnp.array(reconstructed[i]).reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
        plt.show()

        print(f"\n\nFinal test loss = {test_loss}")