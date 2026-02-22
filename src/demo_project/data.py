import jax.numpy as jnp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping


def numpy_collate(batch):
    """Converts a batch of torch tensors to jax numpy arrays."""
    if isinstance(batch[0], (list, tuple)):
        return [numpy_collate(samples) for samples in zip(*batch)]
    return jnp.array(batch)


def get_dataloaders(batch_size=64):
    normalise_data = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_dataset = torchvision.datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=normalise_data,
    )
    test_dataset = torchvision.datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=normalise_data,
    )
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader


"""

    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=numpy_collate)

    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=numpy_collate)
    
    return train_loader, test_loader

    """
