import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

from main_project.train import train_classifier
from main_project.model import targetClassifier
from main_project.visualize import plot_transport_images
from main_project.utils import load
import numpy as np
import matplotlib.pyplot as plt
# Hyperparameters






def classifier_confidence(transported_images, target_class=1):

    SEED = 5678

    key = jax.random.PRNGKey(SEED)
    key, subkey = jax.random.split(key, 2)

    classifier = load(name="evaluate_classifier", path="models", model= targetClassifier(subkey))
    x = jnp.asarray(transported_images)
    
    log_probs = jax.vmap(classifier)(x)
    probs = jnp.exp(log_probs)  # Shape: [n, 10]
    
    
    p_target = probs[:, target_class]
    predictions = jnp.argmax(probs, axis=-1)
    classified = jnp.mean(predictions == target_class)
    

    print(f"Mean P(class={target_class}):  {float(jnp.mean(p_target)):.20f}")
    print(f"Fraction class {target_class}:  {float(classified):.20f}")
    
    return p_target


if __name__ == "__main__":
    
    original_images, expected_target_images  = np.load("data/original_images.npy"), np.load("data/expected_target_images.npy")
    intermediate_images = np.load(
            "data/intermediate_images.npy"
        )
    intermediate_images = intermediate_images.transpose(1, 0, 2) #Corrects order
    


    classifier_confidence(expected_target_images, 1)
    plot_transport_images(original_images, expected_target_images)

    fractions = [0.25, 0.5, 0.75, 1.0]

    for frac, imgs in zip(fractions, intermediate_images):
        print(f"t = {frac}")
        classifier_confidence(imgs, 1)
        plot_transport_images(imgs, expected_target_images)


    



