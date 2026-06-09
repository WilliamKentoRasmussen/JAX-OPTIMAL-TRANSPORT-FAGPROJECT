import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
import torch  # https://pytorch.org
import torchvision  # https://pytorch.org
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

from main_project.train import train_classifier
from main_project.utils import load
import numpy as np
import matplotlib.pyplot as plt
# Hyperparameters

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 300
PRINT_EVERY = 30
SEED = 5678

key = jax.random.PRNGKey(SEED)

# print("starting training")
# model, history, test_loss = train_classifier(
#     epochs=50,
#     val_split=0.2,
#     model=model,
#     model_name="evaluate_classifier",
# )

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


key, subkey = jax.random.split(key, 2)
model = load(name="evaluate_classifier", path="models", model= targetClassifier(subkey))



def plot_transport_images(original_images, expected_target_images, n=5, title = "Transport plot"):
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))

    for i in range(n):
        axes[0, i].imshow(original_images[i].reshape(28, 28), cmap="gray")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(expected_target_images[i].reshape(28, 28), cmap="gray")
        axes[1, i].axis("off")
    
    axes[0, 0].set_ylabel("Source (0)", fontsize=12)
    axes[1, 0].set_ylabel("Transported (1)", fontsize=12)

    plt.suptitle(f"OT Transport: digit 0 -> digit 1 - with {title} ")
    plt.tight_layout()
    plt.show()

def classifier_confidence(transported_images, classifier, target_class=1):
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
    


    classifier_confidence(expected_target_images, model, 1)
    plot_transport_images(original_images, expected_target_images)

    fractions = [0.25, 0.5, 0.75, 1.0]

    for frac, imgs in zip(fractions, intermediate_images):
        print(f"t = {frac}")
        classifier_confidence(imgs, model, 1)
        plot_transport_images(imgs, expected_target_images)


    



