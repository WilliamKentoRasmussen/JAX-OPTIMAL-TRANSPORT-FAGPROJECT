import torch
import numpy as np
from main_project.evaluateClassifier import plot_transport_images
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
def MMD(x: Array, y: Array, kernel):
    
    xx, yy, zz = jnp.matmul(x, x.T), jnp.matmul(y, y.T), jnp.matmul(x, y.T)

    rx = jnp.diag(xx)[jnp.newaxis, :]  # (1, N)
    ry = jnp.diag(yy)[jnp.newaxis, :]  # (1, M)

    #This is the expanded exponential kernel - XX corresponding to element x_i times x_i 
    dxx = rx.T + rx - 2. * xx
    dyy = ry.T + ry - 2. * yy 
    dxy = rx.T + ry - 2. * zz 


    XX, YY, XY = (jnp.zeros_like(xx),
                  jnp.zeros_like(xx),
                  jnp.zeros_like(xx))


    #Turns distances into similarity scores betweem the distributions
    if kernel == "multiscale":

        #The standard devisation is unkown, so having multiple different bandwidth makes the test sentitive to multiple cases
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)


    #MMD² = E[k(x,x')] + E[k(y,y')] − 2·E[k(x,y)]
    return jnp.mean(XX + YY - 2. * XY)


if __name__ == "__main__":

    source_img, target_img  = np.load("data/original_images.npy"), np.load("data/expected_target_images.npy")
    intermediate_images = np.load(
            "data/intermediate_images.npy"
        )
    intermediate_images = intermediate_images.transpose(1, 0, 2) #Corrects order

    result = MMD(jnp.asarray(source_img), jnp.asarray(target_img), kernel="multiscale")

    print(f"MMD result of X and Y is {result.item()}")

    fractions = [0.25, 0.5, 0.75, 1.0]
    for frac, imgs in zip(fractions, intermediate_images):

        print(f"t = {frac}")
        mmd = MMD(jnp.asarray(imgs), jnp.asarray(target_img), kernel="multiscale")

        print(f"MMD result of fraction {frac} and target is {mmd.item()}")
        plot_transport_images(imgs, target_img, n = 5, title = f"MMD score of {mmd.item()}")

