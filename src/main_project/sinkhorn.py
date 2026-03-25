from typing import Union
import torch
import tqdm
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax



def sinkhorn_simple(s: jax.Array,
    d: jax.Array,
    C: jax.Array,
    gamma: float = 0.1,
    eps: float = 1e-3,
    max_iters: int = 100,
    stop_thresh: float = 1e-5,
    verbose=False,
):
    u, v = jnp.ones_like(s), jnp.ones_like(d)
    K = jnp.exp(-C/gamma)
    for i in range(max_iters):
        u_prev = u
        v_prev = v
        u = s / jnp.dot(K,v)
        v = d / jnp.dot(K.T,u)
        
        if jnp.max(jnp.abs(u_prev-u)) < stop_thresh and jnp.max(jnp.abs(v_prev-v)) < stop_thresh:
            break
    
    rows, columns = K.shape
    T = jnp.zeros((rows,columns))

    T = (u[:, None]) * K * (v[None, :])
    #for i in range(rows):
    #    for j in range(columns): 
    #        T[i][j] = u[i]*K[i][j]*v[j]
    
    return T

s = jnp.array([0.1,0.35, 0.40,0.15])
d = jnp.array([0.2,0.25, 0.30,0.25])

C = jnp.array([
    [0.3,5, 40,1],
    [0.2,35, 0.40,15],
    [0.5,45, 1,5],
    [0.1,2, 10,15]
])
    
if __name__ == "__main__":
    gamma = 0.2
    T = sinkhorn_simple(s,d,C,gamma=gamma) 
    print(T.sum(axis=1), " should equal:", s)
    print(T.sum(axis=0),  " should equal:", d)
    print(T)

@jax.jit
def cdist_euclidean_1D(x, y, p: int = 2):
    diff = y - x
    if p == 1:
        return jnp.sum(jnp.abs(diff))
    return jnp.sum(jnp.abs(diff) ** p) ** (1 / p)

@jax.jit
def cdist_euclidean(x, y):
    """Computes pairwise Euclidean distance between rows of x and y."""
    # (x - y)^2 = x^2 + y^2 - 2xy. Efficiently computed via broadcasting.
    return jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))




def sinkhorn_jax(x: jax.Array,
    y: jax.Array,
    p: float = 2,
    w_x: Union[jax.Array, None] = None,
    w_y: Union[jax.Array, None] = None,
    eps: float = 1e-3,
    max_iters: int = 100,
    stop_thresh: float = 1e-5,
    verbose=False,
):
    
    return 

def sinkhorn_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    p: float = 2,
    w_x: Union[torch.Tensor, None] = None,
    w_y: Union[torch.Tensor, None] = None,
    eps: float = 1e-3,
    max_iters: int = 100,
    stop_thresh: float = 1e-5,
    verbose=False,
):
    if not isinstance(p, int):
        raise TypeError(f"p must be an integer greater than 0, got {p}")
    if p <= 0:
        raise ValueError(f"p must be an integer greater than 0, got {p}")

    if eps <= 0:
        raise ValueError("Entropy regularization term eps must be > 0")

    if not isinstance(max_iters, int):
        raise TypeError(f"max_iters must be an integer > 0, got {max_iters}")
    if max_iters <= 0:
        raise ValueError(f"max_iters must be an integer > 0, got {max_iters}")

    if not isinstance(stop_thresh, float):
        raise TypeError(f"stop_thresh must be a float, got {stop_thresh}")

    if len(x.shape) != 2:
        raise ValueError(f"x must be an [n, d] tensor but got shape {x.shape}")
    if len(y.shape) != 2:
        raise ValueError(f"x must be an [m, d] tensor but got shape {y.shape}")
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"x and y must match in the last dimension (i.e. x.shape=[n, d], "
            f"y.shape[m, d]) but got x.shape = {x.shape}, y.shape={y.shape}"
        )

    if w_x is not None:
        if w_y is None:
            raise ValueError("If w_x is not None, w_y must also be not None")
        if len(w_x.shape) > 1:
            w_x = w_x.squeeze()
        if len(w_x.shape) != 1:
            raise ValueError(
                f"w_x must have shape [n,] or [n, 1] where x.shape = [n, d], but got w_x.shape = {w_x.shape}"
            )
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(
                f"w_x must match the shape of x in dimension 0 but got "
                f"x.shape = {x.shape} and w_x.shape = {w_x.shape}"
            )
    if w_y is not None:
        if w_x is None:
            raise ValueError("If w_y is not None, w_x must also be not None")
        if len(w_y.shape) > 1:
            w_y = w_y.squeeze()
        if len(w_y.shape) != 1:
            raise ValueError(
                f"w_y must have shape [n,] or [n, 1] where x.shape = [n, d], but got w_y.shape = {w_y.shape}"
            )
        if w_x.shape[0] != x.shape[0]:
            raise ValueError(
                f"w_y must match the shape of y in dimension 0 but got "
                f"y.shape = {y.shape} and w_y.shape = {w_y.shape}"
            )

    M = torch.cdist(x, y, p=p)

    if w_x is None and w_y is None:
        w_x = torch.ones(x.shape[0]).to(x) / x.shape[0]
        w_y = torch.ones(y.shape[0]).to(x) / y.shape[0]
        w_y *= w_x.shape[0] / w_y.shape[0]

    sum_w_x = w_x.sum().item()
    sum_w_y = w_y.sum().item()
    if abs(sum_w_x - sum_w_y) > 1e-5:
        raise ValueError(
            f"Weights w_x and w_y do not sum to the same value, "
            f"got w_x.sum() = {sum_w_x} and w_y.sum() = {sum_w_y} "
            f"(absolute difference = {abs(sum_w_x - sum_w_y)}"
        )

    log_a = torch.log(w_x)
    log_b = torch.log(w_y)

    u = torch.zeros_like(w_x)
    v = eps * torch.log(w_y)

    if verbose:
        pbar = tqdm.trange(max_iters)
    else:
        pbar = range(max_iters)

    for _ in pbar:
        u_prev = u
        v_prev = v

        u = eps * (log_a - torch.logsumexp((-M + v.unsqueeze(0)) / eps, dim=1))
        v = eps * (log_b - torch.logsumexp((-M + u.unsqueeze(1)) / eps, dim=0))

        max_err_u = torch.max(torch.abs(u_prev - u))
        max_err_v = torch.max(torch.abs(v_prev - v))
        if verbose:
            pbar.set_postfix({"Current Max Error": max(max_err_u, max_err_v).item()})

        if max_err_u < stop_thresh and max_err_v < stop_thresh:
            break

    log_P = (-M + u.unsqueeze(1) + v.unsqueeze(0)) / eps
    P = log_P.exp()

    approx_corr_1 = P.argmax(dim=1)
    approx_corr_2 = P.argmax(dim=0)

    if u.shape[0] > v.shape[0]:
        distance = (P * M).sum(dim=1).sum()
    else:
        distance = (P * M).sum(dim=0).sum()

    return distance, approx_corr_1, approx_corr_2


if __name__ == "__main__":
    x = torch.randn(10, 2) # (0,0)
    y = torch.randn(10, 2) + 5.0  # (5,5)

    distance, corr_x_to_y, corr_y_to_x = sinkhorn_torch(
        x, y,
        eps=1e-2,
        verbose=False
    )

    print("Distance:", distance.item())

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    plt.scatter(x_np[:, 0], x_np[:, 1], label="x (source)")
    plt.scatter(y_np[:, 0], y_np[:, 1], label="y (target)")

    for i in range(len(x_np)):
        j = corr_x_to_y[i].item()
        plt.plot(
            [x_np[i, 0], y_np[j, 0]],
            [x_np[i, 1], y_np[j, 1]],
            'k--', linewidth=0.5
        )


    plt.legend()
    plt.title("Sinkhorn Correspondences")
    plt.axis("equal")
    plt.show()