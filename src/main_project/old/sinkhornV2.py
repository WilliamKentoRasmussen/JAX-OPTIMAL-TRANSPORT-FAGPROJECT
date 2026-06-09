import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt


@jax.jit
def cdist_euclidean(x: jax.Array, y: jax.Array) -> jax.Array:
    """Computes pairwise Euclidean distances between rows of x and y.

    Args:
        x: Array of shape (N, D)
        y: Array of shape (M, D)

    Returns:
        Distance matrix of shape (N, M)
    """
    return jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))


def sinkhorn_simple(
    s: jax.Array,
    d: jax.Array,
    C: jax.Array,
    gamma: float = 0.1,
    eps: float = 1e-3,
    max_iters: int = 100,
    stop_thresh: float = 1e-5,
    verbose: bool = False,
) -> jax.Array:
    """Sinkhorn algorithm for regularised optimal transport.

    Args:
        s:           Source marginal distribution, shape (N,)
        d:           Target marginal distribution, shape (M,)
        C:           Cost matrix, shape (N, M)
        gamma:       Entropic regularisation strength
        eps:         Unused (kept for API compatibility)
        max_iters:   Maximum number of Sinkhorn iterations
        stop_thresh: Early-stop threshold on change in u / v
        verbose:     Print iteration count on early stop

    Returns:
        Transport plan T of shape (N, M)
    """
    u, v = jnp.ones_like(s), jnp.ones_like(d)
    K = jnp.exp(-C / gamma)

    for i in range(max_iters):
        u_prev, v_prev = u, v
        u = s / (jnp.dot(K, v) + 1e-8)
        v = d / (jnp.dot(K.T, u) + 1e-8)

        if jnp.max(jnp.abs(u_prev - u)) < stop_thresh and jnp.max(jnp.abs(v_prev - v)) < stop_thresh:
            if verbose:
                print(f"Converged at iteration {i}")
            break

    # Outer product scaling: T[i,j] = u[i] * K[i,j] * v[j]
    P = u[:, None] * K * v[None, :]
    return P, u, v


def sinkhorn_log(s, d, C, gamma=0.1, max_iters=1000, stop_thresh=1e-5, verbose=False):
    log_s = jnp.log(s)
    log_d = jnp.log(d)

    u = jnp.zeros_like(s)
    v = jnp.zeros_like(d)
    iter = 0

    for iter in range(max_iters):
        u = gamma * (log_s - jax.nn.logsumexp((v[None, :] - C) / gamma, axis=1))
        v = gamma * (log_d - jax.nn.logsumexp((u[:, None] - C) / gamma, axis=0))
        iter += 1

    # transport plan in log-space
    log_P = (u[:, None] + v[None, :] - C) / gamma
    P = jnp.exp(log_P)

    return P, u, v, iter


# ── Data ────────────────────────────────────────────────────────────────────

s = jnp.array([0.1, 0.35, 0.40, 0.15])  # source marginal (sums to 1)
d = jnp.array([0.2, 0.25, 0.30, 0.25])  # target marginal (sums to 1)

# Source support points: 4 points in 1-D (shape N×1 for cdist)
source_points = jnp.array([[0.0], [1.0], [2.0], [3.0]])

# Target support points: 4 points in 1-D (shape M×1 for cdist)
target_points = jnp.array([[0.5], [1.5], [2.5], [3.5]])

# Cost matrix derived from support points
C = cdist_euclidean(source_points, target_points)  # shape (4, 4)

# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    gamma = 0.2
    P, u, v = sinkhorn_simple(s, d, C, gamma=gamma)

    print("Row sums of T:", P.sum(axis=1), " — should equal s:", s)
    print("Col sums of T:", P.sum(axis=0), " — should equal d:", d)
    print("\nTransport plan T:\n", P)

    # ── Transport a new source point via the barycentric projection ──────────
    # The barycentric projection maps a source point x̂ to:
    #   ŷ = Σ_j  w_j · target_points[j]
    # where  w_j = Σ_i T[i,j] · k(x̂, source_points[i]) / Z
    #
    # For simplicity here we use uniform weights over target columns
    # (i.e. push-forward under T assuming x̂ matches the source distribution).

    new_point = jnp.array([[1.0]])  # shape (1, D) — a single 1-D point

    # Cost from the new point to every target support point
    cost_new = cdist_euclidean(new_point, target_points)  # (1, M)

    # Soft assignment weights via Sinkhorn kernel row
    K_new = jnp.exp(-cost_new / gamma)  # (1, M)
    weights = K_new[0] / K_new[0].sum()  # (M,)  normalised

    # Barycentric projection: weighted sum of target support points
    transported_point = jnp.dot(weights, target_points)  # (D,)

    print(f"\nNew source point:      {np.array(new_point[0])}")
    print(f"Transported to target: {np.array(transported_point)}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    src_np = np.array(source_points[:, 0])
    tgt_np = np.array(target_points[:, 0])
    new_np = float(new_point[0, 0])
    trans_np = float(transported_point[0])

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(src_np, np.zeros_like(src_np), s=s * 2000, label="Source support", alpha=0.7, zorder=3)
    ax.scatter(tgt_np, np.ones_like(tgt_np), s=d * 2000, label="Target support", alpha=0.7, zorder=3)
    ax.scatter([new_np], [0], marker="*", s=300, color="red", label="New point (source)", zorder=4)
    ax.scatter([trans_np], [1], marker="*", s=300, color="darkred", label="Transported point (target)", zorder=4)
    ax.annotate("", xy=(trans_np, 1), xytext=(new_np, 0), arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Source", "Target"])
    ax.set_xlabel("Position")
    ax.set_title("Optimal Transport — barycentric projection of a new point")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("sinkhorn_plot.png", dpi=150)
    plt.show()
    print("Plot saved to sinkhorn_plot.png")
