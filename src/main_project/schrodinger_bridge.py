from jax import errors
import numpy as np
from typing import Optional
from scipy.stats import multivariate_normal
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from main_project.sinkhornV2 import cdist_euclidean
from main_project.utils import load
from main_project.data import getData, getDataloader
from sklearn.decomposition import PCA


@jax.jit
def cdist_euclidean(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))


@jax.jit
def gaussian_kernel(X, Y, t, sigma, d):
    variance = 2 * sigma**2 * t
    dist_sq = cdist_euclidean(X, Y)
    K = jnp.exp(-dist_sq / (variance + 1e-10))
    K = K / (2 * jnp.pi * (variance + 1e-10)) ** (d / 2)
    return K


def density_weights(z, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    dists, _ = nn.kneighbors(z)
    mean_dist = dists[:, 1:].mean(axis=1)
    density = 1.0 / (mean_dist + 1e-8)
    return density / density.sum()


class SchrodingerBridge:
    def __init__(
        self,
        n_steps: int = 100,
        sigma: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
    ):
        self.n_steps = n_steps
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.history_ = []

    def fit(
        self,
        latent_source: jnp.array,
        latent_target: jnp.array,
        weights_x: Optional[jnp.array] = None,
        weights_y: Optional[jnp.array] = None,
    ) -> "SchrodingerBridge":
        X = jnp.asarray(latent_source, dtype=jnp.float32)
        Y = jnp.asarray(latent_target, dtype=jnp.float32)

        m, d = X.shape
        n, _ = Y.shape

        self.X = X
        self.Y = Y
        self.d = d

        self.weights_x = weights_x if weights_x is not None else jnp.ones(m) / m
        self.weights_y = weights_y if weights_y is not None else jnp.ones(n) / n

        self.t = jnp.linspace(0, 1, self.n_steps)
        self.dt = 1.0 / (self.n_steps - 1)

        self.log_u = jnp.zeros((self.n_steps, m))
        self.log_v = jnp.zeros((self.n_steps, n))

        self._iterative_proportional_fitting()
        return self

    def _iterative_proportional_fitting(self):
        errors = []

        outer_bar = tqdm(range(self.max_iter), desc="IPF iterations")

        for iteration in outer_bar:
            log_u_old = self.log_u.copy()
            log_v_old = self.log_v.copy()

            # Forward pass — enforce source marginal
            for i in tqdm(range(self.n_steps - 1), desc="  Forward ", leave=False):
                t_step = self.t[i + 1] - self.t[i]
                K = gaussian_kernel(self.X, self.Y, t_step, self.sigma, self.d)
                v_scaled = self.weights_y * jnp.exp(self.log_v[i + 1])
                self.log_u = self.log_u.at[i].set(jnp.log(self.weights_x) - jnp.log(K @ v_scaled + 1e-16))

            # Backward pass — enforce target marginal
            for i in tqdm(range(self.n_steps - 1, 0, -1), desc="  Backward", leave=False):
                t_step = self.t[i] - self.t[i - 1]
                K = gaussian_kernel(self.X, self.Y, t_step, self.sigma, self.d)
                u_scaled = self.weights_x * jnp.exp(self.log_u[i - 1])
                self.log_v = self.log_v.at[i].set(jnp.log(self.weights_y) - jnp.log(K.T @ u_scaled + 1e-16))

            # Convergence — use absolute difference in log space (same as Sinkhorn stop_thresh)
            err_u = float(jnp.max(jnp.abs(self.log_u - log_u_old)))
            err_v = float(jnp.max(jnp.abs(self.log_v - log_v_old)))
            errors.append(max(err_u, err_v))
            outer_bar.set_postfix({"err_u": f"{err_u:.2e}", "err_v": f"{err_v:.2e}"})

            if err_u < self.tol and err_v < self.tol:
                print(f"Converged at iteration {iteration}")
                break

        self.history_ = {"num_iter": iteration + 1, "errors": errors}

    def get_transport_plan(self) -> jnp.array:
        K = gaussian_kernel(self.X, self.Y, 1.0, self.sigma, self.d)
        u = jnp.exp(self.log_u[0])
        v = jnp.exp(self.log_v[-1])
        P = self.weights_x[:, None] * u[:, None] * K * v[None, :] * self.weights_y[None, :]
        P = P / P.sum()
        return P

    def sample_trajectories(self, P, model, n_samples=10, seed=0):
        key = jax.random.PRNGKey(seed)

        flat = np.array(P).flatten()
        flat = flat / flat.sum()
        idx = np.random.choice(len(flat), size=n_samples, p=flat)
        rows, cols = np.unravel_index(idx, P.shape)

        trajectories = np.zeros((n_samples, self.n_steps, self.d))
        decoded = np.zeros((n_samples, self.n_steps, 784))

        for s in tqdm(range(n_samples), desc="Sampling trajectories"):
            x_current = np.array(self.X[rows[s]])
            trajectories[s, 0] = x_current

            for i in range(self.n_steps - 1):
                t_step = float(self.t[i + 1] - self.t[i])
                K = np.array(gaussian_kernel(jnp.array(x_current[None]), self.Y, t_step, self.sigma, self.d))  # [1, n]

                # Use learned v potential to weight transitions — this is the curvature
                v_scaled = np.array(self.weights_y) * np.exp(np.array(self.log_v[i + 1]))
                weights = (K * v_scaled).flatten()
                weights = np.array(weights, dtype=np.float64)  # float64 for numerical precision
                weights = np.abs(weights)  # guard against tiny negatives
                weights = weights / (weights.sum() + 1e-16)
                weights = weights / weights.sum()

                # Sample next anchor point from target according to weights
                j = np.random.choice(len(self.Y), p=weights)
                anchor = np.array(self.Y[j])

                # Step toward anchor with small Brownian noise
                key, sub = jax.random.split(key)
                noise = np.array(jax.random.normal(sub, shape=(self.d,)))
                x_current = anchor + self.sigma * np.sqrt(t_step) * noise

                trajectories[s, i + 1] = x_current

            # Decode
            frames = jax.vmap(model.decoder)(jnp.array(trajectories[s]))
            decoded[s] = np.array(frames)

        return {
            "trajectories": trajectories,
            "decoded": decoded,
            "source_idx": rows,
            "target_idx": cols,
        }

    def plot_trajectories(self, result: dict, n_display: int = 5):
        """
        Plot decoded images along sampled trajectories.

        Each row is one trajectory: source → intermediates → target
        """
        n_display = min(n_display, len(result["trajectories"]))
        steps_show = np.linspace(0, self.n_steps - 1, 8, dtype=int)  # 8 evenly spaced steps
        n_cols = len(steps_show)

        fig, axes = plt.subplots(n_display, n_cols, figsize=(2 * n_cols, 2 * n_display))

        for row in range(n_display):
            for col, step in enumerate(steps_show):
                img = result["decoded"][row, step].reshape(28, 28)
                axes[row, col].imshow(img, cmap="gray")
                axes[row, col].axis("off")
                if row == 0:
                    t_val = self.t[step]
                    axes[row, col].set_title(f"t={t_val:.2f}", fontsize=8)

        axes[0, 0].set_title("Source", fontsize=8)
        axes[0, -1].set_title("Target", fontsize=8)
        plt.suptitle("Schrödinger Bridge Trajectories")
        plt.tight_layout()
        plt.savefig("figures/sb_trajectories.png", dpi=150, bbox_inches="tight")
        plt.show()

    def plot_latent_paths(
        self,
        result: dict,
        n_display: int = 10,
        point_size: int = 4,
        alpha: float = 0.4,
    ):
        trajectories = result["trajectories"]  # [n_samples, n_steps, d]
        source_idx = result["source_idx"]
        target_idx = result["target_idx"]

        X_np = np.array(self.X)
        Y_np = np.array(self.Y)
        d = X_np.shape[1]

        # --- Project to 2D if needed ---
        if d > 2:
            all_points = np.vstack([X_np, Y_np])
            pca = PCA(n_components=2).fit(all_points)
            X_2d = pca.transform(X_np)
            Y_2d = pca.transform(Y_np)
            paths_2d = np.stack(
                [
                    pca.transform(trajectories[s])  # [n_steps, 2]
                    for s in range(len(trajectories))
                ]
            )
            axis_labels = ("PC1", "PC2")
        else:
            X_2d = X_np
            Y_2d = Y_np
            paths_2d = trajectories[:, :, :2]
            axis_labels = ("z₁", "z₂")

        fig, ax = plt.subplots(figsize=(9, 7))

        # --- Background cloud: all source and target points ---
        ax.scatter(X_2d[:, 0], X_2d[:, 1], color="steelblue", alpha=0.15, s=point_size, zorder=0, label="Source (all)")
        ax.scatter(Y_2d[:, 0], Y_2d[:, 1], color="coral", alpha=0.15, s=point_size, zorder=0, label="Target (all)")

        # --- Draw each sampled trajectory ---
        for s in range(min(n_display, len(paths_2d))):
            path = paths_2d[s]  # [n_steps, 2]

            # Path line
            ax.plot(path[:, 0], path[:, 1], color="gray", alpha=0.5, linewidth=0.8, zorder=1)

            # Color intermediate points by time t
            for i, t in enumerate(np.array(self.t)):
                color = plt.cm.plasma(t)
                ax.scatter(path[i, 0], path[i, 1], color=color, s=8, alpha=0.6, zorder=2)

            # Source endpoint
            x0 = X_2d[source_idx[s]]
            ax.scatter(
                *x0, color="steelblue", edgecolors="black", s=60, zorder=4, label="Source point" if s == 0 else ""
            )

            # Target endpoint
            x1 = Y_2d[target_idx[s]]
            ax.scatter(*x1, color="coral", edgecolors="black", s=60, zorder=4, label="Target point" if s == 0 else "")

            # Arrow source → target
            ax.annotate("", xy=x1, xytext=x0, arrowprops=dict(arrowstyle="->", color="black", alpha=0.4, lw=0.8))

        # --- Colorbar for time t ---
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="t  (0=source, 1=target)")

        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(
            f"Schrödinger Bridge paths in latent space  " f"(σ={self.sigma}, {len(trajectories)} trajectories)"
        )
        ax.legend(loc="upper left", markerscale=2)
        plt.tight_layout()
        plt.savefig("figures/sb_latent_paths.png", dpi=150, bbox_inches="tight")
        plt.show()


# ── Test ─────────────────────────────────────────────────────────────────────
model = load(name="ae_best_model_lat2", path="models")

training_data, _ = getData()
train_loader = getDataloader(training_data)

source_data, target_data = [], []
for x, y in train_loader:
    source_data.extend(x[y == 0])
    target_data.extend(x[y == 1])

source_arr = jnp.array(np.stack([np.array(img).flatten() for img in source_data]))
target_arr = jnp.array(np.stack([np.array(img).flatten() for img in target_data]))


def recon(x):
    r, z = model(x)
    return r, z


latent_source = jax.vmap(recon)(source_arr)[1]
latent_target = jax.vmap(recon)(target_arr)[1]

min_count = min(latent_source.shape[0], latent_target.shape[0])
latent_source = latent_source[:min_count]
latent_target = latent_target[:min_count]

weights_x = jnp.array(density_weights(np.array(latent_source), k=5))
weights_y = jnp.array(density_weights(np.array(latent_target), k=5))

bridge = SchrodingerBridge(n_steps=20, sigma=0.5, max_iter=100, tol=1e-6)
bridge.fit(latent_source=latent_source, latent_target=latent_target, weights_x=weights_x, weights_y=weights_y)

P = bridge.get_transport_plan()
result = bridge.sample_trajectories(P=P, model=model, n_samples=10)
bridge.plot_trajectories(result, n_display=5)
bridge.plot_latent_paths(result, n_display=1)
