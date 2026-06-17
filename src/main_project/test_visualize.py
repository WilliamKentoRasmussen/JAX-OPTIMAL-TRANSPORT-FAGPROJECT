import pickle
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# from main_project.optimal_transport import get_trajectory
from main_project.data import getData
from main_project.environment import MODELS_DIM, INTERMEDIATE_FRACTIONS, GAMMA
from main_project.utils import load_with_hyperparams
from main_project.visualize import plot_transport_images

def evaluate_KNN_lantent_quality(model_name="ae_best_model_bo_2"):
    training_data, test_data = getData()

    x_train = jnp.array(training_data.data.numpy()).reshape(-1, 784) / 255.0 
    y_train = jnp.array(training_data.targets.numpy())

    x_test = jnp.array(test_data.data.numpy()).reshape(-1, 784) / 255.0 
    y_test = jnp.array(test_data.targets.numpy())

    model = load_with_hyperparams(name=model_name, path="models")
    _, z_train = jax.vmap(model)(x_train)
    _, z_test = jax.vmap(model)(x_test)

    z_train = np.array(z_train)
    z_test = np.array(z_test)

    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(z_train, y_train)

    score = neigh.score(z_test, y_test)
    n = len(z_test) 
    ci = 1.96 * np.sqrt(score * (1 - score) / n)
    # ADD CI !!! 
    return score, ci

def evaluate_logistic_regression(model_name="ae_best_model_bo_2"): 
    training_data, test_data = getData()

    x_train = jnp.array(training_data.data.numpy()).reshape(-1, 784) / 255.0 
    y_train = jnp.array(training_data.targets.numpy())

    x_test = jnp.array(test_data.data.numpy()).reshape(-1, 784) / 255.0 
    y_test = jnp.array(test_data.targets.numpy())

    model = load_with_hyperparams(name=model_name, path="models")
    _, z_train = jax.vmap(model)(x_train)
    _, z_test = jax.vmap(model)(x_test)

    z_train = np.array(z_train)
    z_test = np.array(z_test) 

    clf = LogisticRegression(random_state=0).fit(z_train, y_train)
    z_pred = clf.predict(z_test)
    #print(classification_report(z_pred, y_test))

    acc = clf.score(z_test, y_test)
    # 95 % confidenze intervals b
    n = len(z_test)
    ci = 1.96 * np.sqrt(acc * (1 - acc) / n)

    return acc, ci



def evaluate_test_MSE(model_name="ae_best_model_bo_2"):
    training_data, test_data = getData()

    x_train = jnp.array(training_data.data.numpy()).reshape(-1, 784) / 255.0 
    y_train = jnp.array(training_data.targets.numpy())

    x_test = jnp.array(test_data.data.numpy()).reshape(-1, 784) / 255.0 
    y_test = jnp.array(test_data.targets.numpy())

    model = load_with_hyperparams(name=model_name, path="models")

    _, z_train = jax.vmap(model)(x_train)
    x_hat_test, z_test = jax.vmap(model)(x_test)

    z_train = np.array(z_train)
    z_test = np.array(z_test)

    # Reconstruction MSE with 95% CI via standard error
    per_sample_mse = np.mean((np.array(x_test) - np.array(x_hat_test)) ** 2, axis=1)
    mse = np.mean(per_sample_mse)
    # CI need to bee added !!! 
    mse_ci = 1.96 * np.std(per_sample_mse) / np.sqrt(len(per_sample_mse))
    
    return mse, mse_ci

    

def plot_reconstruction_for_all_dim(save = False):
    _, test_data = getData()

    # pick one example per digit class (0-9)
    class_examples = {}
    for img, label in test_data:
        label = int(label)
        if label not in class_examples:
            class_examples[label] = img.numpy()
        if len(class_examples) == 10:
            break
    labels = sorted(class_examples.keys())
    xs = [class_examples[c] for c in labels]

    x = jnp.array(xs).reshape(len(xs), -1)
    x_img = np.array(x).reshape(-1, 28, 28)

    recons = []
    for dim in MODELS_DIM:
        model = load_with_hyperparams(name=f"ae_best_model_bo_{dim}", path="models")
        recon, _ = jax.vmap(model)(x)
        recons.append(np.array(recon).reshape(-1, 28, 28))

    def hide_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    n_cols = len(labels)
    n_rows = 1 + len(MODELS_DIM)
    _, axes = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.5 * n_rows))

    for i, label in enumerate(labels):
        axes[0, i].imshow(x_img[i], cmap="gray")
        hide_ticks(axes[0, i])
        # axes[0, i].set_title(f"class {label}", fontsize=8)
    axes[0, 0].set_ylabel("Original", fontsize=10)

    for row, (dim, recon) in enumerate(zip(MODELS_DIM, recons), start=1):
        for i in range(n_cols):
            axes[row, i].imshow(recon[i], cmap="gray")
            hide_ticks(axes[row, i])
        axes[row, 0].set_ylabel(f"dim={dim}", fontsize=10)

    plt.tight_layout()
    if save: 
        plt.savefig("figures/all_dim_reconstruction.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_interpolation_reconstructions(
    model_name="ae_best_model_bo_2",
    gamma=0.001,
    source_label=0,
    target_label=1,
    n=5,
    fractions=None,
    save=False,
):
    if fractions is None:
        fractions = INTERMEDIATE_FRACTIONS

    model = load_with_hyperparams(name=model_name, path="models")

    with open(f"data/{model_name}_ot_data.pkl", "rb") as f:
        ot_data = pickle.load(f)

    trajectory = ot_data[gamma][f"source_{source_label}_target_{target_label}"]

    source_images = trajectory["source_images"][:n]             # [n, 784]
    expected_target_latent = trajectory["expected_target"][:n]  # [n, latent_dim]
    target_images = trajectory["target_images"][:n]             # [n, 784]

    # re-encode source images to get source latents for interpolation
    _, source_latent = jax.vmap(model)(jnp.array(source_images))

    # columns: Source | t=f0 | t=f1 | ... | Actual target
    n_cols = 2 + len(fractions)
    _, axes = plt.subplots(n, n_cols, figsize=(2 * n_cols, 2 * n))

    for i in range(n):
        z0 = jnp.array(source_latent[i])
        z1 = jnp.array(expected_target_latent[i])

        axes[i, 0].imshow(source_images[i].reshape(28, 28), cmap="gray")
        axes[i, 0].axis("off")

        for j, f in enumerate(fractions):
            z_inter = (1.0 - f) * z0 + f * z1
            img = np.array(model.decoder(z_inter)).reshape(28, 28)
            axes[i, j + 1].imshow(img, cmap="gray")
            axes[i, j + 1].axis("off")

        axes[i, -1].imshow(target_images[i].reshape(28, 28), cmap="gray")
        axes[i, -1].axis("off")

    axes[0, 0].set_title("Source", fontsize=9)
    for j, f in enumerate(fractions):
        axes[0, j + 1].set_title(f"t={f}", fontsize=9)
    axes[0, -1].set_title("Target", fontsize=9)

    plt.suptitle(f"OT Interpolation: {source_label} → {target_label}  |  γ={gamma}", fontsize=12)
    plt.tight_layout()
    if save:
        plt.savefig("figures/interpolation_reconstructions.png", dpi=150, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    results = []
    for dim in MODELS_DIM:
        KNN_score, KNN_ci = evaluate_KNN_lantent_quality(model_name=f"ae_best_model_bo_{dim}")
        MSE, mse_ci = evaluate_test_MSE(model_name=f"ae_best_model_bo_{dim}")
        LR_acc, LR_ci = evaluate_logistic_regression(model_name=f"ae_best_model_bo_{dim}")
        print(f"KNN accuracy for dimension {dim}: {KNN_score}")
        print(f"Reconstruction error for dimension {dim}: {MSE}")
        print(f"Logistic Regression acc. for dimension {dim}: {LR_acc}")
        results.append({"dim": dim, 
                        "reconstruction_mse": (MSE, mse_ci),
                        "knn_accuracy": (KNN_score, KNN_ci),
                        "Logistic Regression acc." : (LR_acc, LR_ci)})


    print(pd.DataFrame(results).to_latex(index=False))
