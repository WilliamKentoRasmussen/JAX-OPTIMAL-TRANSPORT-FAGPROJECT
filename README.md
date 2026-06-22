# Sinkhorn-Based Optimal Transport for Image Translation in an Autoencoder's Latent Space

This project investigates discrete Optimal Transport (OT) via the Sinkhorn algorithm applied in a
latent space learned by an Autoencoder (AE) on the MNIST dataset. It examines how the entropic
regularization parameter γ and bottleneck dimensionality affect source-to-target distribution alignment.

**Authors:** Kerem Ozemre (s244794), Markus Thomasson (s244705), William Rasmussen (s245310)  
**Course:** 02466 Project Work, DTU Compute, June 2026

---

> README templated with [Claude](https://claude.ai/claude-code).

## Project structure

```
├── .github/                          # GitHub Actions CI, Dependabot, issue templates
│   ├── workflows/
│   │   ├── tests.yaml
│   │   └── linting.yaml
│   └── ISSUE_TEMPLATE/
├── data/                             # OT pickle files and evaluation CSVs
├── figures/                          # Generated figures
│   ├── rapport/                      # Final report figures
│   └── plots/
├── models/                           # Saved autoencoder checkpoints (ae_best_model_bo_{dim})
├── src/
│   ├── main_project/
│   │   ├── environment.py            # All global constants (dims, γ, Kmax, τ)
│   │   ├── model.py                  # Autoencoder architecture (AEv2)
│   │   ├── data.py                   # MNIST data loading
│   │   ├── trainnew.py               # AETrainer with train/val loop and early stopping
│   │   ├── hyperparameter_optimization.py  # Bayesian HP search (Optuna) + final retraining
│   │   ├── sinkhorn.py               # Log-domain Sinkhorn algorithm
│   │   ├── optimal_transport.py      # OT pipeline: encode → transport → decode → save pickle
│   │   ├── evaluate.py               # Evaluation metrics (MMD, Wasserstein, entropy, classifier confidence)
│   │   ├── experiment.py             # Entry point: runs OT pipeline + evaluation
│   │   ├── AE_visualize_and_metrics.py  # AE reconstruction figures and metric tables
│   │   ├── visualize.py              # OT result visualizations
│   │   ├── gamma_visualization.py    # γ sensitivity plots
│   │   └── utils.py                  # Model save/load helpers
│   └── HPC/
│       └── image_classifier.py       # CNN classifier training (run on DTU HPC)
├── .python-version                   # Python 3.13
├── pyproject.toml                    # Dependencies
└── uv.lock                           # Pinned dependency lockfile
```

---

## Reproducing the results

### 1. Requirements

- Python 3.13 (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) package manager

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies from the lockfile
uv sync
```

### 2. Step 1 — Train the autoencoders

Run Bayesian hyperparameter search (30 trials per latent dimension) and retrain the best model
for each of the 6 latent dimensions `{2, 4, 8, 10, 16, 32}`:

```bash
uv run python src/main_project/hyperparameter_optimization.py
```

This saves six model checkpoints: `models/ae_best_model_bo_{dim}` for dim in `{2, 4, 8, 10, 16, 32}`.

### 3. Step 2 — Run the OT pipeline and evaluation

```bash
uv run python src/main_project/experiment.py
```

This encodes the MNIST test set for each latent dimension, runs Log-Domain Sinkhorn across all
γ ∈ {0.001, 0.01, 0.1, 1} and all 90 ordered source-target digit pairs, applies barycentric
projection, decodes transported samples, and saves evaluation metrics (MMD, Wasserstein, CNN
classifier probability, entropy).

### 4. Step 3 — Generate figures

```bash
uv run python src/main_project/AE_visualize_and_metrics.py   # Table 4.1 and Figs 4.1, 4.2
uv run python src/main_project/gamma_visualization.py        # Fig 3.1
uv run python src/main_project/visualize.py                  # Table 4.2 and Figs 4.3, 4.4, 4.5, Appendix heatmaps and images
```
---

## Randomness and reproducibility

All random seeds are fixed. The table below documents every seed used:

| Component | Seed | Location |
|---|---|---|
| Optuna TPE sampler (HP search) | `42` | `hyperparameter_optimization.py` — `TPESampler(seed=42)` |
| Each HP search trial (model init) | `trial.number` (0–29) | `hyperparameter_optimization.py` — `jr.PRNGKey(trial.number)` |
| Final model retraining (model init) | `999` | `hyperparameter_optimization.py` — `jr.PRNGKey(999)` |
| CNN classifier initialization | `5678` | `evaluate.py` — `SEED = 5678` |

> **Note:** Because all six final AEs are initialized with `PRNGKey(999)`, the only source of variation
> between models is the architecture and hyperparameters selected by Bayesian optimization — not
> random initialization.

---

## Key hyperparameters and algorithm constants

All global constants are defined in `src/main_project/environment.py`:

| Constant | Value | Description |
|---|---|---|
| `MODELS_DIM` | `[2, 4, 8, 10, 16, 32]` | Latent dimensions evaluated |
| `GAMMA` | `[0.001, 0.01, 0.1, 1]` | Entropic regularization values |
| `MAX_ITERATION` | `10000` | Kmax — maximum Sinkhorn iterations |
| `STOP_THRESHOLD` | `1e-4` | τ — relative convergence tolerance |
| `OPTIMAL_GAMMA` | `0.001` | γ used for the dimension comparison (Table 4.2) |
| `MAX_POINTS` | `700` | Maximum source/target samples per OT run |

---

## Dependencies

Dependencies are managed with `uv` and pinned in `uv.lock`. Key libraries:

| Library | Version | License |
|---|---|---|
| JAX | ≥ 0.9.0.1 | Apache 2.0 |
| Equinox | ≥ 0.13.4 | MIT |
| Optax | ≥ 0.2.7 | Apache 2.0 |
| Optuna | ≥ 4.9.0 | MIT |
| PyTorch | 2.6.0 | BSD-3 |
| torchvision | ≥ 0.21.0 | BSD-3 |
| scikit-learn | ≥ 1.8.0 | BSD-3 |

Full pinned versions: see `uv.lock`.

---

## Data

The **MNIST** dataset (LeCun et al., 1998) is downloaded automatically via `torchvision.datasets.MNIST`.
It is open-source and freely available for research use. No custom datasets are used or released.
