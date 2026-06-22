# Code Review — JAX Optimal Transport Project

**Reviewed:** 2026-06-16  
**Reviewer:** Claude (automated review)  
**Scope:** All source files under `src/main_project/`, `tests/`, `utils`, and CI configuration

---

## Table of Contents

1. [Critical — Will crash at runtime](#1-critical--will-crash-at-runtime)
2. [Logic bugs — Wrong results without crashing](#2-logic-bugs--wrong-results-without-crashing)
3. [Code quality](#3-code-quality)
4. [Summary table](#4-summary-table)

---

## 1. Critical — Will crash at runtime

These issues prevent the pipeline from running at all and should be fixed first.

---

### 1.1 `STOP_THRESSHOLD` typo causes `ImportError`

**Files:** `src/main_project/sinkhorn.py:28`, `src/main_project/optimal_transport.py:26`

`environment.py` defines `STOP_THRESHOLD = 1e-4` (single S), but both `sinkhorn.py` and `optimal_transport.py` import the misspelled name `STOP_THRESSHOLD` (double S). Python raises `ImportError` at startup, preventing the entire pipeline from loading.

```python
# environment.py (correct)
STOP_THRESHOLD = 1e-4

# sinkhorn.py (wrong — double S)
from main_project.environment import (
    ...
    STOP_THRESSHOLD,   # ← ImportError
)
```

**Fix:** Rename the constant in `environment.py` to `STOP_THRESSHOLD`, or update all import sites to use `STOP_THRESHOLD`.

---

### 1.2 `NameError` in `sinkhorn_log` default argument

**File:** `src/main_project/sinkhorn.py:111`

```python
def sinkhorn_log(s, d, C, gamma=0.1, max_iters=MAX_ITERATION, stop_thresh=STOP_THRESHOLD, verbose=True):
```

`STOP_THRESHOLD` (correct spelling) is used as a default argument but is not imported in `sinkhorn.py`. Default arguments are evaluated at function definition time (module load), so this raises `NameError` immediately on import.

**Fix:** Import `STOP_THRESHOLD` from `environment`, or pass the value explicitly at each call site.

---

### 1.3 Duplicate empty function definition — `SyntaxError`

**File:** `src/main_project/hyperparameter_optimization.py:43–45`

```python
def objective(trial, latent_dim=2):   # ← no body

def objective(trial, latent_dim=2):   # ← second definition (the real one)
    ...
```

A function with no body is a `SyntaxError`. The file cannot be imported or executed.

**Fix:** Delete the first (empty) `def objective` line.

---

### 1.4 Pandas `and` on two Series — `ValueError`

**File:** `src/main_project/visualize.py:841`

```python
mask = summary_df["latent_dim"] == dim and summary_df["gamma"] == 0.1
```

Using Python's `and` on two pandas `Series` raises:
```
ValueError: The truth value of a Series is ambiguous.
```

**Fix:** Use the element-wise operator:
```python
mask = (summary_df["latent_dim"] == dim) & (summary_df["gamma"] == 0.1)
```

---

### 1.5 `pd.DataFrame` column count mismatch

**File:** `src/main_project/evaluate.py:138–156`

```python
columns = ["Fraction of transport", "MMD", "Confidence of Classifier", "FID"]  # 4 columns
...
data.append([frac, float(mmd), float(jnp.mean(classifier_conf))])  # 3 values (FID commented out)
```

`pd.DataFrame(data, columns=columns)` raises `ValueError` because 4 column names are provided but each row only has 3 values.

**Fix:** Either remove `"FID"` from `columns`, or restore the FID calculation.

---

### 1.6 Return value unpacking mismatch

**Files:** `src/main_project/sinkhorn.py:258`, `src/main_project/W_distance_test.py:17`

`run_sinkhorn_by_model` returns **7** values (`latent_source, latent_target, P, u, v, iter, running_time`), but both call sites unpack only **6**:

```python
# sinkhorn.py main()
latent_source, latent_target, P, u, v, iter = run_sinkhorn_by_model(...)  # missing running_time

# W_distance_test.py
latent_source, latent_target, P, _, _, _ = run_sinkhorn_by_model(...)    # missing running_time
```

Both raise `ValueError: too many values to unpack`.

**Fix:** Add `running_time` (or `_`) to each unpacking.

---

### 1.7 `MyDataset` imported in test but does not exist

**File:** `tests/test_data.py:3`

```python
from main_project.data import MyDataset
```

`MyDataset` is fully commented out in `data.py`. The test fails at import with `ImportError`.

**Fix:** Either implement `MyDataset` in `data.py`, or remove/replace this test.

---

### 1.8 `train` undefined in `utils.py` `__main__` block

**File:** `src/main_project/utils.py:57`

```python
if __name__ == "__main__":
    model, train_losses = train(1)   # `train` is never imported
```

Running this file directly raises `NameError: name 'train' is not defined`.

**Fix:** Import `train` from `main_project.train`, or remove the `__main__` block.

---

## 2. Logic bugs — Wrong results without crashing

These produce silently incorrect output.

---

### 2.1 Gaussian kernel uses distance instead of squared distance

**File:** `src/main_project/schrodinger_bridge.py:28–33`

```python
def gaussian_kernel(X, Y, t, sigma, d):
    variance = 2 * sigma**2 * t
    dist_sq = cdist_euclidean(X, Y)           # ← returns ||x−y||, NOT ||x−y||²
    K = jnp.exp(-dist_sq / (variance + 1e-10))
```

`cdist_euclidean` returns Euclidean *distance*. The variable is named `dist_sq` and used in a Gaussian kernel formula that expects *squared* distance. As written, the kernel computes:

```
K(x, y) = exp(−||x−y|| / variance)
```

instead of the correct Gaussian:

```
K(x, y) = exp(−||x−y||² / variance)
```

This changes the shape of the Schrödinger bridge and all downstream trajectory results.

**Fix:**
```python
dist_sq = cdist_euclidean(X, Y) ** 2
```

---

### 2.2 `jax.random.randint` with empty range

**File:** `src/main_project/model.py:94`

```python
x = jax.random.randint(subkey, (784,), 10, 10)   # range [10, 10) is empty
```

`minval=10, maxval=10` produces an empty range. JAX either returns zeros or raises depending on version.

**Fix:** Use a valid range, e.g. `(0, 256)` for pixel intensities.

---

### 2.3 Second `__main__` guard in `data.py` never triggers

**File:** `src/main_project/data.py:46`

```python
if __name__ == "main":    # should be "__main__"
    training_data, test_data = getData()
    print(len(training_data), len(test_data))
```

The string `"main"` never equals `__name__`. This block is dead code.

**Fix:** Change `"main"` to `"__main__"`.

---

### 2.4 `W_distance_test.py` — outer `W_distances` list is immediately reset

**File:** `src/main_project/W_distance_test.py:11–15`

```python
W_distances = []                 # outer list (never used)
for dim in MODELS_DIM:
    W_distances = []             # ← resets on every iteration
    for gamma_val in GAMMA:
        ...
        W_distances.append(W_gamma)
    print(W_distances)           # prints only the current dim's values
```

The outer list is overwritten immediately in the loop. If the intent is to accumulate across all dimensions, this is a bug.

**Fix:** Remove the inner `W_distances = []` reassignment, or use a dict keyed by `dim`.

---

### 2.5 `train.py::train` — best model never saved

**File:** `src/main_project/train.py:114–121`

The comment says `# --- Save best model ---` but the `save()` call occurs after the loop, saving whichever model state remains at the last epoch — not the one with the lowest validation loss. The `AETrainer` class in `trainnew.py` correctly implements early stopping and best-model tracking; `train.py` does not.

**Fix:** Track `best_model` and `best_val_loss` inside the epoch loop, and call `save(best_model, ...)`.

---

### 2.6 Mixed random systems in `sample_trajectories`

**File:** `src/main_project/schrodinger_bridge.py:131–158`

A JAX key is created and properly split throughout the method, but the actual random choices use NumPy's global RNG:

```python
idx = np.random.choice(len(flat), size=n_samples, p=flat)   # numpy RNG, not seeded from key
...
j = np.random.choice(len(self.D), p=weights)                 # numpy RNG again
```

The JAX key is only used to generate Brownian noise. This means the sampling is not reproducible via the `seed` argument alone; you would also need to call `np.random.seed(seed)`.

**Fix:** Either use `np.random.default_rng(seed)` consistently, or convert all random draws to JAX.

---

## 3. Code quality

Lower-severity issues that affect maintainability and readability.

---

### 3.1 Duplicate imports

**File:** `src/main_project/model.py:1–4`

```python
import equinox as eqx   # line 1
import jax               # line 2
import equinox as eqx   # line 3 — duplicate
import jax               # line 4 — duplicate
```

**Fix:** Remove the duplicate lines.

---

### 3.2 Wrong import from `pyexpat` in `trainnew.py`

**File:** `src/main_project/trainnew.py:1`

```python
from pyexpat import model   # pyexpat is Python's XML parsing module
```

`pyexpat.model` is a C-extension constant for XML element content models — completely unrelated to this project. It is never used, and the name `model` is immediately shadowed by actual model code.

Also, `import re` appears on both line 2 and line 19.

**Fix:** Remove both the `pyexpat` import and the duplicate `import re`.

---

### 3.3 `from jax import errors` is shadowed and unused

**File:** `src/main_project/schrodinger_bridge.py:1, 88`

```python
from jax import errors   # line 1 — imports jax.errors module

def _iterative_proportional_fitting(self):
    errors = []           # line 88 — shadows the import with a plain list
```

The `jax.errors` import is never actually used, and the local variable shadows it inside the method.

**Fix:** Remove `from jax import errors` at line 1.

---

### 3.4 `cdist_euclidean` imported then immediately redefined

**File:** `src/main_project/schrodinger_bridge.py:11, 23`

```python
from main_project.sinkhorn import cdist_euclidean  # line 11 — import

@jax.jit
def cdist_euclidean(x, y):                          # line 23 — redefines it locally
    return jnp.sqrt(jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
```

The import is dead code. The local version also uses the slower broadcasting formula instead of the optimized BLAS version in `sinkhorn.py`.

**Fix:** Remove the local `cdist_euclidean` definition and use the imported one. (After fixing issue 2.1, which affects which formula is correct.)

---

### 3.5 `iter` shadows Python built-in

**File:** `src/main_project/sinkhorn.py:117`

```python
for iter in tqdm(range(max_iters), ...):
```

`iter` is a Python built-in function. Naming a loop variable `iter` shadows it for the rest of the function scope.

**Fix:** Rename to `i` or `step`.

---

### 3.6 Classifier loaded at module import time

**File:** `src/main_project/evaluate.py:108`

```python
classifier = load(name="evaluate_classifier", path="models", model=targetClassifier(subkey))
```

This runs when `evaluate` is first imported. If `models/evaluate_classifier.eqx` is missing, every module that imports from `evaluate` will crash — including during test collection. This makes testing and development harder.

**Fix:** Move the load inside `run_evaluation()`, or use a module-level `classifier = None` and lazy-load on first use.

---

### 3.7 Duplicate function definitions across files

The following functions are defined in **both** `test_visualize.py` and `visualize.py`, with slight differences between the copies:

- `evaluate_KNN_lantent_quality`
- `evaluate_test_MSE`
- `plot_reconstruction_for_all_dim`

Duplicate code means bug fixes and improvements must be applied in two places.

**Fix:** Keep one canonical version in `visualize.py` and import it in `test_visualize.py`.

---

### 3.8 `mse_ci` computed but discarded

**Files:** `src/main_project/test_visualize.py:86`, `src/main_project/visualize.py:761`

```python
mse_ci = 1.96 * np.std(per_sample_mse) / np.sqrt(len(per_sample_mse))
return mse   # mse_ci is never returned or logged
```

The 95% confidence interval is computed and then silently thrown away. The TODO comment `# CI need to bee added !!!` suggests this is known.

**Fix:** Return `(mse, mse_ci)` and propagate it to the results table/CSV.

---

### 3.9 `test_model.py` and `test_api.py` are empty

**Files:** `tests/test_model.py`, `tests/test_api.py`

Both files exist but contain no content (1-line blank file). The CI runs coverage against a test suite that has essentially no real tests. The one test that does exist (`test_data.py`) fails at import due to issue 1.7.

**Fix:** Add meaningful tests, or at minimum, add a single smoke test per module.

---

### 3.10 Leftover debug comments in `sinkhorn_simple`

**File:** `src/main_project/sinkhorn.py:106–107`

```python
# Outer product scaling: T[i,j] = u[i] * K[i,j] * v[j] Elementwisemultiplicaiton? 
P = u[:, None] * K * v[None, :]  #Output: (3, 1) instead of (3,)
```

These look like debugging notes from development. The shape comment `(3, 1) instead of (3,)` refers to an intermediate investigation that no longer applies to the current code.

**Fix:** Remove or clean up these comments.

---

## 4. Summary table

| # | File | Severity | Issue |
|---|------|----------|-------|
| 1.1 | `sinkhorn.py`, `optimal_transport.py` | **Critical** | `STOP_THRESSHOLD` typo → `ImportError` |
| 1.2 | `sinkhorn.py:111` | **Critical** | `STOP_THRESHOLD` not imported → `NameError` at load |
| 1.3 | `hyperparameter_optimization.py:43` | **Critical** | Empty duplicate `def objective` → `SyntaxError` |
| 1.4 | `visualize.py:841` | **Critical** | `and` on Series → `ValueError` |
| 1.5 | `evaluate.py:138` | **Critical** | 4 column names, 3 data values → `DataFrame ValueError` |
| 1.6 | `sinkhorn.py:258`, `W_distance_test.py:17` | **Critical** | 6-value unpack of 7-value return → `ValueError` |
| 1.7 | `tests/test_data.py:3` | **Critical** | `MyDataset` doesn't exist → `ImportError` |
| 1.8 | `utils.py:57` | **Critical** | `train` undefined in `__main__` → `NameError` |
| 2.1 | `schrodinger_bridge.py:31` | **Logic** | Distance used instead of squared distance in Gaussian kernel |
| 2.2 | `model.py:94` | **Logic** | `randint` with empty range `[10, 10)` |
| 2.3 | `data.py:46` | **Logic** | `if __name__ == "main"` never triggers |
| 2.4 | `W_distance_test.py:15` | **Logic** | Outer accumulator list reset in loop |
| 2.5 | `train.py:114` | **Logic** | Final model saved, not best-val-loss model |
| 2.6 | `schrodinger_bridge.py:135` | **Logic** | JAX key and `np.random` mixed — not reproducible |
| 3.1 | `model.py:1–4` | Quality | Duplicate `import equinox`, `import jax` |
| 3.2 | `trainnew.py:1,19` | Quality | `from pyexpat import model` wrong import; duplicate `import re` |
| 3.3 | `schrodinger_bridge.py:1` | Quality | `from jax import errors` unused and shadowed |
| 3.4 | `schrodinger_bridge.py:23` | Quality | `cdist_euclidean` imported then immediately redefined |
| 3.5 | `sinkhorn.py:117` | Quality | `iter` shadows Python built-in |
| 3.6 | `evaluate.py:108` | Quality | Classifier loaded at module import time |
| 3.7 | `test_visualize.py` / `visualize.py` | Quality | Three functions duplicated across files |
| 3.8 | `test_visualize.py:86`, `visualize.py:761` | Quality | `mse_ci` computed but never returned |
| 3.9 | `tests/test_model.py`, `tests/test_api.py` | Quality | Empty test files; no real test coverage |
| 3.10 | `sinkhorn.py:106–107` | Quality | Leftover debug shape comments |

---

*Generated by automated review — verify each finding against the current branch before acting on it.*
