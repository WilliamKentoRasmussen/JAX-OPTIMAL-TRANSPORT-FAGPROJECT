import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

from main_project.environment import LABELS, OPTIMAL_GAMMA


# ── Shared style ────────────────────────────────────────────────────────────────
PALETTE = [
    "#2c5f8a", "#c0392b", "#27ae60", "#8e44ad",
    "#e67e22", "#16a085", "#d35400", "#2980b9",
]


def _ci95(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(vals)
    ci = 1.96 * np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, mean - ci, mean + ci


# ── Figure A: Metric correlation heatmap ───────────────────────────────────────
def fig_A_metric_correlation_heatmap(summary_df, save_dir="figures/plots", save=True):
    """
    Pearson correlation matrix across all evaluation metrics at OPTIMAL_GAMMA.
    Answers: which metrics are redundant and which are independent?
    """
    os.makedirs(save_dir, exist_ok=True)

    df = summary_df[summary_df["gamma"] == OPTIMAL_GAMMA].copy()

    metric_cols = [
        "entropy",
        "wasserstein_distance_latent",
        "mmd_latent",
        "mmd_image",
        "classifier_confidence_image",
    ]
    labels = [
        "Entropy\n$H(P)$",
        "Wasserstein\n(latent)",
        "MMD\n(latent)",
        "MMD\n(image)",
        "Classifier\nconfidence",
    ]

    n = len(metric_cols)
    corr = np.full((n, n), np.nan)
    pvals = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            x = df[metric_cols[i]].dropna().values.astype(float)
            y = df[metric_cols[j]].dropna().values.astype(float)
            # align on shared index
            idx = df[metric_cols[i]].notna() & df[metric_cols[j]].notna()
            x = df.loc[idx, metric_cols[i]].values.astype(float)
            y = df.loc[idx, metric_cols[j]].values.astype(float)
            if len(x) > 2:
                r, p = pearsonr(x, y)
                corr[i, j] = r
                pvals[i, j] = p

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, label="Pearson r", fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(
        f"Metric Correlation Matrix  (γ = {OPTIMAL_GAMMA})\n"
        "* p < 0.05   ** p < 0.01   *** p < 0.001",
        fontsize=11,
    )

    for i in range(n):
        for j in range(n):
            r = corr[i, j]
            p = pvals[i, j]
            if np.isnan(r):
                continue
            stars = ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            txt = f"{r:.2f}{stars}"
            color = "white" if abs(r) > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    if save:
        path = os.path.join(save_dir, "metric_correlation_heatmap.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Figure B: Per digit-pair transport difficulty ranking ──────────────────────
def fig_B_digit_pair_difficulty_ranking(
    summary_df,
    top_n=20,
    save_dir="figures/plots",
    save=True,
):
    """
    Horizontal bar chart ranking digit pairs by mean MMD image at OPTIMAL_GAMMA.
    Shows the top_n hardest and top_n easiest pairs.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = summary_df[summary_df["gamma"] == OPTIMAL_GAMMA].copy()

    grouped = (
        df.groupby(["source_label", "target_label"])["mmd_image"]
        .mean()
        .reset_index()
        .rename(columns={"mmd_image": "mean_mmd"})
    )
    # exclude self-transport if present
    grouped = grouped[grouped["source_label"] != grouped["target_label"]]
    grouped["pair"] = grouped.apply(
        lambda r: f"{int(r.source_label)}→{int(r.target_label)}", axis=1
    )
    grouped = grouped.sort_values("mean_mmd", ascending=False).reset_index(drop=True)

    hardest = grouped.head(top_n)
    easiest = grouped.tail(top_n).sort_values("mean_mmd", ascending=True)

    source_colors = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(LABELS)}

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, top_n * 0.35)))
    fig.suptitle(
        f"Digit-pair transport difficulty  (γ = {OPTIMAL_GAMMA}, ranked by MMD image)",
        fontsize=12,
    )

    for ax, subset, title in [
        (axes[0], hardest, f"Hardest {top_n} pairs"),
        (axes[1], easiest, f"Easiest {top_n} pairs"),
    ]:
        colors = [source_colors[int(r.source_label)] for _, r in subset.iterrows()]
        bars = ax.barh(subset["pair"], subset["mean_mmd"], color=colors, edgecolor="white", linewidth=0.4)
        ax.set_xlabel("Mean MMD (image space)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, (_, row) in zip(bars, subset.iterrows()):
            ax.text(
                bar.get_width() + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{row.mean_mmd:.3f}",
                va="center",
                fontsize=7.5,
            )

    legend_elements = [
        Line2D([0], [0], marker="s", color="w",
               markerfacecolor=source_colors[lab], markersize=10,
               label=f"Source {lab}")
        for lab in LABELS
    ]
    fig.legend(
        handles=legend_elements,
        title="Source digit",
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout()
    if save:
        path = os.path.join(save_dir, "digit_pair_difficulty_ranking.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Figure C: Latent dim × digit pair interaction ──────────────────────────────
def fig_C_dim_vs_pair_mmd(
    summary_df,
    n_hardest=5,
    n_easiest=5,
    save_dir="figures/plots",
    save=True,
):
    """
    Line plot: x = latent_dim, y = mean MMD image, one line per digit pair.
    Only the n_hardest and n_easiest pairs (at OPTIMAL_GAMMA, averaged over dims)
    are shown to keep the plot readable.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = summary_df[summary_df["gamma"] == OPTIMAL_GAMMA].copy()
    df = df[df["source_label"] != df["target_label"]]

    latent_dims = sorted(df["latent_dim"].unique())

    # Rank pairs by overall mean MMD
    pair_means = (
        df.groupby(["source_label", "target_label"])["mmd_image"]
        .mean()
        .reset_index()
        .sort_values("mmd_image", ascending=False)
    )
    hardest = list(
        pair_means.head(n_hardest)[["source_label", "target_label"]].itertuples(index=False, name=None)
    )
    easiest = list(
        pair_means.tail(n_easiest)[["source_label", "target_label"]].itertuples(index=False, name=None)
    )
    selected = hardest + easiest

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (src, tgt) in enumerate(selected):
        means, lowers, uppers = [], [], []
        for dim in latent_dims:
            vals = df.loc[
                (df["latent_dim"] == dim)
                & (df["source_label"] == src)
                & (df["target_label"] == tgt),
                "mmd_image",
            ].values
            m, lo, hi = _ci95(vals)
            means.append(m)
            lowers.append(lo)
            uppers.append(hi)

        is_hard = idx < n_hardest
        color = PALETTE[idx % len(PALETTE)]
        ls = "-" if is_hard else "--"
        label = f"{int(src)}→{int(tgt)}  {'(hard)' if is_hard else '(easy)'}"

        ax.plot(latent_dims, means, marker="o", linewidth=1.8,
                linestyle=ls, color=color, label=label)
        ax.fill_between(latent_dims, lowers, uppers, alpha=0.10, color=color)

    ax.set_xlabel("Latent dimension $d$", fontsize=11)
    ax.set_ylabel("Mean MMD (image space)", fontsize=11)
    ax.set_title(
        f"Latent dim × digit-pair interaction  (γ = {OPTIMAL_GAMMA})\n"
        f"Solid = {n_hardest} hardest pairs · Dashed = {n_easiest} easiest pairs",
        fontsize=11,
    )
    ax.legend(fontsize=8.5, frameon=False, ncol=2, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, "dim_vs_pair_mmd.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Figure D: MMD latent vs MMD image scatter ──────────────────────────────────
def fig_D_latent_vs_image_mmd_scatter(summary_df, save_dir="figures/plots", save=True):
    """
    Scatter: x = mmd_latent, y = mmd_image, coloured by latent_dim.
    Reveals whether good latent transport reliably produces good image transport.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = summary_df[summary_df["gamma"] == OPTIMAL_GAMMA].copy()
    latent_dims = sorted(df["latent_dim"].unique())
    cmap = plt.cm.viridis
    colors = {dim: cmap(i / max(1, len(latent_dims) - 1)) for i, dim in enumerate(latent_dims)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for dim in latent_dims:
        sub = df[df["latent_dim"] == dim]
        ax.scatter(
            sub["mmd_latent"].astype(float),
            sub["mmd_image"].astype(float),
            color=colors[dim],
            alpha=0.45,
            s=22,
            label=f"dim={dim}",
            edgecolors="none",
        )

    # Overall correlation line
    x_all = df["mmd_latent"].astype(float).values
    y_all = df["mmd_image"].astype(float).values
    mask = ~(np.isnan(x_all) | np.isnan(y_all))
    x_all, y_all = x_all[mask], y_all[mask]
    r, p = pearsonr(x_all, y_all)
    m, b = np.polyfit(x_all, y_all, 1)
    xs = np.linspace(x_all.min(), x_all.max(), 200)
    ax.plot(xs, m * xs + b, color="#c0392b", linewidth=1.5, linestyle="--",
            label=f"OLS fit  r={r:.2f}, p={p:.3f}")

    ax.set_xlabel("MMD (latent space)", fontsize=11)
    ax.set_ylabel("MMD (image space)", fontsize=11)
    ax.set_title(
        f"Latent vs image MMD  (γ = {OPTIMAL_GAMMA})\n"
        "Does good latent transport → good image transport?",
        fontsize=11,
    )
    ax.legend(fontsize=9, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, "latent_vs_image_mmd_scatter.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Figure E: Wasserstein vs MMD comparison across dims ───────────────────────
def fig_E_wasserstein_vs_mmd_across_dims(summary_df, save_dir="figures/plots", save=True):
    """
    Dual-axis line plot: Wasserstein (latent) and MMD (latent) vs latent_dim,
    both normalised to [0,1]. Do the two metrics agree?
    """
    os.makedirs(save_dir, exist_ok=True)

    df = summary_df[summary_df["gamma"] == OPTIMAL_GAMMA].copy()
    latent_dims = sorted(df["latent_dim"].unique())

    def _collect(col):
        means, lowers, uppers = [], [], []
        for dim in latent_dims:
            vals = df.loc[df["latent_dim"] == dim, col].astype(float).values
            m, lo, hi = _ci95(vals)
            means.append(m); lowers.append(lo); uppers.append(hi)
        return np.array(means), np.array(lowers), np.array(uppers)

    def _norm(arr):
        lo, hi = np.nanmin(arr), np.nanmax(arr)
        return (arr - lo) / (hi - lo + 1e-12)

    wass_m, wass_lo, wass_hi = _collect("wasserstein_distance_latent")
    mmd_m,  mmd_lo,  mmd_hi  = _collect("mmd_latent")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(latent_dims, _norm(wass_m), marker="o", linewidth=2,
            color=PALETTE[0], label="Wasserstein (latent)")
    ax.fill_between(latent_dims, _norm(wass_lo), _norm(wass_hi),
                    alpha=0.15, color=PALETTE[0])

    ax.plot(latent_dims, _norm(mmd_m), marker="s", linewidth=2,
            linestyle="--", color=PALETTE[1], label="MMD (latent)")
    ax.fill_between(latent_dims, _norm(mmd_lo), _norm(mmd_hi),
                    alpha=0.15, color=PALETTE[1])

    ax.set_xlabel("Latent dimension $d$", fontsize=11)
    ax.set_ylabel("Normalised metric value  [0, 1]", fontsize=11)
    ax.set_title(
        f"Wasserstein vs MMD across latent dims  (γ = {OPTIMAL_GAMMA})\n"
        "Both normalised — divergence indicates disagreement",
        fontsize=11,
    )
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, "wasserstein_vs_mmd_across_dims.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Figure F: Running time vs MMD Pareto front ────────────────────────────────
def fig_F_pareto_time_vs_mmd(summary_df, save_dir="figures/plots", save=True):
    """
    Scatter: x = mean running time, y = mean MMD image, one point per latent_dim.
    Points on the Pareto front (best quality per unit time) are highlighted.
    """
    os.makedirs(save_dir, exist_ok=True)

    df = summary_df[summary_df["gamma"] == OPTIMAL_GAMMA].copy()
    latent_dims = sorted(df["latent_dim"].unique())

    times, mmds, time_cis, mmd_cis = [], [], [], []
    for dim in latent_dims:
        sub = df[df["latent_dim"] == dim]
        t_m, t_lo, t_hi = _ci95(sub["running_time"].astype(float).values)
        m_m, m_lo, m_hi = _ci95(sub["mmd_image"].astype(float).values)
        times.append(t_m)
        mmds.append(m_m)
        time_cis.append(t_m - t_lo)
        mmd_cis.append(m_m - m_lo)

    times = np.array(times)
    mmds = np.array(mmds)

    # Pareto front: minimise both time and MMD
    def _is_pareto(costs):
        n = len(costs)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                    dominated[i] = True
                    break
        return ~dominated

    costs = np.column_stack([times, mmds])
    pareto_mask = _is_pareto(costs)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Error bars
    ax.errorbar(
        times, mmds,
        xerr=time_cis, yerr=mmd_cis,
        fmt="none", color="#aaaaaa", linewidth=0.9, capsize=3, zorder=1,
    )

    # All points
    sc = ax.scatter(
        times, mmds,
        c=[PALETTE[i % len(PALETTE)] for i in range(len(latent_dims))],
        s=90, zorder=3, edgecolors="white", linewidth=0.8,
    )

    # Highlight Pareto front
    pareto_times = times[pareto_mask]
    pareto_mmds = mmds[pareto_mask]
    order = np.argsort(pareto_times)
    ax.plot(pareto_times[order], pareto_mmds[order],
            color="#c0392b", linewidth=1.4, linestyle="--",
            zorder=2, label="Pareto front")
    ax.scatter(pareto_times, pareto_mmds,
               s=130, facecolors="none", edgecolors="#c0392b",
               linewidth=1.8, zorder=4, label="Pareto-optimal dim")

    # Label each point
    for i, dim in enumerate(latent_dims):
        ax.annotate(
            f"d={dim}",
            (times[i], mmds[i]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    ax.set_xlabel("Mean running time per pair (s)", fontsize=11)
    ax.set_ylabel("Mean MMD (image space)", fontsize=11)
    ax.set_title(
        f"Running time vs transport quality  (γ = {OPTIMAL_GAMMA})\n"
        "Bottom-left = fast AND good  ·  Pareto front highlighted",
        fontsize=11,
    )
    ax.legend(fontsize=10, frameon=False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save:
        path = os.path.join(save_dir, "pareto_time_vs_mmd.png")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.show()


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary_df = pd.read_csv("data/evaluation_summary.csv")
    fig_A_metric_correlation_heatmap(summary_df)
    fig_B_digit_pair_difficulty_ranking(summary_df)
    fig_C_dim_vs_pair_mmd(summary_df)
    fig_D_latent_vs_image_mmd_scatter(summary_df)
    fig_E_wasserstein_vs_mmd_across_dims(summary_df)
    fig_F_pareto_time_vs_mmd(summary_df)