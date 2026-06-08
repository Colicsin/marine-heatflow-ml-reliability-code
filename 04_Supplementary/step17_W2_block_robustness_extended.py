"""Analyze spatial-block robustness across scales and random seeds."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error, median_absolute_error)
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree

ROOT      = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/dataset_D_no_aggregation.csv"
OUT_DIR   = ROOT / "outputs"
FIG_DIR   = ROOT / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "CRUST1.0_moho_depth_0.5deg",
    "CRUST1.0_upper_crust_thickness_0.5deg",
    "CRUST1.0_mid_crust_thickness_0.5deg",
    "CRUST1.0_mantle_rho_0.5deg",
    "hotspot_min_hotspot_distance_km",
    "volcano_latest_vocano_dist",
    "topo_topo_mean",
    "topo_topo_diff",
    "topo_topo_median",
    "EMAG2_sealevel",
    "EMAG2_upcont",
    "LITH_IDW_lab",
    "LITH_IDW_moho",
    "oceanic_crust_age_Ma",
]
TARGET = "q"
SEEDS = [42, 123, 456, 789, 2024]
BLOCK_SIZES = [1.0, 1.5, 2.0, 3.0, 4.0]


def spatial_block_split(data, block_size, seed, min_per_block=3):
    """Split samples by spatial blocks and return split metadata."""
    d = data.copy()
    d["block_id"] = (
        (d["grid_lat"] // block_size * block_size).astype(str) + "_" +
        (d["grid_lon"] // block_size * block_size).astype(str)
    )
    bc = d["block_id"].value_counts()
    valid_blocks = bc[bc >= min_per_block].index
    n_dropped = len(d) - d["block_id"].isin(valid_blocks).sum()
    d = d[d["block_id"].isin(valid_blocks)]

    rng = np.random.default_rng(seed)
    blocks = d["block_id"].unique()
    rng.shuffle(blocks)
    n_test = int(len(blocks) * 0.3)
    test_blocks = set(blocks[:n_test])

    tr = d[~d["block_id"].isin(test_blocks)]
    te = d[d["block_id"].isin(test_blocks)]

    meta = {
        "n_total_blocks": len(blocks) + (bc < min_per_block).sum(),
        "n_valid_blocks": len(blocks),
        "n_dropped_samples": n_dropped,
        "n_test_blocks": n_test,
        "n_train_blocks": len(blocks) - n_test,
    }
    return tr, te, meta


def calc_metrics(y_true, y_pred):
    return {
        "R2":    r2_score(y_true, y_pred),
        "RMSE":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":   float(mean_absolute_error(y_true, y_pred)),
        "MedAE": float(median_absolute_error(y_true, y_pred)),
        "Bias":  float(np.mean(y_pred - y_true)),
    }


def calc_moran_knn(coords, values, k=8):
    n = len(values)
    z = values - np.mean(values)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    num = sum(z[i] * z[indices[i, j]]
              for i in range(n) for j in range(1, k + 1))
    W = n * k
    return float((n / W) * (num / np.sum(z ** 2)))


def basin_distribution(data):
    """Return basin-level sample proportions."""
    if "basin" not in data.columns:
        return {}
    counts = data["basin"].value_counts(normalize=True)
    return counts.to_dict()


def heatflow_stats(data):
    """Return summary statistics for heat-flow values."""
    q = data[TARGET]
    return {
        "q_mean":   float(q.mean()),
        "q_median": float(q.median()),
        "q_std":    float(q.std()),
        "q_p10":    float(q.quantile(0.1)),
        "q_p90":    float(q.quantile(0.9)),
        "pct_high": float((q > 100).mean() * 100),
    }


print("=" * 72)
print("W2 supplementary experiment: block-size robustness and non-monotonicity analysis")
print("=" * 72)

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS + [TARGET])
df = df[df[TARGET] > 0].copy()
print(f"  observations: {len(df):,}\n")


print("=" * 72)
print("Part A: multi-seed, multi-scale experiment with a random-split baseline")
print("=" * 72)

all_results = []


print("\n--- random-split baseline ---")
for seed in SEEDS:
    X = df[FEATURE_COLS].values
    y = df[TARGET].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    et = ExtraTreesRegressor(n_estimators=100, max_depth=20,
                              random_state=42, n_jobs=-1)
    et.fit(X_tr, y_tr)
    pred = et.predict(X_te)
    m = calc_metrics(y_te, pred)
    coords_te = df.iloc[
        train_test_split(range(len(df)), test_size=0.3,
                         random_state=seed)[1]
    ][["grid_lat", "grid_lon"]].values
    moran_i = calc_moran_knn(coords_te, pred - y_te, k=8)

    row = {"block_size": "random", "seed": seed,
           "n_train": len(X_tr), "n_test": len(X_te),
           "n_dropped": 0, "n_blocks_total": "N/A",
           "n_blocks_valid": "N/A",
           **m, "Moran_I": moran_i}
    all_results.append(row)
    print(f"  random seed={seed:>4}: n_test={len(X_te):>6,}  "
          f"R²={m['R2']:.4f}  RMSE={m['RMSE']:.2f}  Moran_I={moran_i:.4f}")


print("\n--- spatial-block experiments ---")
composition_data = []

for bs in BLOCK_SIZES:
    print(f"\n  === {bs}°×{bs}° ===")
    for seed in SEEDS:
        tr, te, meta = spatial_block_split(df, bs, seed)
        if len(te) < 50:
            print(f"    seed={seed}: test set too small ({len(te)}), skipped")
            continue

        et = ExtraTreesRegressor(n_estimators=100, max_depth=20,
                                  random_state=42, n_jobs=-1)
        et.fit(tr[FEATURE_COLS].values, tr[TARGET].values)
        pred = et.predict(te[FEATURE_COLS].values)
        m = calc_metrics(te[TARGET].values, pred)
        coords_te = te[["grid_lat", "grid_lon"]].values
        moran_i = calc_moran_knn(coords_te, pred - te[TARGET].values, k=8)

        row = {"block_size": bs, "seed": seed,
               "n_train": len(tr), "n_test": len(te),
               "n_dropped": meta["n_dropped_samples"],
               "n_blocks_total": meta["n_total_blocks"],
               "n_blocks_valid": meta["n_valid_blocks"],
               **m, "Moran_I": moran_i}
        all_results.append(row)


        tr_basin = basin_distribution(tr)
        te_basin = basin_distribution(te)
        tr_hf = heatflow_stats(tr)
        te_hf = heatflow_stats(te)
        comp_row = {
            "block_size": bs, "seed": seed,
            "n_train": len(tr), "n_test": len(te),
            "n_dropped": meta["n_dropped_samples"],
            "R2": m["R2"], "RMSE": m["RMSE"],
        }
        for basin_name in ["Pacific", "Atlantic", "Indian", "Southern"]:
            comp_row[f"train_{basin_name}_pct"] = tr_basin.get(basin_name, 0) * 100
            comp_row[f"test_{basin_name}_pct"] = te_basin.get(basin_name, 0) * 100
        for prefix, stats in [("train", tr_hf), ("test", te_hf)]:
            for k, v in stats.items():
                comp_row[f"{prefix}_{k}"] = v
        composition_data.append(comp_row)

        print(f"    seed={seed:>4}: n_train={len(tr):>6,} n_test={len(te):>6,} "
              f"dropped={meta['n_dropped_samples']:>5,}  "
              f"R²={m['R2']:.4f}  RMSE={m['RMSE']:.2f}  Moran_I={moran_i:.4f}")


res_df = pd.DataFrame(all_results)
res_df.to_csv(OUT_DIR / "step17_W2_block_full_results.csv", index=False)
print(f"\nsaved: outputs/step17_W2_block_full_results.csv")


comp_df = pd.DataFrame(composition_data)
comp_df.to_csv(OUT_DIR / "step17_W2_block_composition_analysis.csv", index=False)
print(f"saved: outputs/step17_W2_block_composition_analysis.csv")


print("\n" + "=" * 72)
print("Part B: summary statistics by scale (mean ± std)")
print("=" * 72)

print(f"\n{'scale':<12} {'R² mean':>10} {'R² std':>10} {'RMSE mean':>12} "
      f"{'RMSE std':>10} {'Moran mean':>12} {'n_test mean':>12} "
      f"{'n_dropped':>12}")
print("-" * 100)


sub_rand = res_df[res_df["block_size"] == "random"]
print(f"{'random':<12} {sub_rand['R2'].mean():>10.4f} "
      f"{sub_rand['R2'].std():>10.4f} "
      f"{sub_rand['RMSE'].mean():>12.2f} "
      f"{sub_rand['RMSE'].std():>10.2f} "
      f"{sub_rand['Moran_I'].mean():>12.4f} "
      f"{sub_rand['n_test'].mean():>12.0f} "
      f"{'0':>12}")


summary_rows = []
for bs in BLOCK_SIZES:
    sub = res_df[res_df["block_size"] == bs]
    if len(sub) == 0:
        continue
    row = {
        "block_size": f"{bs}°",
        "R2_mean": sub["R2"].mean(),
        "R2_std": sub["R2"].std(),
        "RMSE_mean": sub["RMSE"].mean(),
        "RMSE_std": sub["RMSE"].std(),
        "Moran_I_mean": sub["Moran_I"].mean(),
        "Moran_I_std": sub["Moran_I"].std(),
        "n_test_mean": sub["n_test"].mean(),
        "n_dropped_mean": sub["n_dropped"].mean(),
    }
    summary_rows.append(row)
    print(f"{bs}°×{bs}°{'':<6} {row['R2_mean']:>10.4f} "
          f"{row['R2_std']:>10.4f} "
          f"{row['RMSE_mean']:>12.2f} "
          f"{row['RMSE_std']:>10.2f} "
          f"{row['Moran_I_mean']:>12.4f} "
          f"{row['n_test_mean']:>12.0f} "
          f"{row['n_dropped_mean']:>12.0f}")


r2_random_mean = sub_rand["R2"].mean()
print(f"\nrandom-split mean R² = {r2_random_mean:.4f}")
print("Spatial-block ΔR² values (random - spatial):")
for bs in BLOCK_SIZES:
    sub = res_df[res_df["block_size"] == bs]
    if len(sub) == 0:
        continue
    delta = r2_random_mean - sub["R2"].mean()
    print(f"  {bs}°×{bs}°: ΔR² = {delta:.4f} "
          f"(relative decrease {delta / r2_random_mean * 100:.1f}%)")


print("\n" + "=" * 72)
print("Part C: attribution of non-monotonicity through test-set composition analysis")
print("=" * 72)

if len(comp_df) > 0:
    print("\nTest-set composition differences by scale (seed=42 example):")
    print(f"{'scale':<10} {'n_test':>8} {'dropped':>8} "
          f"{'Pac%':>8} {'Atl%':>8} {'Ind%':>8} "
          f"{'q_mean':>8} {'q_std':>8} {'high%':>8} {'R²':>8}")
    print("-" * 90)
    for bs in BLOCK_SIZES:
        row = comp_df[(comp_df["block_size"] == bs) &
                      (comp_df["seed"] == 42)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        print(f"{bs}°×{bs}°{'':<4} {r['n_test']:>8.0f} "
              f"{r['n_dropped']:>8.0f} "
              f"{r['test_Pacific_pct']:>8.1f} "
              f"{r['test_Atlantic_pct']:>8.1f} "
              f"{r['test_Indian_pct']:>8.1f} "
              f"{r['test_q_mean']:>8.1f} "
              f"{r['test_q_std']:>8.1f} "
              f"{r['test_pct_high']:>8.1f} "
              f"{r['R2']:>8.4f}")


    print("\nCross-seed variation in test-set composition by scale:")
    print(f"{'scale':<10} {'n_test CV%':>12} {'q_mean CV%':>12} "
          f"{'Pac% range':>12} {'high% range':>12}")
    print("-" * 62)
    for bs in BLOCK_SIZES:
        sub = comp_df[comp_df["block_size"] == bs]
        if len(sub) < 2:
            continue
        n_cv = sub["n_test"].std() / sub["n_test"].mean() * 100
        q_cv = sub["test_q_mean"].std() / sub["test_q_mean"].mean() * 100
        pac_range = sub["test_Pacific_pct"].max() - sub["test_Pacific_pct"].min()
        high_range = sub["test_pct_high"].max() - sub["test_pct_high"].min()
        print(f"{bs}°×{bs}°{'':<4} {n_cv:>12.1f} {q_cv:>12.1f} "
              f"{pac_range:>12.1f} {high_range:>12.1f}")


    print("\nNon-monotonicity explanation:")
    for bs_a, bs_b in [(1.5, 2.0)]:
        sub_a = comp_df[comp_df["block_size"] == bs_a]
        sub_b = comp_df[comp_df["block_size"] == bs_b]
        if len(sub_a) == 0 or len(sub_b) == 0:
            continue
        print(f"\n  {bs_a}° vs {bs_b}° comparison:")
        print(f"    {bs_a}°: R² mean={sub_a['R2'].mean():.4f}  "
              f"n_dropped mean={sub_a['n_dropped'].mean():.0f}  "
              f"n_test mean={sub_a['n_test'].mean():.0f}")
        print(f"    {bs_b}°: R² mean={sub_b['R2'].mean():.4f}  "
              f"n_dropped mean={sub_b['n_dropped'].mean():.0f}  "
              f"n_test mean={sub_b['n_test'].mean():.0f}")
        drop_diff = sub_a["n_dropped"].mean() - sub_b["n_dropped"].mean()
        print(f"    {bs_a}° removes {drop_diff:.0f} more samples than {bs_b}°")
        print(f"    → Smaller block sizes create more sparse groups (<3 samples),"
              f"and removing them shifts the test-set composition")


print("\n" + "=" * 72)
print("Part D: visualization")
print("=" * 72)


print("plotting Figure 1: R² distribution boxplot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))


ax = axes[0]
positions = list(range(len(BLOCK_SIZES) + 1))
labels_x = ["Random"] + [f"{bs}°" for bs in BLOCK_SIZES]
colors = ["#8da0cb"] + ["#fc8d62"] * len(BLOCK_SIZES)

bp_data = [sub_rand["R2"].values]
for bs in BLOCK_SIZES:
    sub = res_df[res_df["block_size"] == bs]
    bp_data.append(sub["R2"].values if len(sub) > 0 else np.array([]))

bp = ax.boxplot(bp_data, positions=positions, widths=0.5, patch_artist=True,
                showmeans=True, meanprops=dict(marker="D", markerfacecolor="red",
                                                markersize=6))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)


for pos, data_arr, color in zip(positions, bp_data, colors):
    if len(data_arr) > 0:
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(data_arr))
        ax.scatter([pos + j for j in jitter], data_arr,
                   c="black", s=25, zorder=5, alpha=0.7)

ax.set_xticks(positions)
ax.set_xticklabels(labels_x, fontsize=10)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("R² Distribution by Validation Strategy\n(5 seeds per setting)",
             fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(sub_rand["R2"].mean(), color="#8da0cb", linewidth=1,
           linestyle="--", alpha=0.5)


ax = axes[1]
delta_means = []
delta_stds = []
for bs in BLOCK_SIZES:
    sub = res_df[res_df["block_size"] == bs]
    if len(sub) > 0:
        delta_means.append(r2_random_mean - sub["R2"].mean())
        delta_stds.append(sub["R2"].std())
    else:
        delta_means.append(0)
        delta_stds.append(0)

x_pos = range(len(BLOCK_SIZES))
bars = ax.bar(x_pos, delta_means, yerr=delta_stds, capsize=5,
              color="#fc8d62", alpha=0.8, edgecolor="white")
for bar, val in zip(bars, delta_means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{bs}°" for bs in BLOCK_SIZES], fontsize=10)
ax.set_ylabel("ΔR² (Random − Spatial)", fontsize=12)
ax.set_title("Overestimation by Random Split\n(higher = more overestimation)",
             fontsize=12, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(FIG_DIR / "step17_W2_r2_by_blocksize_seeds.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("  saved: step17_W2_r2_by_blocksize_seeds.png")


print("plotting Figure 2: test-set composition analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

if len(comp_df) > 0:

    ax = axes[0, 0]
    for bs in BLOCK_SIZES:
        sub = comp_df[comp_df["block_size"] == bs]
        ax.scatter(sub["n_test"], sub["R2"], s=50, alpha=0.8,
                   label=f"{bs}°", zorder=3)
    ax.set_xlabel("Test Set Size", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("Test Set Size vs R²", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


    ax = axes[0, 1]
    for bs in BLOCK_SIZES:
        sub = comp_df[comp_df["block_size"] == bs]
        ax.scatter(sub["n_dropped"], sub["R2"], s=50, alpha=0.8,
                   label=f"{bs}°", zorder=3)
    ax.set_xlabel("Dropped Samples (sparse blocks)", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("Dropped Samples vs R²", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


    ax = axes[1, 0]
    for bs in BLOCK_SIZES:
        sub = comp_df[comp_df["block_size"] == bs]
        ax.scatter(sub["test_pct_high"], sub["R2"], s=50, alpha=0.8,
                   label=f"{bs}°", zorder=3)
    ax.set_xlabel("Test Set High HF (>100 mW/m²) %", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("High Heat Flow Proportion vs R²", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


    ax = axes[1, 1]
    for bs in BLOCK_SIZES:
        sub = comp_df[comp_df["block_size"] == bs]
        ax.scatter(sub["test_q_mean"], sub["R2"], s=50, alpha=0.8,
                   label=f"{bs}°", zorder=3)
    ax.set_xlabel("Test Set Mean Heat Flow (mW/m²)", fontsize=11)
    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("Test Set Mean HF vs R²", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("W2: Test Set Composition Analysis  -  What Drives R² Variation?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step17_W2_composition_analysis.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("  saved: step17_W2_composition_analysis.png")


print("plotting Figure 3: overview comparison of random and spatial-block validation...")
fig, ax = plt.subplots(figsize=(12, 6))


seed_colors = {42: "#e41a1c", 123: "#377eb8", 456: "#4daf4a",
               789: "#984ea3", 2024: "#ff7f00"}
x_labels = ["Random"] + [f"{bs}°" for bs in BLOCK_SIZES]
x_pos_line = list(range(len(x_labels)))

for seed in SEEDS:
    r2_line = []

    r2_rand = res_df[(res_df["block_size"] == "random") &
                     (res_df["seed"] == seed)]["R2"].values
    r2_line.append(r2_rand[0] if len(r2_rand) > 0 else np.nan)

    for bs in BLOCK_SIZES:
        r2_sp = res_df[(res_df["block_size"] == bs) &
                       (res_df["seed"] == seed)]["R2"].values
        r2_line.append(r2_sp[0] if len(r2_sp) > 0 else np.nan)
    ax.plot(x_pos_line, r2_line, "o-", color=seed_colors[seed],
            linewidth=1.5, markersize=6, alpha=0.8, label=f"seed={seed}")


r2_means = [sub_rand["R2"].mean()]
for bs in BLOCK_SIZES:
    sub = res_df[res_df["block_size"] == bs]
    r2_means.append(sub["R2"].mean() if len(sub) > 0 else np.nan)
ax.plot(x_pos_line, r2_means, "s--", color="black", linewidth=2.5,
        markersize=8, label="Mean", zorder=10)

ax.set_xticks(x_pos_line)
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("R² Across Validation Strategies  -  All Seeds\n"
             "(Any reasonable spatial blocking significantly reduces R²)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, ncol=3)
ax.spines[["top", "right"]].set_visible(False)
ax.axhspan(0, 0.35, alpha=0.05, color="red")

plt.tight_layout()
fig.savefig(FIG_DIR / "step17_W2_random_vs_spatial_all.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("  saved: step17_W2_random_vs_spatial_all.png")


print("\n" + "=" * 72)
print("W2 supplementary experiment all done!")
print(f"  CSV   outputs/step17_W2_block_full_results.csv")
print(f"  CSV   outputs/step17_W2_block_composition_analysis.csv")
print(f"  Figure 1   step17_W2_r2_by_blocksize_seeds.png")
print(f"  Figure 2   step17_W2_composition_analysis.png")
print(f"  Figure 3   step17_W2_random_vs_spatial_all.png")
print("=" * 72)
