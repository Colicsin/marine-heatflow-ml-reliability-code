"""Evaluate feature engineering, feature sensitivity, and spatial-block sensitivity experiments."""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/dataset_D_no_aggregation.csv"
FIG_DIR = ROOT / "outputs/figures"

FEATURE_COLS = [
    "CRUST1.0_moho_depth_0.5deg", "CRUST1.0_upper_crust_thickness_0.5deg",
    "CRUST1.0_mid_crust_thickness_0.5deg", "CRUST1.0_mantle_rho_0.5deg",
    "hotspot_min_hotspot_distance_km", "volcano_latest_vocano_dist",
    "topo_topo_mean", "topo_topo_diff", "topo_topo_median",
    "EMAG2_sealevel", "EMAG2_upcont", "LITH_IDW_lab", "LITH_IDW_moho",
    "oceanic_crust_age_Ma",
]
TARGET = "q"
EPS = 1e-3

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS + [TARGET])
print(f"Dataset D: {len(df):,} records\n")

def spatial_block_split(data, block_size=2.0, test_ratio=0.3, seed=42, min_per_block=3):
    d = data.copy()
    d["block_id"] = ((d["grid_lat"] // block_size) * block_size).astype(str) + "_" +\
                    ((d["grid_lon"] // block_size) * block_size).astype(str)
    bc = d["block_id"].value_counts()
    d = d[d["block_id"].isin(bc[bc >= min_per_block].index)]
    rng = np.random.default_rng(seed)
    blocks = d["block_id"].unique(); rng.shuffle(blocks)
    n_test = int(len(blocks) * test_ratio)
    test_blocks = set(blocks[:n_test])
    return d[~d["block_id"].isin(test_blocks)], d[d["block_id"].isin(test_blocks)]

def calc_moran_knn(coords, values, k=8):
    from scipy.spatial import cKDTree
    n = len(values); z = values - np.mean(values)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k+1)
    num = sum(z[i] * z[indices[i, j]] for i in range(n) for j in range(1, k+1))
    W = n * k
    return (n / W) * (num / np.sum(z**2))


print("=" * 80)
print("Experiment 5: feature engineering assessment, corresponding to manuscript Table 13 and Figure 7")
print("=" * 80)


df["hotspot_volcano_inter"] = df["hotspot_min_hotspot_distance_km"] / (df["volcano_latest_vocano_dist"] + EPS)
df["mantle_volcano_inter"] = df["CRUST1.0_mantle_rho_0.5deg"] * df["volcano_latest_vocano_dist"]
df["age_volcano_inter"] = df["oceanic_crust_age_Ma"] * df["volcano_latest_vocano_dist"]
df["age_mantle_inter"] = df["oceanic_crust_age_Ma"] * df["CRUST1.0_mantle_rho_0.5deg"]
df["hotspot_mantle_inter"] = df["hotspot_min_hotspot_distance_km"] * df["CRUST1.0_mantle_rho_0.5deg"]


df["inv_upper_crust"] = 1.0 / (df["CRUST1.0_upper_crust_thickness_0.5deg"] + EPS)
df["inv_moho_depth"] = 1.0 / (df["CRUST1.0_moho_depth_0.5deg"] + EPS)
df["log_hotspot_dist"] = np.log1p(df["hotspot_min_hotspot_distance_km"])
df["log_volcano_dist"] = np.log1p(df["volcano_latest_vocano_dist"])
df["sqrt_age"] = np.sqrt(df["oceanic_crust_age_Ma"].clip(lower=0))
df["inv_age"] = 1.0 / (df["oceanic_crust_age_Ma"].clip(lower=0) + EPS)

INTER = ["hotspot_volcano_inter", "mantle_volcano_inter", "age_volcano_inter",
         "age_mantle_inter", "hotspot_mantle_inter"]
TRANS = ["inv_upper_crust", "inv_moho_depth", "log_hotspot_dist",
         "log_volcano_dist", "sqrt_age", "inv_age"]


df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLS + INTER + TRANS + [TARGET])

FEAT_SETS = [
    ("A: 14base features", FEATURE_COLS),
    ("B: +5interactions(19)", FEATURE_COLS + INTER),
    ("C: +interactions+transforms(25)", FEATURE_COLS + INTER + TRANS),
]

print(f"\n{'scheme':<25} {'n_feat':>6} {'random R²':>8} {'spatial R²':>8} {'random RMSE':>10} {'spatial RMSE':>10}")
print("-" * 75)

for label, feat_cols in FEAT_SETS:
    X, y = df[feat_cols].values, df[TARGET].values

    from sklearn.model_selection import train_test_split
    Xr, Xe, yr, ye = train_test_split(X, y, test_size=0.3, random_state=42)
    et = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et.fit(Xr, yr)
    r2_rand = r2_score(ye, et.predict(Xe))
    rmse_rand = np.sqrt(mean_squared_error(ye, et.predict(Xe)))

    tr, te = spatial_block_split(df)
    et2 = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et2.fit(tr[feat_cols].values, tr[TARGET].values)
    pred_s = et2.predict(te[feat_cols].values)
    r2_sp = r2_score(te[TARGET].values, pred_s)
    rmse_sp = np.sqrt(mean_squared_error(te[TARGET].values, pred_s))
    print(f"{label:<25} {len(feat_cols):>6} {r2_rand:>8.4f} {r2_sp:>8.4f} {rmse_rand:>10.2f} {rmse_sp:>10.2f}")


all_25 = FEATURE_COLS + INTER + TRANS
et_full = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
et_full.fit(df[all_25].values, df[TARGET].values)
mdi = et_full.feature_importances_
mdi_order = np.argsort(mdi)[::-1]

print("\n--- 25feature MDI ranking ---")
for rank, i in enumerate(mdi_order):
    marker = " ★" if rank < 14 else " (removed)"
    print(f"  {rank+1:>2}. {all_25[i]:<35} MDI={mdi[i]:>8.2f}{marker}")


top14_cols = [all_25[i] for i in mdi_order[:14]]
Xr14, Xe14, yr14, ye14 = train_test_split(df[top14_cols].values, df[TARGET].values, test_size=0.3, random_state=42)
et14 = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
et14.fit(Xr14, yr14)
r2_14 = r2_score(ye14, et14.predict(Xe14))
rmse_14 = np.sqrt(mean_squared_error(ye14, et14.predict(Xe14)))
tr14, te14 = spatial_block_split(df)
et14s = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
et14s.fit(tr14[top14_cols].values, tr14[TARGET].values)
r2_14s = r2_score(te14[TARGET].values, et14s.predict(te14[top14_cols].values))
rmse_14s = np.sqrt(mean_squared_error(te14[TARGET].values, et14s.predict(te14[top14_cols].values)))
print(f"\n{'D: reduced to Top 14':<25} {14:>6} {r2_14:>8.4f} {r2_14s:>8.4f} {rmse_14:>10.2f} {rmse_14s:>10.2f}")


fig, ax = plt.subplots(figsize=(10, 6))
top14_names = [all_25[i].split("_0.5deg")[0].split("_km")[0] for i in mdi_order[:14]]
top14_mdi = [mdi[i] for i in mdi_order[:14]]
colors = []
for i in mdi_order[:14]:
    name = all_25[i]
    if name in INTER:
        colors.append("#e74c3c")
    elif name in TRANS:
        colors.append("#2ecc71")
    else:
        colors.append("#3498db")
ax.barh(range(13, -1, -1), top14_mdi, color=colors)
ax.set_yticks(range(13, -1, -1))
ax.set_yticklabels(top14_names, fontsize=9)
ax.set_xlabel("MDI Importance", fontsize=12)
ax.set_title("Top 14 Feature Importance (MDI, ExtraTrees)", fontsize=13)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#3498db", label="Original"),
                   Patch(facecolor="#e74c3c", label="Interaction"),
                   Patch(facecolor="#2ecc71", label="Transformed")]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "step7_mdi_top14.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nsaved figure: {FIG_DIR / 'step7_mdi_top14.png'}")


print("\n" + "=" * 80)
print("Experiment 6: feature-parameter sensitivity, corresponding to manuscript Figure 8")
print("=" * 80)


tr_s, te_s = spatial_block_split(df)
et_base = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
et_base.fit(tr_s[FEATURE_COLS].values, tr_s[TARGET].values)
r2_base = r2_score(te_s[TARGET].values, et_base.predict(te_s[FEATURE_COLS].values))
print(f"baseline R² (spatial block split): {r2_base:.4f}\n")

noise_levels = [5, 10, 20]
sensitivity = {f: [] for f in FEATURE_COLS}

SHORT_NAMES = [
    "moho_depth", "upper_crust", "mid_crust", "mantle_rho",
    "hotspot_dist", "volcano_dist", "topo_mean", "topo_diff",
    "topo_median", "EMAG2_sea", "EMAG2_up", "LITH_lab", "LITH_moho",
    "crust_age",
]

print(f"{'feature':<20}", end="")
for nl in noise_levels:
    print(f"  {'noise='+str(nl)+'%':>12}", end="")
print()
print("-" * 60)

for fi, feat in enumerate(FEATURE_COLS):
    for nl in noise_levels:
        rng = np.random.default_rng(42)
        te_noisy = te_s.copy()
        std_val = te_noisy[feat].std()
        noise = rng.normal(0, std_val * nl / 100, size=len(te_noisy))
        te_noisy[feat] = te_noisy[feat] + noise
        pred_noisy = et_base.predict(te_noisy[FEATURE_COLS].values)
        r2_noisy = r2_score(te_noisy[TARGET].values, pred_noisy)
        r2_drop = (r2_base - r2_noisy) / r2_base * 100
        sensitivity[feat].append(r2_drop)
    print(f"{SHORT_NAMES[fi]:<20}", end="")
    for drop in sensitivity[feat]:
        print(f"  {drop:>11.2f}%", end="")
    print()


fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(FEATURE_COLS))
width = 0.25
for i, nl in enumerate(noise_levels):
    drops = [sensitivity[f][i] for f in FEATURE_COLS]
    ax.bar(x + i * width, drops, width, label=f"Noise {nl}%")
ax.set_xticks(x + width)
ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("R² Drop (%)", fontsize=12)
ax.set_title("Feature Sensitivity to Noise Perturbation (Spatial Block 2°×2°)", fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "step7_feature_sensitivity.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\nsaved figure: {FIG_DIR / 'step7_feature_sensitivity.png'}")


print("\n" + "=" * 80)
print("Experiment 7: block-size sensitivity, corresponding to manuscript Table 14")
print("=" * 80)

block_sizes = [1.0, 1.5, 2.0, 4.0]
print(f"\n{'block size':<12} {'n_train':>8} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Moran I':>8}")
print("-" * 68)

for bs in block_sizes:
    tr_b, te_b = spatial_block_split(df, block_size=bs)
    if len(te_b) < 50:
        print(f"{bs}°×{bs}°     insufficient samples")
        continue
    et_b = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et_b.fit(tr_b[FEATURE_COLS].values, tr_b[TARGET].values)
    pred_b = et_b.predict(te_b[FEATURE_COLS].values)
    r2_b = r2_score(te_b[TARGET].values, pred_b)
    rmse_b = np.sqrt(mean_squared_error(te_b[TARGET].values, pred_b))
    mae_b = mean_absolute_error(te_b[TARGET].values, pred_b)
    res_b = te_b[TARGET].values - pred_b
    coords_b = te_b[["grid_lat", "grid_lon"]].values
    mi_b = calc_moran_knn(coords_b, res_b, k=8)
    print(f"{bs}°×{bs}°{'':<6} {len(tr_b):>8,} {len(te_b):>8,} {r2_b:>8.4f} {rmse_b:>8.2f} {mae_b:>8.2f} {mi_b:>8.4f}")

print("\n" + "=" * 80)
print("batch-2 experiments (feature engineering and sensitivity) all done!")
print("=" * 80)
