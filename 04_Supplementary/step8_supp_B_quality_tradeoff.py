"""Evaluate the trade-off between data quality filtering and sample size."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/dataset_D_no_aggregation.csv"

FEATURE_COLS = [
    "CRUST1.0_moho_depth_0.5deg", "CRUST1.0_upper_crust_thickness_0.5deg",
    "CRUST1.0_mid_crust_thickness_0.5deg", "CRUST1.0_mantle_rho_0.5deg",
    "hotspot_min_hotspot_distance_km", "volcano_latest_vocano_dist",
    "topo_topo_mean", "topo_topo_diff", "topo_topo_median",
    "EMAG2_sealevel", "EMAG2_upcont", "LITH_IDW_lab", "LITH_IDW_moho",
    "oceanic_crust_age_Ma",
]
TARGET = "q"

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS + [TARGET])

def spatial_block_split(data, block_size=2.0, seed=42, min_per_block=3):
    d = data.copy()
    d["_bid"] = ((d["grid_lat"] // block_size) * block_size).astype(str) + "_" +\
                ((d["grid_lon"] // block_size) * block_size).astype(str)
    bc = d["_bid"].value_counts()
    d = d[d["_bid"].isin(bc[bc >= min_per_block].index)]
    rng = np.random.default_rng(seed)
    blocks = d["_bid"].unique()
    rng.shuffle(blocks)
    n_test = int(len(blocks) * 0.3)
    test_set = set(blocks[:n_test])
    tr = d[~d["_bid"].isin(test_set)]
    te = d[d["_bid"].isin(test_set)]
    return tr, te

def calc_moran_knn(coords, values, k=8):
    from scipy.spatial import cKDTree
    n = len(values)
    z = values - np.mean(values)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k+1)
    num = sum(z[i] * z[indices[i, j]] for i in range(n) for j in range(1, k+1))
    W = n * k
    return (n / W) * (num / np.sum(z**2))


DATASETS = {
    "A: M1+M1x":       ["M1", "M1x"],
    "B: M1-M2(含x)":   ["M1", "M1x", "M2", "M2x"],
    "C: M1-M3(含x)":   ["M1", "M1x", "M2", "M2x", "M3", "M3x"],
    "ALL: 排除Mx":     ["M1", "M1x", "M2", "M2x", "M3", "M3x", "M4", "M4x"],
    "D: 全部(含Mx)":   None,
}

print("=" * 90)
print("补充实验 B：数据质量 vs 数据量的权衡")
print("=" * 90)

print(f"\n{'数据集':<18} {'记录数':>8} {'网格数':>8} {'随机R²':>8} {'空间R²':>8} "
      f"{'空间RMSE':>10} {'空间MAE':>10} {'Moran I':>8}")
print("-" * 95)

from sklearn.model_selection import train_test_split

for ds_label, m_levels in DATASETS.items():
    if m_levels is None:
        sub = df.copy()
    else:
        sub = df[df["qc_m"].isin(m_levels)].copy()

    n_grids = sub[["grid_lat", "grid_lon"]].drop_duplicates().shape[0]

    if len(sub) < 50:
        print(f"{ds_label:<18} {len(sub):>8,} {n_grids:>8,}  样本不足")
        continue

    X, y = sub[FEATURE_COLS].values, sub[TARGET].values


    Xr, Xe, yr, ye = train_test_split(X, y, test_size=0.3, random_state=42)
    et = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et.fit(Xr, yr)
    r2_rand = r2_score(ye, et.predict(Xe))


    tr, te = spatial_block_split(sub)
    if len(te) < 30:
        print(f"{ds_label:<18} {len(sub):>8,} {n_grids:>8,} {r2_rand:>8.4f}  空间分组样本不足")
        continue

    et2 = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et2.fit(tr[FEATURE_COLS].values, tr[TARGET].values)
    pred_s = et2.predict(te[FEATURE_COLS].values)
    r2_sp = r2_score(te[TARGET].values, pred_s)
    rmse_sp = float(np.sqrt(mean_squared_error(te[TARGET].values, pred_s)))
    mae_sp = float(mean_absolute_error(te[TARGET].values, pred_s))


    res = te[TARGET].values - pred_s
    coords = te[["grid_lat", "grid_lon"]].values
    mi = calc_moran_knn(coords, res, k=8)

    print(f"{ds_label:<18} {len(sub):>8,} {n_grids:>8,} {r2_rand:>8.4f} {r2_sp:>8.4f} "
          f"{rmse_sp:>10.2f} {mae_sp:>10.2f} {mi:>8.4f}")

print("\n补充实验 B 完成!")
