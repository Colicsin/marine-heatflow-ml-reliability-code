"""
补充实验 A：Kriging 插值标签 vs 真实观测标签的直接对比
目的：证明插值标签导致性能虚高
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
FEAT_PATH = ROOT / "data/features/Ocean_HeatFlow_Prediction_Data_with_Age.csv"
OBS_PATH  = ROOT / "data/processed/dataset_D_no_aggregation.csv"

FEATURE_COLS = [
    "CRUST1.0_moho_depth_0.5deg", "CRUST1.0_upper_crust_thickness_0.5deg",
    "CRUST1.0_mid_crust_thickness_0.5deg", "CRUST1.0_mantle_rho_0.5deg",
    "hotspot_min_hotspot_distance_km", "volcano_latest_vocano_dist",
    "topo_topo_mean", "topo_topo_diff", "topo_topo_median",
    "EMAG2_sealevel", "EMAG2_upcont", "LITH_IDW_lab", "LITH_IDW_moho",
    "oceanic_crust_age_Ma",
]

def spatial_block_split(data, lat_col, lon_col, block_size=2.0,
                        test_ratio=0.3, seed=42, min_per_block=3):
    d = data.copy()
    d["_bid"] = ((d[lat_col] // block_size) * block_size).astype(str) + "_" + \
                ((d[lon_col] // block_size) * block_size).astype(str)
    bc = d["_bid"].value_counts()
    d = d[d["_bid"].isin(bc[bc >= min_per_block].index)]
    rng = np.random.default_rng(seed)
    blocks = d["_bid"].unique()
    rng.shuffle(blocks)
    n_test = int(len(blocks) * test_ratio)
    test_set = set(blocks[:n_test])
    tr = d[~d["_bid"].isin(test_set)].drop(columns=["_bid"])
    te = d[d["_bid"].isin(test_set)].drop(columns=["_bid"])
    return tr, te, len(blocks)

def calc_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }

def assign_basin(lon, lat):
    if lat < -60: return "Southern"
    if lon < -20 or lon > 140: return "Pacific"
    if -20 <= lon <= 20: return "Atlantic"
    return "Indian"

# ═══════════════════════════════════════════════════════════════════
# 1. 准备 Kriging 标签数据集
# ═══════════════════════════════════════════════════════════════════
print("=" * 80)
print("补充实验 A：Kriging 插值标签 vs 真实观测标签")
print("=" * 80)

kriging = pd.read_csv(FEAT_PATH)
kriging = kriging.rename(columns={"heatflow_heatflow-value": "q",
                                   "lon": "grid_lon", "lat": "grid_lat"})
kriging["grid_lon"] = kriging["grid_lon"].round(2)
kriging["grid_lat"] = kriging["grid_lat"].round(2)
kriging["oceanic_crust_age_Ma"] = kriging["oceanic_crust_age_Ma"].fillna(-1.0)
kriging = kriging.dropna(subset=FEATURE_COLS + ["q"])
kriging["basin"] = kriging.apply(
    lambda r: assign_basin(r["grid_lon"], r["grid_lat"]), axis=1)
print(f"\nKriging 标签数据集: {len(kriging):,} 网格点")
print(f"  q: mean={kriging['q'].mean():.1f}, std={kriging['q'].std():.1f}")

# ═══════════════════════════════════════════════════════════════════
# 2. 准备真实观测数据集
# ═══════════════════════════════════════════════════════════════════
obs = pd.read_csv(OBS_PATH)
obs = obs.dropna(subset=FEATURE_COLS + ["q"])
print(f"\n真实观测数据集: {len(obs):,} 条记录, "
      f"{obs[['grid_lat','grid_lon']].drop_duplicates().shape[0]:,} 网格")
print(f"  q: mean={obs['q'].mean():.1f}, std={obs['q'].std():.1f}")

# ═══════════════════════════════════════════════════════════════════
# 3. 对比实验
# ═══════════════════════════════════════════════════════════════════
datasets = {
    "Kriging标签(123k网格)": (kriging, FEATURE_COLS, "q", "grid_lat", "grid_lon"),
    "真实观测(28k记录)":     (obs,     FEATURE_COLS, "q", "grid_lat", "grid_lon"),
}

results = []

for ds_name, (data, feat_cols, target, lat_col, lon_col) in datasets.items():
    X = data[feat_cols].values
    y = data[target].values

    print(f"\n{'='*70}")
    print(f"数据集: {ds_name}")
    print(f"{'='*70}")

    # 方案1：随机划分 7:3
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    et = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et.fit(X_tr, y_tr)
    m = calc_metrics(y_te, et.predict(X_te))
    results.append((ds_name, "随机划分", len(y_te), m))
    print(f"  随机划分:     n_test={len(y_te):>7,}  R²={m['R2']:.4f}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}")

    # 方案2：空间分组 2°×2°
    tr, te, n_blocks = spatial_block_split(data, lat_col, lon_col)
    et2 = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et2.fit(tr[feat_cols].values, tr[target].values)
    m2 = calc_metrics(te[target].values, et2.predict(te[feat_cols].values))
    results.append((ds_name, "空间分组2°×2°", len(te), m2))
    print(f"  空间分组2°×2°: n_test={len(te):>7,}  R²={m2['R2']:.4f}  RMSE={m2['RMSE']:.2f}  MAE={m2['MAE']:.2f}")

    # 方案3：跨洋盆
    for basin in ["Pacific", "Atlantic", "Indian"]:
        tr_b = data[data["basin"] != basin]
        te_b = data[data["basin"] == basin]
        if len(te_b) > 10 and len(tr_b) > 10:
            et3 = ExtraTreesRegressor(n_estimators=100, max_depth=20,
                                       random_state=42, n_jobs=-1)
            et3.fit(tr_b[feat_cols].values, tr_b[target].values)
            m3 = calc_metrics(te_b[target].values, et3.predict(te_b[feat_cols].values))
            results.append((ds_name, f"跨盆-{basin}", len(te_b), m3))
            print(f"  跨盆-{basin:<9} n_test={len(te_b):>7,}  R²={m3['R2']:.4f}  RMSE={m3['RMSE']:.2f}  MAE={m3['MAE']:.2f}")

# ═══════════════════════════════════════════════════════════════════
# 4. 汇总对比表
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("汇总对比表：Kriging 标签 vs 真实观测")
print("=" * 90)
print(f"{'数据集':<22} {'验证方案':<16} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8}")
print("-" * 82)
for ds_name, scheme, n, m in results:
    tag = "K" if "Kriging" in ds_name else "O"
    print(f"[{tag}] {scheme:<18} {n:>8,} {m['R2']:>8.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f} {m['Bias']:>8.2f}")

# 计算差异
print("\n--- R² 差异 (Kriging - 观测) ---")
kriging_results = {r[1]: r[3] for r in results if "Kriging" in r[0]}
obs_results = {r[1]: r[3] for r in results if "真实" in r[0]}
for scheme in kriging_results:
    if scheme in obs_results:
        diff = kriging_results[scheme]["R2"] - obs_results[scheme]["R2"]
        print(f"  {scheme:<18} Kriging R²={kriging_results[scheme]['R2']:.4f}  "
              f"观测 R²={obs_results[scheme]['R2']:.4f}  差={diff:+.4f}")

print("\n补充实验 A 完成!")
