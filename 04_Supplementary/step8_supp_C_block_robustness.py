"""
补充实验 C：分组尺度稳健性检验（多种子）
目的：检验分组敏感性结论的稳健性，解释 1.5° 非单调现象
"""
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
print(f"Dataset D: {len(df):,} records\n")

def spatial_block_split(data, block_size, seed, min_per_block=3):
    d = data.copy()
    d["_bid"] = ((d["grid_lat"] // block_size) * block_size).astype(str) + "_" + \
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

SEEDS = [42, 123, 456, 789, 2024]
BLOCK_SIZES = [1.0, 1.5, 2.0, 4.0]

print("=" * 80)
print("补充实验 C：分组尺度稳健性检验（5 种子 × 4 尺度）")
print("=" * 80)

# 逐个跑
all_results = []
for bs in BLOCK_SIZES:
    for seed in SEEDS:
        tr, te = spatial_block_split(df, bs, seed)
        if len(te) < 50:
            continue
        et = ExtraTreesRegressor(n_estimators=100, max_depth=20,
                                  random_state=42, n_jobs=-1)
        et.fit(tr[FEATURE_COLS].values, tr[TARGET].values)
        pred = et.predict(te[FEATURE_COLS].values)
        r2 = r2_score(te[TARGET].values, pred)
        rmse = float(np.sqrt(mean_squared_error(te[TARGET].values, pred)))
        mae = float(mean_absolute_error(te[TARGET].values, pred))
        all_results.append({
            "block_size": bs, "seed": seed,
            "n_train": len(tr), "n_test": len(te),
            "R2": r2, "RMSE": rmse, "MAE": mae
        })
        print(f"  {bs}°×{bs}° seed={seed:>4}: n_test={len(te):>6,}  "
              f"R²={r2:.4f}  RMSE={rmse:.2f}")

# 汇总统计
print("\n" + "=" * 80)
print("汇总：各尺度的均值 ± 标准差（5 种子）")
print("=" * 80)
print(f"{'尺度':<12} {'R² mean':>10} {'R² std':>10} {'RMSE mean':>12} {'RMSE std':>10} {'MAE mean':>10}")
print("-" * 68)

res_df = pd.DataFrame(all_results)
for bs in BLOCK_SIZES:
    sub = res_df[res_df["block_size"] == bs]
    if len(sub) == 0:
        continue
    print(f"{bs}°×{bs}°{'':<6} {sub['R2'].mean():>10.4f} {sub['R2'].std():>10.4f} "
          f"{sub['RMSE'].mean():>12.2f} {sub['RMSE'].std():>10.2f} {sub['MAE'].mean():>10.2f}")

# 详细表格
print("\n--- 完整结果 ---")
print(f"{'尺度':<8} {'seed':>6} {'n_train':>8} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 58)
for r in all_results:
    print(f"{r['block_size']}°{'':<4} {r['seed']:>6} {r['n_train']:>8,} {r['n_test']:>8,} "
          f"{r['R2']:>8.4f} {r['RMSE']:>8.2f} {r['MAE']:>8.2f}")

print("\n补充实验 C 完成!")
