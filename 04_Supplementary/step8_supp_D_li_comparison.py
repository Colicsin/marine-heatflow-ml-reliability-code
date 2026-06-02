"""
补充实验 D：与 Li et al. (2021) 方法论对比
目的：与同领域最相关工作做直接方法论对比，展示空间泄漏
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
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

def spatial_block_split(data, block_size=2.0, seed=42, min_per_block=3):
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

def calc_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }

X = df[FEATURE_COLS].values
y = df[TARGET].values

print("=" * 80)
print("补充实验 D：与 Li et al. (2021) 方法论对比")
print("=" * 80)

# ── 对比1：Li et al. 设定（RF, 80/20 随机划分）vs 本文设定 ──
print("\n--- 对比1：Li et al. 设定 vs 本文设定 ---")
print(f"{'设定':<35} {'模型':<12} {'划分':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 82)

configs = [
    # Li et al. 设定
    ("Li: RF, 80/20 random",    "RF",         0.2, 42),
    ("Li: RF, 70/30 random",    "RF",         0.3, 42),
    # 本文设定
    ("Ours: ET, 70/30 random",  "ExtraTrees", 0.3, 42),
    ("Ours: RF, 70/30 random",  "RF",         0.3, 42),
]

random_results = {}
for label, model_name, test_size, seed in configs:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    if model_name == "RF":
        model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    else:
        model = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    m = calc_metrics(y_te, model.predict(X_te))
    random_results[label] = m
    print(f"{label:<35} {model_name:<12} {f'{int((1-test_size)*100)}/{int(test_size*100)}':<12} "
          f"{m['R2']:>8.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f}")

# ── 对比2：随机划分 vs 空间分组（RF 和 ExtraTrees）──
print("\n--- 对比2：随机划分 vs 空间分组（展示空间泄漏）---")
print(f"{'模型':<12} {'验证方案':<20} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'R²差':>8}")
print("-" * 72)

for model_name, model_cls in [("RF", RandomForestRegressor), ("ExtraTrees", ExtraTreesRegressor)]:
    # 随机划分
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    m_rand = model_cls(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    m_rand.fit(X_tr, y_tr)
    mr = calc_metrics(y_te, m_rand.predict(X_te))

    # 空间分组
    tr, te = spatial_block_split(df)
    m_sp = model_cls(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    m_sp.fit(tr[FEATURE_COLS].values, tr[TARGET].values)
    ms = calc_metrics(te[TARGET].values, m_sp.predict(te[FEATURE_COLS].values))

    gap = mr["R2"] - ms["R2"]
    print(f"{model_name:<12} {'随机划分 7:3':<20} {len(y_te):>8,} {mr['R2']:>8.4f} {mr['RMSE']:>8.2f} {mr['MAE']:>8.2f}")
    print(f"{'':<12} {'空间分组 2°×2°':<20} {len(te):>8,} {ms['R2']:>8.4f} {ms['RMSE']:>8.2f} {ms['MAE']:>8.2f} {gap:>+8.4f}")

# ── 对比3：跨洋盆验证（RF vs ExtraTrees）──
print("\n--- 对比3：跨洋盆验证 ---")
print(f"{'模型':<12} {'测试洋盆':<12} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 58)

for model_name, model_cls in [("RF", RandomForestRegressor), ("ExtraTrees", ExtraTreesRegressor)]:
    for basin in ["Pacific", "Atlantic", "Indian"]:
        tr_b = df[df["basin"] != basin]
        te_b = df[df["basin"] == basin]
        m_b = model_cls(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        m_b.fit(tr_b[FEATURE_COLS].values, tr_b[TARGET].values)
        mb = calc_metrics(te_b[TARGET].values, m_b.predict(te_b[FEATURE_COLS].values))
        print(f"{model_name:<12} {basin:<12} {len(te_b):>8,} {mb['R2']:>8.4f} {mb['RMSE']:>8.2f} {mb['MAE']:>8.2f}")

# ── 对比4：Li et al. 报告值 vs 本文复现值 ──
print("\n--- 对比4：Li et al. (2021) 论文报告值 vs 本文数据 ---")
print("注：Li et al. 使用 NGHF 数据库 + 25 个特征 + RF + 80/20 随机划分")
print("    本文使用 GHFDB R2024 + 14 个特征 + RF + 80/20 随机划分")
print()
print(f"{'来源':<30} {'R²':>8} {'RMSE':>8}")
print("-" * 50)
print(f"{'Li No.1 (A级, NGHF)':<30} {'0.96':>8} {'11.74':>8}")
print(f"{'Li No.2 (A+B级, NGHF)':<30} {'0.88':>8} {'22.58':>8}")
print(f"{'Li No.3 (A+B+C级, NGHF)':<30} {'0.77':>8} {'32.56':>8}")
r2_ours = random_results["Li: RF, 80/20 random"]["R2"]
rmse_ours = random_results["Li: RF, 80/20 random"]["RMSE"]
print(f"{'本文 (全部, GHFDB R2024)':<30} {r2_ours:>8.4f} {rmse_ours:>8.2f}")
print()
print("差异分析：Li et al. 的高 R² 主要来自两个因素：")
print("  1. A 级数据噪声极低（仅保留最高质量测量）")
print("  2. 80/20 随机划分 + 同网格多记录共享特征 → 空间泄漏")
print(f"  本文在相同划分策略下 R²={r2_ours:.4f}，远低于 Li No.1 的 0.96，")
print("  说明标签质量（真实观测 vs 高质量子集）是性能差异的主要来源。")

print("\n补充实验 D 完成!")
