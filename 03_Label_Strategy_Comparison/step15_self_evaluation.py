"""
Step15: 三种标签策略自评估对比实验
每种方法在各自的数据集上独立划分训练/测试集，并在各自的ground truth上评估：
  方法A: 全局 Kriging 伪标签 → 123k网格划分 → 在Kriging值上评估
  方法B: 真实观测直接预测 → 28k点划分 → 在真实q上评估
  方法C: 局部 Kriging 伪标签 → 有效网格划分 → 在Kriging值上评估

对比 step14（共享测试集）vs step15（各自评估），量化标签质量偏差的影响

默认输出（空间分组）：
  CSV:  outputs/step15_self_eval_summary.csv
  图1:  step15_scatter_3methods.png
  图2:  step15_residual_hist_3methods.png
  图3:  step15_vs_step14_comparison.png

随机划分版本：
  python experiments/step15_self_evaluation.py --split random
  CSV:  outputs/step15_random_self_eval_summary.csv
  图1:  step15_random_scatter_3methods.png
  图2:  step15_random_residual_hist_3methods.png

依赖: pip install pykrige
运行时间: 约40-80分钟（局部Kriging是瓶颈）
"""

import argparse
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from pykrige.ok import OrdinaryKriging

ROOT       = Path(__file__).resolve().parents[1]
DATA_PATH  = ROOT / "data/processed/dataset_D_no_aggregation.csv"
GLOBE_PATH = ROOT / "data/features/Ocean_HeatFlow_Prediction_Data_with_Age.csv"
OUT_DIR    = ROOT / "outputs"
FIG_DIR    = ROOT / "outputs/figures"
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
METHOD_NAMES = ["A: Global Kriging", "B: Direct Obs", "C: Local Kriging"]
METHOD_COLORS = ["#e06c3a", "#3a7ebf", "#4caf7d"]

parser = argparse.ArgumentParser(description="Step15 自评估实验")
parser.add_argument("--split", choices=["spatial", "random"], default="spatial",
                    help="训练/测试划分方式，默认 spatial")
args = parser.parse_args()

SPLIT_MODE = args.split
SPLIT_LABEL = "空间分组 2°×2°" if SPLIT_MODE == "spatial" else "随机划分 7:3"
OUT_PREFIX = "step15" if SPLIT_MODE == "spatial" else "step15_random"

# ════════════════════════════════════════════════════════════════════
# 工具函数（复用自 step14）
# ════════════════════════════════════════════════════════════════════
def spatial_block_split(data, lat_col="grid_lat", lon_col="grid_lon",
                       block_size=2.0, test_ratio=0.3, seed=42, min_per_block=3):
    """空间分组划分，支持不同的列名"""
    d = data.copy()
    d["block_id"] = ((d[lat_col] // block_size) * block_size).astype(str) + "_" + \
                    ((d[lon_col] // block_size) * block_size).astype(str)
    bc = d["block_id"].value_counts()
    d = d[d["block_id"].isin(bc[bc >= min_per_block].index)]
    rng = np.random.default_rng(seed)
    blocks = d["block_id"].unique()
    rng.shuffle(blocks)
    n_test = int(len(blocks) * test_ratio)
    test_blocks = set(blocks[:n_test])
    tr = d[~d["block_id"].isin(test_blocks)]
    te = d[d["block_id"].isin(test_blocks)]
    return tr, te


def split_dataset(data, lat_col="grid_lat", lon_col="grid_lon"):
    if SPLIT_MODE == "random":
        tr, te = train_test_split(data, test_size=0.3, random_state=42)
        return tr.copy(), te.copy()
    return spatial_block_split(data, lat_col=lat_col, lon_col=lon_col)


def calc_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Bias": bias}


def calc_moran_knn(coords, values, k=8):
    n = len(values)
    z = values - np.mean(values)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k+1)
    num = sum(z[i] * z[indices[i, j]] for i in range(n) for j in range(1, k+1))
    W = n * k
    return float((n / W) * (num / np.sum(z**2)))


def assign_basin(lat, lon):
    """根据 lat/lon 赋值 basin"""
    if -180 <= lon < -60 and -60 <= lat <= 60:
        return "Atlantic"
    elif -60 <= lon < 120 and -60 <= lat <= 60:
        return "Indian"
    elif (120 <= lon <= 180 or -180 <= lon < -60) and -60 <= lat <= 60:
        return "Pacific"
    else:
        return "Southern"


def base_map(ax, title, fontsize=11):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#d8d3cc", zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf4fb", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#444444", zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)


def chord_to_km(chord_dist):
    """弦距转地表距离（km）"""
    R_earth = 6371.0
    return 2 * R_earth * np.arcsin(np.clip(chord_dist / (2 * R_earth), 0, 1))


# ════════════════════════════════════════════════════════════════════
# 阶段0：数据准备
# ════════════════════════════════════════════════════════════════════
print("=" * 70)
print("阶段0：数据准备")
print("=" * 70)
print(f"  划分方式: {SPLIT_LABEL}")

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS + [TARGET])
df = df[df[TARGET] > 0].copy()
print(f"  观测数据: {len(df):,} 条")

# Kriging 支撑点按网格中心聚合，避免重复坐标导致的巨大开销和数值不稳定
kriging_grid = df.groupby(["grid_lat", "grid_lon"], as_index=False)[TARGET].median()
print(f"  Kriging 支撑网格: {len(kriging_grid):,} 个")

globe = pd.read_csv(GLOBE_PATH, usecols=["lon", "lat"] + FEATURE_COLS)
globe = globe.dropna(subset=FEATURE_COLS)
print(f"  全球网格: {len(globe):,} 个")

# 方法B 用的训练/测试划分
train_df_B, test_df_B = split_dataset(df, lat_col="grid_lat", lon_col="grid_lon")
print(f"  方法B 训练集: {len(train_df_B):,}  测试集: {len(test_df_B):,}")

# 存储三种方法的结果
results_all = {}


# ════════════════════════════════════════════════════════════════════
# 阶段1：方法A — 全局 Kriging 自评估
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段1：方法A — 全局 Kriging 自评估")
print("=" * 70)

print("  用全部观测点做全局 Kriging...")
t0 = time.time()
ok_global = OrdinaryKriging(
    kriging_grid["grid_lon"].values,
    kriging_grid["grid_lat"].values,
    kriging_grid[TARGET].values,
    variogram_model="spherical",
    verbose=False,
    enable_plotting=False,
    nlags=20,
)

# 批量执行，比逐点循环快得多
z_global, _ = ok_global.execute(
    "points",
    globe["lon"].values,
    globe["lat"].values,
    n_closest_points=50,
    backend="loop",
)
z_global = np.asarray(z_global)
print(f"  完成（耗时 {time.time()-t0:.1f}s）")

# 构建 globe_A
globe_A = globe.copy()
globe_A["q_label"] = z_global
globe_A = globe_A.dropna(subset=["q_label"])
print(f"  有效网格: {len(globe_A):,} 个")

# 对 globe_A 做划分
train_A, test_A = split_dataset(globe_A, lat_col="lat", lon_col="lon")
print(f"  训练集: {len(train_A):,}  测试集: {len(test_A):,}")

# 训练模型
X_train_A = train_A[FEATURE_COLS].values
y_train_A = train_A["q_label"].values
X_test_A = test_A[FEATURE_COLS].values
y_test_A = test_A["q_label"].values

model_A = ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model_A.fit(X_train_A, y_train_A)
pred_A = model_A.predict(X_test_A)

# 评估
metrics_A = calc_metrics(y_test_A, pred_A)
coords_A = test_A[["lat", "lon"]].values
moran_A = calc_moran_knn(coords_A, pred_A - y_test_A, k=8)

results_all["A"] = {
    "metrics": metrics_A,
    "moran": moran_A,
    "y_test": y_test_A,
    "pred": pred_A,
    "coords": coords_A,
    "n_test": len(test_A),
    "ground_truth": "Kriging"
}

print(f"  R²={metrics_A['R2']:.4f}  RMSE={metrics_A['RMSE']:.2f}  MAE={metrics_A['MAE']:.2f}  Bias={metrics_A['Bias']:.2f}")
print(f"  Moran's I={moran_A:.4f}")


# ════════════════════════════════════════════════════════════════════
# 阶段2：方法B — 真实观测自评估
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段2：方法B — 真实观测自评估")
print("=" * 70)

X_train_B = train_df_B[FEATURE_COLS].values
y_train_B = train_df_B[TARGET].values
X_test_B = test_df_B[FEATURE_COLS].values
y_test_B = test_df_B[TARGET].values

model_B = ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model_B.fit(X_train_B, y_train_B)
pred_B = model_B.predict(X_test_B)

# 评估
metrics_B = calc_metrics(y_test_B, pred_B)
coords_B = test_df_B[["grid_lat", "grid_lon"]].values
moran_B = calc_moran_knn(coords_B, pred_B - y_test_B, k=8)

results_all["B"] = {
    "metrics": metrics_B,
    "moran": moran_B,
    "y_test": y_test_B,
    "pred": pred_B,
    "coords": coords_B,
    "n_test": len(test_df_B),
    "ground_truth": "Real Obs"
}

print(f"  R²={metrics_B['R2']:.4f}  RMSE={metrics_B['RMSE']:.2f}  MAE={metrics_B['MAE']:.2f}  Bias={metrics_B['Bias']:.2f}")
print(f"  Moran's I={moran_B:.4f}")


# ════════════════════════════════════════════════════════════════════
# 阶段3：方法C — 局部 Kriging 自评估
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段3：方法C — 局部 Kriging 自评估")
print("=" * 70)

print("  用全部观测点做局部 Kriging...")
t0 = time.time()

# 参数
WINDOW_SIZE = 6.0  # 6°×6°
N_MIN = 5
D_MAX_KM = 500
HALF_W = WINDOW_SIZE / 2.0

# 预计算距离
obs_lon = kriging_grid["grid_lon"].values
obs_lat = kriging_grid["grid_lat"].values
obs_q = kriging_grid[TARGET].values

train_xyz = np.column_stack([
    np.cos(np.radians(obs_lat)) * np.cos(np.radians(obs_lon)),
    np.cos(np.radians(obs_lat)) * np.sin(np.radians(obs_lon)),
    np.sin(np.radians(obs_lat))
])
globe_lon = globe["lon"].values
globe_lat = globe["lat"].values
grid_xyz = np.column_stack([
    np.cos(np.radians(globe_lat)) * np.cos(np.radians(globe_lon)),
    np.cos(np.radians(globe_lat)) * np.sin(np.radians(globe_lon)),
    np.sin(np.radians(globe_lat))
])
tree = cKDTree(train_xyz)
dist_km_all = chord_to_km(tree.query(grid_xyz, k=1)[0])

# 局部 Kriging
z_local = np.full(len(globe), np.nan)
valid_count = 0

for i in range(len(globe)):
    if i % 20000 == 0:
        print(f"    进度: {i}/{len(globe):,}")

    if dist_km_all[i] > D_MAX_KM:
        continue

    lon_i = globe_lon[i]
    lat_i = globe_lat[i]

    # 真实 6°×6° 局部窗口：中心点两侧各 3°
    mask_window = (
        (obs_lon >= lon_i - HALF_W) & (obs_lon <= lon_i + HALF_W) &
        (obs_lat >= lat_i - HALF_W) & (obs_lat <= lat_i + HALF_W)
    )

    n_local = int(mask_window.sum())
    if n_local >= N_MIN:
        try:
            ok_local = OrdinaryKriging(
                obs_lon[mask_window],
                obs_lat[mask_window],
                obs_q[mask_window],
                variogram_model="spherical",
                verbose=False,
                enable_plotting=False,
            )
            z_pred, _ = ok_local.execute("points", [lon_i], [lat_i])
            val = float(z_pred[0])
            if np.isfinite(val) and 0 < val < 500:
                z_local[i] = val
                valid_count += 1
        except Exception:
            pass

print(f"  完成（耗时 {time.time()-t0:.1f}s，有效网格 {valid_count:,}）")

# 构建 globe_C
globe_C = globe.copy()
globe_C["q_label"] = z_local
globe_C = globe_C.dropna(subset=["q_label"])
print(f"  有效网格: {len(globe_C):,} 个")

# 对 globe_C 做划分
train_C, test_C = split_dataset(globe_C, lat_col="lat", lon_col="lon")
print(f"  训练集: {len(train_C):,}  测试集: {len(test_C):,}")

# 训练模型
X_train_C = train_C[FEATURE_COLS].values
y_train_C = train_C["q_label"].values
X_test_C = test_C[FEATURE_COLS].values
y_test_C = test_C["q_label"].values

model_C = ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model_C.fit(X_train_C, y_train_C)
pred_C = model_C.predict(X_test_C)

# 评估
metrics_C = calc_metrics(y_test_C, pred_C)
coords_C = test_C[["lat", "lon"]].values
moran_C = calc_moran_knn(coords_C, pred_C - y_test_C, k=8)

results_all["C"] = {
    "metrics": metrics_C,
    "moran": moran_C,
    "y_test": y_test_C,
    "pred": pred_C,
    "coords": coords_C,
    "n_test": len(test_C),
    "ground_truth": "Kriging"
}

print(f"  R²={metrics_C['R2']:.4f}  RMSE={metrics_C['RMSE']:.2f}  MAE={metrics_C['MAE']:.2f}  Bias={metrics_C['Bias']:.2f}")
print(f"  Moran's I={moran_C:.4f}")


# ════════════════════════════════════════════════════════════════════
# 阶段4：汇总评估
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段4：汇总评估")
print("=" * 70)

print(f"\n{'方法':<25} {'Ground Truth':<15} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'Moran I':>8}")
print("-" * 85)
for key, name in zip(["A", "B", "C"], METHOD_NAMES):
    r = results_all[key]
    m = r["metrics"]
    print(f"  {name:<23} {r['ground_truth']:<15} {m['R2']:>8.4f} {m['RMSE']:>8.2f} "
          f"{m['MAE']:>8.2f} {m['Bias']:>8.2f} {r['moran']:>8.4f}")

# 构建汇总 CSV
summary_data = []
for key, name in zip(["A", "B", "C"], METHOD_NAMES):
    r = results_all[key]
    m = r["metrics"]
    summary_data.append({
        "Method": name,
        "Ground_Truth": r["ground_truth"],
        "N_Test": r["n_test"],
        "R2": m["R2"],
        "RMSE": m["RMSE"],
        "MAE": m["MAE"],
        "Bias": m["Bias"],
        "Moran_I": r["moran"]
    })

summary_df = pd.DataFrame(summary_data)
summary_name = f"{OUT_PREFIX}_self_eval_summary.csv"
summary_df.to_csv(OUT_DIR / summary_name, index=False)
print(f"\n已保存: outputs/{summary_name}")


# ════════════════════════════════════════════════════════════════════
# 阶段5：可视化
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段5：可视化")
print("=" * 70)

# 图1：三种方法的散点图
print("绘制图1：三种方法散点图...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (key, name, color) in enumerate(zip(["A", "B", "C"], METHOD_NAMES, METHOD_COLORS)):
    ax = axes[idx]
    r = results_all[key]
    y_t = r["y_test"]
    pred = r["pred"]
    m = r["metrics"]

    ax.scatter(y_t, pred, c=color, s=4, alpha=0.5, linewidths=0)

    lim = (np.min(y_t) - 10, np.max(y_t) + 10)
    ax.plot(lim, lim, "k--", linewidth=1.2, label="1:1 line")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(f"{r['ground_truth']} (mW/m²)", fontsize=11)
    ax.set_ylabel("Predicted (mW/m²)", fontsize=11)
    ax.set_title(f"{name}\nR²={m['R2']:.4f}  RMSE={m['RMSE']:.1f}  n={r['n_test']:,}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle(f"Step15: Self-Evaluation ({SPLIT_LABEL})",
             fontsize=13, fontweight="bold")
plt.tight_layout()
scatter_name = f"{OUT_PREFIX}_scatter_3methods.png"
fig.savefig(FIG_DIR / scatter_name, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: {scatter_name}")

# 图2：残差直方图
print("绘制图2：残差直方图...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, (key, name, color) in enumerate(zip(["A", "B", "C"], METHOD_NAMES, METHOD_COLORS)):
    ax = axes[idx]
    r = results_all[key]
    residuals = r["pred"] - r["y_test"]
    m = r["metrics"]

    ax.hist(residuals, bins=60, color=color, alpha=0.75, edgecolor="none")
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.axvline(m["Bias"], color="#cc3333", linewidth=1.5, linestyle="-",
               label=f"Bias={m['Bias']:.1f}")

    ax.set_xlabel("Residual (mW/m²)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{name}\nRMSE={m['RMSE']:.1f}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle(f"Step15: Residual Distribution ({SPLIT_LABEL})", fontsize=13, fontweight="bold")
plt.tight_layout()
resid_name = f"{OUT_PREFIX}_residual_hist_3methods.png"
fig.savefig(FIG_DIR / resid_name, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: {resid_name}")

compare_name = None
if SPLIT_MODE == "spatial":
    # 图3：与 step14 对比
    print("绘制图3：与 step14 对比...")

    # 加载 step14 结果
    step14_df = pd.read_csv(OUT_DIR / "step14_comparison_summary.csv")
    step14_results = {}
    for _, row in step14_df.iterrows():
        method_key = row["Method"].split(":")[0].strip()
        step14_results[method_key] = {
            "R2": row["R2"],
            "RMSE": row["RMSE"],
            "MAE": row["MAE"]
        }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：R² 对比
    ax = axes[0]
    methods = ["A", "B", "C"]
    x = np.arange(len(methods))
    width = 0.35

    r2_step15 = [results_all[m]["metrics"]["R2"] for m in methods]
    r2_step14 = [step14_results[m]["R2"] for m in methods]

    ax.bar(x - width/2, r2_step15, width, label="Step15 (Self-Eval)", color="#3a7ebf", alpha=0.8)
    ax.bar(x + width/2, r2_step14, width, label="Step14 (Cross-Eval)", color="#e06c3a", alpha=0.8)

    ax.set_ylabel("R²", fontsize=11)
    ax.set_title("R² Comparison: Self-Evaluation vs Cross-Evaluation", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_NAMES)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.8)

    # 子图2：RMSE 对比
    ax = axes[1]
    rmse_step15 = [results_all[m]["metrics"]["RMSE"] for m in methods]
    rmse_step14 = [step14_results[m]["RMSE"] for m in methods]

    ax.bar(x - width/2, rmse_step15, width, label="Step15 (Self-Eval)", color="#3a7ebf", alpha=0.8)
    ax.bar(x + width/2, rmse_step14, width, label="Step14 (Cross-Eval)", color="#e06c3a", alpha=0.8)

    ax.set_ylabel("RMSE (mW/m²)", fontsize=11)
    ax.set_title("RMSE Comparison: Self-Evaluation vs Cross-Evaluation", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_NAMES)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Step15 vs Step14: Impact of Label Quality", fontsize=13, fontweight="bold")
    plt.tight_layout()
    compare_name = f"{OUT_PREFIX}_vs_step14_comparison.png"
    fig.savefig(FIG_DIR / compare_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存: {compare_name}")
else:
    print("跳过图3：随机划分版本不与 step14 空间分组结果直接对比")


# ════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("全部完成！输出文件：")
print(f"  CSV   outputs/{summary_name}")
print(f"  图1   {scatter_name}        — 三种方法散点图")
print(f"  图2   {resid_name}  — 残差直方图")
if compare_name is not None:
    print(f"  图3   {compare_name}    — 与 step14 对比")
print("=" * 70)
