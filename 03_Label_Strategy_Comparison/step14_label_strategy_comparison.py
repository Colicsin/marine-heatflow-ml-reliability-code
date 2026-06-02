"""
Step14: 三种标签策略对比实验
对比训练标签来源对ML预测精度的影响：
  方法A: 全局 Ordinary Kriging 伪标签 → ExtraTrees
  方法B: 真实观测直接预测（当前方法，基线）
  方法C: 局部滑动窗口 Kriging 伪标签 → ExtraTrees

三种方法共享：14个特征、ExtraTrees模型、空间分组2°×2°验证、真实观测q作为ground truth

输出：
  CSV:  outputs/step14_comparison_summary.csv
  图1:  step14_scatter_3methods.png
  图2:  step14_residual_map_3methods.png
  图3:  step14_basin_comparison.png
  图4:  step14_residual_hist_3methods.png
  图5:  step14_high_hf_comparison.png

依赖: pip install pykrige
运行时间: 约40-80分钟（局部Kriging是瓶颈）
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial import cKDTree

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
METHOD_NAMES = ["A: Global Kriging", "B: Direct Obs (Baseline)", "C: Local Kriging"]
METHOD_COLORS = ["#e06c3a", "#3a7ebf", "#4caf7d"]


# ════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════
def spatial_block_split(data, block_size=2.0, test_ratio=0.3, seed=42, min_per_block=3):
    d = data.copy()
    d["block_id"] = ((d["grid_lat"] // block_size) * block_size).astype(str) + "_" + \
                    ((d["grid_lon"] // block_size) * block_size).astype(str)
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


def base_map(ax, title, fontsize=11):
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#d8d3cc", zorder=2)
    ax.add_feature(cfeature.OCEAN, facecolor="#eaf4fb", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#444444", zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)


# ════════════════════════════════════════════════════════════════════
# 阶段0：数据准备
# ════════════════════════════════════════════════════════════════════
print("=" * 70)
print("阶段0：数据准备")
print("=" * 70)

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS + [TARGET])
df = df[df[TARGET] > 0].copy()
print(f"  观测数据: {len(df):,} 条")

globe = pd.read_csv(GLOBE_PATH, usecols=["lon", "lat"] + FEATURE_COLS)
globe = globe.dropna(subset=FEATURE_COLS)
print(f"  全球网格: {len(globe):,} 个")

# 统一空间分组划分（三种方法共享同一测试集）
train_df, test_df = spatial_block_split(df)
print(f"  训练集: {len(train_df):,}  测试集: {len(test_df):,}")

X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET].values
basins  = test_df["basin"].values if "basin" in test_df.columns else np.array(["Unknown"] * len(test_df))

# 存储三种方法的预测结果
all_preds = {}


# ════════════════════════════════════════════════════════════════════
# 阶段1：方法A — 全局 Kriging 伪标签
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段1：方法A — 全局 Ordinary Kriging 伪标签")
print("=" * 70)

from pykrige.ok import OrdinaryKriging

# 用训练集观测点做 Kriging（去重到网格级别，取中位数，减少点数加速）
train_grid = train_df.groupby(["grid_lat", "grid_lon"])[TARGET].median().reset_index()
print(f"  Kriging 输入点数: {len(train_grid):,}（训练集网格中位数）")

t0 = time.time()
ok_global = OrdinaryKriging(
    train_grid["grid_lon"].values,
    train_grid["grid_lat"].values,
    train_grid[TARGET].values,
    variogram_model="spherical",
    verbose=False,
    enable_plotting=False,
    nlags=20,
)

print("  对全球网格执行 Kriging 插值（n_closest_points=50）...")
z_global, ss_global = ok_global.execute(
    "points",
    globe["lon"].values,
    globe["lat"].values,
    n_closest_points=50,
    backend="loop",
)
z_global = np.asarray(z_global)
# 过滤不合理值
bad = ~np.isfinite(z_global) | (z_global < 0) | (z_global > 500)
z_global[bad] = np.nan
n_valid_A = np.isfinite(z_global).sum()
print(f"  Kriging 完成: {time.time()-t0:.1f}s  有效网格: {n_valid_A:,}/{len(globe):,}")

# 用 Kriging 伪标签 + 特征训练 ExtraTrees
globe_A = globe.copy()
globe_A["q_label"] = z_global
globe_A = globe_A.dropna(subset=["q_label"])
print(f"  方法A训练集: {len(globe_A):,} 个网格（Kriging伪标签）")

model_A = ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model_A.fit(globe_A[FEATURE_COLS].values, globe_A["q_label"].values)
pred_A = model_A.predict(X_test)
all_preds["A"] = pred_A

m_A = calc_metrics(y_test, pred_A)
print(f"  测试集: R²={m_A['R2']:.4f}  RMSE={m_A['RMSE']:.2f}  MAE={m_A['MAE']:.2f}  Bias={m_A['Bias']:.2f}")


# ════════════════════════════════════════════════════════════════════
# 阶段2：方法B — 真实观测直接预测（基线）
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段2：方法B — 真实观测直接预测（基线）")
print("=" * 70)

X_train_B = train_df[FEATURE_COLS].values
y_train_B = train_df[TARGET].values
print(f"  方法B训练集: {len(train_df):,} 条真实观测")

model_B = ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model_B.fit(X_train_B, y_train_B)
pred_B = model_B.predict(X_test)
all_preds["B"] = pred_B

m_B = calc_metrics(y_test, pred_B)
print(f"  测试集: R²={m_B['R2']:.4f}  RMSE={m_B['RMSE']:.2f}  MAE={m_B['MAE']:.2f}  Bias={m_B['Bias']:.2f}")


# ════════════════════════════════════════════════════════════════════
# 阶段3：方法C — 局部滑动窗口 Kriging 伪标签
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段3：方法C — 局部滑动窗口 Kriging 伪标签")
print("=" * 70)

WINDOW_DEG = 6.0
N_MIN      = 5
D_MAX_KM   = 500.0
HALF_W     = WINDOW_DEG / 2.0

# 训练集观测点坐标（用原始点，不聚合）
obs_lon = train_df["long_EW"].values
obs_lat = train_df["lat_NS"].values
obs_q   = train_df[TARGET].values

# 预计算全球网格到最近观测点的距离（用于快速跳过远距离网格）
def _to_xyz(lon, lat):
    lon_r, lat_r = np.radians(lon), np.radians(lat)
    return np.column_stack([np.cos(lat_r)*np.cos(lon_r),
                            np.cos(lat_r)*np.sin(lon_r),
                            np.sin(lat_r)])

obs_tree = cKDTree(_to_xyz(obs_lon, obs_lat))
grid_xyz = _to_xyz(globe["lon"].values, globe["lat"].values)
dist_chord, _ = obs_tree.query(grid_xyz, k=1)
dist_km_all = dist_chord * 6371.0  # 弦距离近似转km

n_skip = (dist_km_all > D_MAX_KM).sum()
print(f"  窗口: {WINDOW_DEG}°×{WINDOW_DEG}°  n_min={N_MIN}  d_max={D_MAX_KM}km")
print(f"  跳过远距离网格: {n_skip:,}/{len(globe):,} ({100*n_skip/len(globe):.1f}%)")

# 逐点局部 Kriging
preds_C = np.full(len(globe), np.nan)
t0 = time.time()
n_done = 0

for i in range(len(globe)):
    if dist_km_all[i] > D_MAX_KM:
        continue

    glon = globe["lon"].values[i]
    glat = globe["lat"].values[i]

    # 矩形窗口快速过滤
    lon_mask = (obs_lon >= glon - HALF_W) & (obs_lon <= glon + HALF_W)
    lat_mask = (obs_lat >= glat - HALF_W) & (obs_lat <= glat + HALF_W)
    local_mask = lon_mask & lat_mask
    n_local = local_mask.sum()

    if n_local < N_MIN:
        continue

    try:
        ok_local = OrdinaryKriging(
            obs_lon[local_mask], obs_lat[local_mask], obs_q[local_mask],
            variogram_model="spherical",
            verbose=False, enable_plotting=False,
        )
        z_loc, _ = ok_local.execute("points", np.array([glon]), np.array([glat]))
        val = float(z_loc[0])
        if np.isfinite(val) and 0 < val < 500:
            preds_C[i] = val
    except Exception:
        pass

    n_done += 1
    if n_done % 2000 == 0:
        elapsed = time.time() - t0
        n_filled = np.isfinite(preds_C).sum()
        print(f"    进度: {n_done:,} 个网格已处理  filled={n_filled:,}  "
              f"耗时={elapsed:.0f}s")

n_valid_C = np.isfinite(preds_C).sum()
print(f"  局部Kriging完成: {time.time()-t0:.1f}s  有效网格: {n_valid_C:,}/{len(globe):,} "
      f"({100*n_valid_C/len(globe):.1f}%)")

# 用局部 Kriging 伪标签 + 特征训练 ExtraTrees
globe_C = globe.copy()
globe_C["q_label"] = preds_C
globe_C = globe_C.dropna(subset=["q_label"])
print(f"  方法C训练集: {len(globe_C):,} 个网格（局部Kriging伪标签）")

model_C = ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model_C.fit(globe_C[FEATURE_COLS].values, globe_C["q_label"].values)
pred_C = model_C.predict(X_test)
all_preds["C"] = pred_C

m_C = calc_metrics(y_test, pred_C)
print(f"  测试集: R²={m_C['R2']:.4f}  RMSE={m_C['RMSE']:.2f}  MAE={m_C['MAE']:.2f}  Bias={m_C['Bias']:.2f}")


# ════════════════════════════════════════════════════════════════════
# 阶段4：统一评估与对比
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段4：统一评估与对比")
print("=" * 70)

all_metrics = {"A": m_A, "B": m_B, "C": m_C}
basin_list = ["Pacific", "Atlantic", "Indian"]

# 分洋盆指标
basin_metrics = {}
for key in ["A", "B", "C"]:
    pred = all_preds[key]
    basin_metrics[key] = {}
    for basin in basin_list:
        mask = basins == basin
        if mask.sum() > 10:
            basin_metrics[key][basin] = calc_metrics(y_test[mask], pred[mask])

# 高热流区指标
high_mask = y_test > 100
high_metrics = {}
for key in ["A", "B", "C"]:
    pred = all_preds[key]
    if high_mask.sum() > 10:
        high_metrics[key] = calc_metrics(y_test[high_mask], pred[high_mask])

# Moran's I
coords_test = test_df[["grid_lat", "grid_lon"]].values
moran_vals = {}
for key in ["A", "B", "C"]:
    residuals = all_preds[key] - y_test
    moran_vals[key] = calc_moran_knn(coords_test, residuals, k=8)

# 打印汇总表
print(f"\n{'方法':<25} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'Moran I':>8}")
print("-" * 70)
for key, name in zip(["A", "B", "C"], METHOD_NAMES):
    m = all_metrics[key]
    mi = moran_vals[key]
    print(f"  {name:<23} {m['R2']:>8.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f} "
          f"{m['Bias']:>8.2f} {mi:>8.4f}")

print(f"\n分洋盆 R² / RMSE:")
print(f"{'方法':<25}", end="")
for basin in basin_list:
    print(f"  {basin+' R²':>12} {basin+' RMSE':>12}", end="")
print()
print("-" * 100)
for key, name in zip(["A", "B", "C"], METHOD_NAMES):
    print(f"  {name:<23}", end="")
    for basin in basin_list:
        bm = basin_metrics[key].get(basin, {})
        print(f"  {bm.get('R2', float('nan')):>12.4f} {bm.get('RMSE', float('nan')):>12.2f}", end="")
    print()

print(f"\n高热流区 (>100 mW/m², n={high_mask.sum():,}):")
print(f"{'方法':<25} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 55)
for key, name in zip(["A", "B", "C"], METHOD_NAMES):
    hm = high_metrics.get(key, {})
    print(f"  {name:<23} {hm.get('R2', float('nan')):>8.4f} "
          f"{hm.get('RMSE', float('nan')):>8.2f} {hm.get('MAE', float('nan')):>8.2f}")

# 保存汇总 CSV
rows = []
for key, name in zip(["A", "B", "C"], METHOD_NAMES):
    m = all_metrics[key]
    row = {"Method": name, **m, "Moran_I": moran_vals[key]}
    for basin in basin_list:
        bm = basin_metrics[key].get(basin, {})
        row[f"{basin}_R2"] = bm.get("R2", np.nan)
        row[f"{basin}_RMSE"] = bm.get("RMSE", np.nan)
    hm = high_metrics.get(key, {})
    row["HighHF_R2"] = hm.get("R2", np.nan)
    row["HighHF_RMSE"] = hm.get("RMSE", np.nan)
    rows.append(row)
summary_df = pd.DataFrame(rows)
summary_df.to_csv(OUT_DIR / "step14_comparison_summary.csv", index=False)
print(f"\n已保存: outputs/step14_comparison_summary.csv")


# ════════════════════════════════════════════════════════════════════
# 阶段5：可视化
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("阶段5：可视化")
print("=" * 70)

# ── 图1：三种方法散点图对比 ──────────────────────────────────────
print("绘制图1：散点图对比...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
basin_colors = {"Pacific": "#3a7ebf", "Atlantic": "#e06c3a", "Indian": "#4caf7d"}

for col, (key, name, mcolor) in enumerate(zip(["A", "B", "C"], METHOD_NAMES, METHOD_COLORS)):
    ax = axes[col]
    pred = all_preds[key]
    m = all_metrics[key]

    for basin, bcolor in basin_colors.items():
        mask = basins == basin
        ax.scatter(y_test[mask], pred[mask], c=bcolor, s=3, alpha=0.4,
                   linewidths=0, label=basin)

    lim = (0, 300)
    ax.plot(lim, lim, "k--", linewidth=1.2, label="1:1 line")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel("Observed (mW/m²)", fontsize=10)
    ax.set_ylabel("Predicted (mW/m²)", fontsize=10)
    ax.set_title(f"{name}\nR²={m['R2']:.4f}  RMSE={m['RMSE']:.1f}  Bias={m['Bias']:.2f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Label Strategy Comparison — Observed vs Predicted (Test Set)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step14_scatter_3methods.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step14_scatter_3methods.png")

# ── 图2：三种方法残差空间分布 ────────────────────────────────────
print("绘制图2：残差空间分布...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6),
                         subplot_kw={"projection": ccrs.Robinson()})
norm_res = mcolors.TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80)

for col, (key, name) in enumerate(zip(["A", "B", "C"], METHOD_NAMES)):
    ax = axes[col]
    residual = all_preds[key] - y_test
    base_map(ax, f"{name}\nBias={all_metrics[key]['Bias']:.2f} mW/m²", fontsize=10)
    sc = ax.scatter(
        test_df["long_EW"].values, test_df["lat_NS"].values,
        c=residual, cmap="RdBu_r", norm=norm_res,
        s=4, alpha=0.8, linewidths=0,
        transform=ccrs.PlateCarree(), zorder=4
    )

cbar = plt.colorbar(sc, ax=axes, orientation="horizontal",
                    pad=0.05, fraction=0.03, aspect=50, extend="both")
cbar.set_label("Residual (mW/m²)  Red=overestimate  Blue=underestimate", fontsize=10)

plt.suptitle("Residual Spatial Distribution by Label Strategy",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "step14_residual_map_3methods.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step14_residual_map_3methods.png")

# ── 图3：分洋盆 R²/RMSE 柱状图 ──────────────────────────────────
print("绘制图3：分洋盆对比...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
x = np.arange(len(basin_list))
width = 0.25

# R²
ax = axes[0]
for i, (key, name, color) in enumerate(zip(["A", "B", "C"], METHOD_NAMES, METHOD_COLORS)):
    vals = [basin_metrics[key].get(b, {}).get("R2", 0) for b in basin_list]
    ax.bar(x + i * width, vals, width, color=color, alpha=0.8, label=name)
ax.set_xticks(x + width)
ax.set_xticklabels(basin_list, fontsize=11)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("R² by Ocean Basin", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# RMSE
ax = axes[1]
for i, (key, name, color) in enumerate(zip(["A", "B", "C"], METHOD_NAMES, METHOD_COLORS)):
    vals = [basin_metrics[key].get(b, {}).get("RMSE", 0) for b in basin_list]
    ax.bar(x + i * width, vals, width, color=color, alpha=0.8, label=name)
ax.set_xticks(x + width)
ax.set_xticklabels(basin_list, fontsize=11)
ax.set_ylabel("RMSE (mW/m²)", fontsize=12)
ax.set_title("RMSE by Ocean Basin", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Basin-level Performance Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step14_basin_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step14_basin_comparison.png")

# ── 图4：残差直方图叠加对比 ──────────────────────────────────────
print("绘制图4：残差直方图...")

fig, ax = plt.subplots(figsize=(10, 6))
bins_hist = np.linspace(-150, 150, 80)

for key, name, color in zip(["A", "B", "C"], METHOD_NAMES, METHOD_COLORS):
    residual = all_preds[key] - y_test
    ax.hist(residual, bins=bins_hist, color=color, alpha=0.45, edgecolor="none",
            label=f"{name}  Bias={all_metrics[key]['Bias']:.1f}")

ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
ax.set_xlabel("Residual (mW/m²)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("Residual Distribution Comparison (Test Set)", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(FIG_DIR / "step14_residual_hist_3methods.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step14_residual_hist_3methods.png")

# ── 图5：高热流区 RMSE 分段对比 ──────────────────────────────────
print("绘制图5：高热流区分段对比...")

hf_bins = [0, 60, 100, 150, 200, 350]
hf_labels = ["0-60", "60-100", "100-150", "150-200", ">200"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# RMSE 分段
ax = axes[0]
x = np.arange(len(hf_labels))
for i, (key, name, color) in enumerate(zip(["A", "B", "C"], METHOD_NAMES, METHOD_COLORS)):
    pred = all_preds[key]
    rmse_bins = []
    for lo, hi in zip(hf_bins[:-1], hf_bins[1:]):
        m = (y_test >= lo) & (y_test < hi)
        if m.sum() > 0:
            rmse_bins.append(float(np.sqrt(np.mean((pred[m] - y_test[m])**2))))
        else:
            rmse_bins.append(0)
    ax.bar(x + i * 0.25, rmse_bins, 0.25, color=color, alpha=0.8, label=name)
ax.set_xticks(x + 0.25)
ax.set_xticklabels(hf_labels, fontsize=10)
ax.set_xlabel("Observed Heat Flow Bin (mW/m²)", fontsize=11)
ax.set_ylabel("RMSE (mW/m²)", fontsize=11)
ax.set_title("RMSE by Heat Flow Bin", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# Bias 分段
ax = axes[1]
for i, (key, name, color) in enumerate(zip(["A", "B", "C"], METHOD_NAMES, METHOD_COLORS)):
    pred = all_preds[key]
    bias_bins = []
    for lo, hi in zip(hf_bins[:-1], hf_bins[1:]):
        m = (y_test >= lo) & (y_test < hi)
        if m.sum() > 0:
            bias_bins.append(float(np.mean(pred[m] - y_test[m])))
        else:
            bias_bins.append(0)
    ax.bar(x + i * 0.25, bias_bins, 0.25, color=color, alpha=0.8, label=name)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.set_xticks(x + 0.25)
ax.set_xticklabels(hf_labels, fontsize=10)
ax.set_xlabel("Observed Heat Flow Bin (mW/m²)", fontsize=11)
ax.set_ylabel("Bias (mW/m²)", fontsize=11)
ax.set_title("Bias by Heat Flow Bin", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("High Heat Flow Region Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step14_high_hf_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step14_high_hf_comparison.png")


# ════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("全部完成！输出文件：")
print(f"  CSV   outputs/step14_comparison_summary.csv")
print(f"  图1   step14_scatter_3methods.png        — 散点图对比")
print(f"  图2   step14_residual_map_3methods.png   — 残差空间分布")
print(f"  图3   step14_basin_comparison.png        — 分洋盆 R²/RMSE")
print(f"  图4   step14_residual_hist_3methods.png  — 残差直方图叠加")
print(f"  图5   step14_high_hf_comparison.png      — 高热流区分段对比")
print("=" * 70)
