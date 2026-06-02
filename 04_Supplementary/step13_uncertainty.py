"""
实验：ExtraTrees 预测不确定性估计
原理：ExtraTrees 由多棵决策树组成，对同一输入各棵树给出不同预测值。
      用各棵树预测值的标准差作为不确定性（epistemic uncertainty）。

输出：
  CSV:
    outputs/step13_test_uncertainty.csv        — 测试集预测值 + 不确定性
    outputs/step13_global_uncertainty.csv      — 全球网格预测值 + 不确定性
  图:
    outputs/figures/step13_uncertainty_map.png         — 全球不确定性空间分布
    outputs/figures/step13_test_uncertainty_map.png    — 测试集不确定性空间分布
    outputs/figures/step13_uncertainty_vs_error.png    — 不确定性 vs 实际误差关系
    outputs/figures/step13_calibration.png             — 不确定性校准曲线
    outputs/figures/step13_basin_uncertainty.png       — 各洋盆不确定性分布
"""

from pathlib import Path
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

def predict_with_uncertainty(model, X):
    """
    用各棵树的预测值计算均值和标准差。
    返回: mean_pred, std_pred (shape: [n_samples])
    """
    # shape: (n_estimators, n_samples)
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    mean_pred = tree_preds.mean(axis=0)
    std_pred  = tree_preds.std(axis=0)
    return mean_pred, std_pred

def base_map(ax, title, fontsize=12):
    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="#d8d3cc", zorder=2)
    ax.add_feature(cfeature.OCEAN,     facecolor="#eaf4fb", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#444444", zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)

# ════════════════════════════════════════════════════════════════════
# 1. 加载数据，划分训练/测试集
# ════════════════════════════════════════════════════════════════════
print("=" * 65)
print("加载数据...")
df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS + [TARGET])
print(f"  记录数: {len(df):,}")

train_df, test_df = spatial_block_split(df)
print(f"  训练集: {len(train_df):,}  测试集: {len(test_df):,}")

X_train = train_df[FEATURE_COLS].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df[TARGET].values

# ════════════════════════════════════════════════════════════════════
# 2. 训练模型（n_estimators=300，树越多不确定性估计越稳定）
# ════════════════════════════════════════════════════════════════════
print("\n训练 ExtraTrees（n_estimators=300）...")
model_cv = ExtraTreesRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
model_cv.fit(X_train, train_df[TARGET].values)

print("计算测试集预测均值和不确定性...")
mean_pred, std_pred = predict_with_uncertainty(model_cv, X_test)

# 基础指标
r2   = r2_score(y_test, mean_pred)
rmse = float(np.sqrt(mean_squared_error(y_test, mean_pred)))
mae  = float(mean_absolute_error(y_test, mean_pred))
bias = float(np.mean(mean_pred - y_test))
abs_error = np.abs(mean_pred - y_test)
print(f"  测试集: R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  Bias={bias:.2f}")
print(f"  不确定性(std): 均值={std_pred.mean():.2f}  中位数={np.median(std_pred):.2f}  "
      f"P95={np.percentile(std_pred, 95):.2f} mW/m²")

# 保存测试集结果
test_out = test_df[["lat_NS", "long_EW", "grid_lat", "grid_lon", "basin", TARGET]].copy()
test_out["q_pred"]       = mean_pred
test_out["q_uncertainty"] = std_pred
test_out["abs_error"]    = abs_error
test_out["residual"]     = mean_pred - y_test
test_out.to_csv(OUT_DIR / "step13_test_uncertainty.csv", index=False)
print(f"  已保存: outputs/step13_test_uncertainty.csv")

# ════════════════════════════════════════════════════════════════════
# 3. 全量训练 → 全球网格不确定性
# ════════════════════════════════════════════════════════════════════
print("\n全量训练（全部数据）...")
model_full = ExtraTreesRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
model_full.fit(df[FEATURE_COLS].values, df[TARGET].values)

print("加载全球网格，计算全球不确定性（可能需要几分钟）...")
globe = pd.read_csv(GLOBE_PATH, usecols=["lon", "lat"] + FEATURE_COLS)
globe = globe.dropna(subset=FEATURE_COLS)
print(f"  全球网格: {len(globe):,} 个")

globe_mean, globe_std = predict_with_uncertainty(model_full, globe[FEATURE_COLS].values)
globe_out = globe[["lon", "lat"]].copy()
globe_out["q_pred_mWm2"]       = globe_mean
globe_out["q_uncertainty_mWm2"] = globe_std
globe_out.to_csv(OUT_DIR / "step13_global_uncertainty.csv", index=False)
print(f"  已保存: outputs/step13_global_uncertainty.csv")
print(f"  全球不确定性: 均值={globe_std.mean():.2f}  P95={np.percentile(globe_std, 95):.2f} mW/m²")

# ════════════════════════════════════════════════════════════════════
# 图1：全球不确定性空间分布
# ════════════════════════════════════════════════════════════════════
print("\n绘制图1：全球不确定性空间分布...")

fig, axes = plt.subplots(2, 1, figsize=(18, 14),
                         subplot_kw={"projection": ccrs.Robinson()})

# 上图：预测值
base_map(axes[0], f"Global Ocean Heat Flow Prediction (ExtraTrees, n={len(globe_out):,})")
norm_hf = mcolors.Normalize(vmin=20, vmax=180)
sc0 = axes[0].scatter(
    globe_out["lon"], globe_out["lat"],
    c=globe_out["q_pred_mWm2"], cmap="RdYlBu_r", norm=norm_hf,
    s=1.5, alpha=0.85, linewidths=0,
    transform=ccrs.PlateCarree(), zorder=4
)
cb0 = plt.colorbar(sc0, ax=axes[0], orientation="horizontal",
                   pad=0.04, fraction=0.025, aspect=50, extend="both")
cb0.set_label("Predicted Heat Flow (mW/m²)", fontsize=11)

# 下图：不确定性
base_map(axes[1], f"Prediction Uncertainty (1σ, Tree Ensemble Std)\n"
                  f"Mean={globe_std.mean():.1f}  P95={np.percentile(globe_std,95):.1f} mW/m²")
norm_unc = mcolors.Normalize(vmin=0, vmax=np.percentile(globe_std, 95))
sc1 = axes[1].scatter(
    globe_out["lon"], globe_out["lat"],
    c=globe_out["q_uncertainty_mWm2"], cmap="YlOrRd", norm=norm_unc,
    s=1.5, alpha=0.85, linewidths=0,
    transform=ccrs.PlateCarree(), zorder=4
)
cb1 = plt.colorbar(sc1, ax=axes[1], orientation="horizontal",
                   pad=0.04, fraction=0.025, aspect=50, extend="max")
cb1.set_label("Uncertainty 1σ (mW/m²)", fontsize=11)

plt.tight_layout()
fig.savefig(FIG_DIR / "step13_uncertainty_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step13_uncertainty_map.png")

# ════════════════════════════════════════════════════════════════════
# 图2：测试集不确定性空间分布
# ════════════════════════════════════════════════════════════════════
print("绘制图2：测试集不确定性空间分布...")

fig, ax = plt.subplots(figsize=(16, 8),
                       subplot_kw={"projection": ccrs.Robinson()})
base_map(ax, f"Test Set Prediction Uncertainty (1σ)\n"
             f"n={len(test_df):,}  Mean σ={std_pred.mean():.1f} mW/m²")

norm_unc_t = mcolors.Normalize(vmin=0, vmax=np.percentile(std_pred, 95))
sc2 = ax.scatter(
    test_df["long_EW"], test_df["lat_NS"],
    c=std_pred, cmap="YlOrRd", norm=norm_unc_t,
    s=6, alpha=0.85, linewidths=0,
    transform=ccrs.PlateCarree(), zorder=4
)
cb2 = plt.colorbar(sc2, ax=ax, orientation="horizontal",
                   pad=0.04, fraction=0.025, aspect=50, extend="max")
cb2.set_label("Uncertainty 1σ (mW/m²)", fontsize=11)

plt.tight_layout()
fig.savefig(FIG_DIR / "step13_test_uncertainty_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step13_test_uncertainty_map.png")

# ════════════════════════════════════════════════════════════════════
# 图3：不确定性 vs 实际误差关系
# ════════════════════════════════════════════════════════════════════
print("绘制图3：不确定性 vs 实际误差...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左：散点图
ax = axes[0]
basin_colors = {"Pacific": "#3a7ebf", "Atlantic": "#e06c3a", "Indian": "#4caf7d"}
for basin, color in basin_colors.items():
    mask = test_df["basin"].values == basin
    ax.scatter(std_pred[mask], abs_error[mask],
               c=color, s=3, alpha=0.35, linewidths=0, label=basin)

# 分箱均值趋势线
bins_unc = np.percentile(std_pred, np.linspace(0, 100, 21))
bin_centers, bin_means = [], []
for lo, hi in zip(bins_unc[:-1], bins_unc[1:]):
    m = (std_pred >= lo) & (std_pred < hi)
    if m.sum() > 5:
        bin_centers.append((lo + hi) / 2)
        bin_means.append(abs_error[m].mean())
ax.plot(bin_centers, bin_means, "k-o", linewidth=2, markersize=4,
        label="Binned mean error", zorder=5)

ax.set_xlabel("Predicted Uncertainty σ (mW/m²)", fontsize=11)
ax.set_ylabel("Actual Absolute Error (mW/m²)", fontsize=11)
ax.set_title("Uncertainty vs Actual Error\n(higher σ → higher error = well-calibrated)",
             fontsize=10, fontweight="bold")
ax.legend(fontsize=9, markerscale=2)
ax.spines[["top", "right"]].set_visible(False)

# 右：不确定性分位数 vs 误差覆盖率（校准曲线）
ax = axes[1]
# 理想校准：预测区间 [pred ± k*σ] 应覆盖 ~68% (k=1), ~95% (k=2) 的真实值
k_values = np.linspace(0.1, 3.0, 30)
coverage = []
for k in k_values:
    lower = mean_pred - k * std_pred
    upper = mean_pred + k * std_pred
    covered = ((y_test >= lower) & (y_test <= upper)).mean()
    coverage.append(covered)

# 理想高斯校准曲线
from scipy import stats as scipy_stats
ideal_coverage = [2 * scipy_stats.norm.cdf(k) - 1 for k in k_values]

ax.plot(k_values, coverage,       "b-o", markersize=3, linewidth=2, label="Actual coverage")
ax.plot(k_values, ideal_coverage, "r--",               linewidth=1.5, label="Ideal (Gaussian)")
ax.fill_between(k_values, ideal_coverage, coverage,
                alpha=0.15, color="blue", label="Calibration gap")
ax.axhline(0.68, color="gray", linewidth=0.8, linestyle=":")
ax.axhline(0.95, color="gray", linewidth=0.8, linestyle=":")
ax.text(2.8, 0.695, "68%", fontsize=8, color="gray")
ax.text(2.8, 0.960, "95%", fontsize=8, color="gray")
ax.set_xlabel("Coverage Factor k  (interval: pred ± k·σ)", fontsize=11)
ax.set_ylabel("Coverage Rate", fontsize=11)
ax.set_title("Uncertainty Calibration Curve", fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.set_xlim(0, 3.1); ax.set_ylim(0, 1.05)
ax.spines[["top", "right"]].set_visible(False)

# 打印关键覆盖率
for k_target in [1.0, 1.645, 2.0]:
    idx = np.argmin(np.abs(k_values - k_target))
    print(f"  k={k_target:.3f}: 实际覆盖率={coverage[idx]:.3f}  理想={ideal_coverage[idx]:.3f}")

plt.suptitle("Uncertainty Quality Assessment", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step13_uncertainty_vs_error.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step13_uncertainty_vs_error.png")

# ════════════════════════════════════════════════════════════════════
# 图4：各洋盆不确定性分布
# ════════════════════════════════════════════════════════════════════
print("绘制图4：各洋盆不确定性分布...")

basins = ["Pacific", "Atlantic", "Indian"]
colors = {"Pacific": "#3a7ebf", "Atlantic": "#e06c3a", "Indian": "#4caf7d"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, basin in enumerate(basins):
    mask = test_df["basin"].values == basin
    unc_b = std_pred[mask]
    err_b = abs_error[mask]

    ax = axes[i]
    ax.hist(unc_b, bins=50, color=colors[basin], alpha=0.75, edgecolor="none",
            label=f"σ  mean={unc_b.mean():.1f}")
    ax.axvline(unc_b.mean(),   color="#333333", linewidth=1.5, linestyle="--")
    ax.axvline(err_b.mean(),   color="#cc3333", linewidth=1.5, linestyle="-",
               label=f"MAE={err_b.mean():.1f}")

    ax.set_title(f"{basin} Ocean\nn={mask.sum():,}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Uncertainty σ (mW/m²)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend(fontsize=9)
    ax.text(0.97, 0.95,
            f"P50={np.median(unc_b):.1f}\nP95={np.percentile(unc_b,95):.1f}",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("Prediction Uncertainty Distribution by Ocean Basin\n"
             "(dashed = mean σ,  red = MAE — ideally σ ≈ MAE)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step13_basin_uncertainty.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step13_basin_uncertainty.png")

# ════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════
print()
print("=" * 65)
print("全部完成！输出文件：")
print(f"  CSV  outputs/step13_test_uncertainty.csv    ({len(test_df):,} 行)")
print(f"  CSV  outputs/step13_global_uncertainty.csv  ({len(globe_out):,} 行)")
print(f"  图1  step13_uncertainty_map.png         — 全球预测值 + 不确定性双图")
print(f"  图2  step13_test_uncertainty_map.png    — 测试集不确定性空间分布")
print(f"  图3  step13_uncertainty_vs_error.png    — 不确定性 vs 误差 + 校准曲线")
print(f"  图4  step13_basin_uncertainty.png       — 各洋盆不确定性分布")
print("=" * 65)
