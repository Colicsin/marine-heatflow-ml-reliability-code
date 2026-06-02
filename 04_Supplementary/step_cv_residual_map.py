"""
空间 K-Fold 交叉验证残差图
- 对所有 28,642 条真实观测记录，每条都获得一个样本外残差
- 方法：按 2°×2° 空间块做 5-fold 交叉验证（每次留出约 20% 的空间块作测试）
- 输出：全局残差空间分布图 + 残差统计
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
DATA     = ROOT / "data/processed/dataset_D_no_aggregation.csv"
FIG_DIR  = ROOT / "outputs/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "CRUST1.0_moho_depth_0.5deg", "CRUST1.0_upper_crust_thickness_0.5deg",
    "CRUST1.0_mid_crust_thickness_0.5deg", "CRUST1.0_mantle_rho_0.5deg",
    "hotspot_min_hotspot_distance_km", "volcano_latest_vocano_dist",
    "topo_topo_mean", "topo_topo_diff", "topo_topo_median",
    "EMAG2_sealevel", "EMAG2_upcont", "LITH_IDW_lab", "LITH_IDW_moho",
    "oceanic_crust_age_Ma",
]
TARGET = "q"

# ── 读数据 ────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
df = df.dropna(subset=FEATURE_COLS + [TARGET]).reset_index(drop=True)
print(f"总记录数: {len(df):,}  网格数: {df[['grid_lat','grid_lon']].drop_duplicates().shape[0]:,}")

# ── 构建 2°×2° 空间块 ID ──────────────────────────────────────────────────────
BLOCK_SIZE = 2.0
df["block_id"] = (
    (df["grid_lat"] // BLOCK_SIZE * BLOCK_SIZE).astype(str) + "_" +
    (df["grid_lon"] // BLOCK_SIZE * BLOCK_SIZE).astype(str)
)
# 过滤样本数过少的块（<3条）
bc = df["block_id"].value_counts()
df = df[df["block_id"].isin(bc[bc >= 3].index)].reset_index(drop=True)
print(f"过滤后记录数: {len(df):,}  空间块数: {df['block_id'].nunique():,}")

# ── 5-Fold 空间交叉验证 ───────────────────────────────────────────────────────
N_FOLDS = 5
rng = np.random.default_rng(42)
blocks = df["block_id"].unique()
rng.shuffle(blocks)
fold_assignments = {b: i % N_FOLDS for i, b in enumerate(blocks)}
df["fold"] = df["block_id"].map(fold_assignments)

df["cv_pred"]     = np.nan
df["cv_residual"] = np.nan

print(f"\n开始 {N_FOLDS}-Fold 空间交叉验证...")
fold_metrics = []

for fold in range(N_FOLDS):
    tr = df[df["fold"] != fold]
    te = df[df["fold"] == fold]

    model = ExtraTreesRegressor(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    model.fit(tr[FEATURE_COLS].values, tr[TARGET].values)
    pred = model.predict(te[FEATURE_COLS].values)

    df.loc[df["fold"] == fold, "cv_pred"]     = pred
    df.loc[df["fold"] == fold, "cv_residual"] = te[TARGET].values - pred

    r2   = r2_score(te[TARGET].values, pred)
    rmse = float(np.sqrt(mean_squared_error(te[TARGET].values, pred)))
    mae  = float(mean_absolute_error(te[TARGET].values, pred))
    fold_metrics.append((fold, len(te), r2, rmse, mae))
    print(f"  Fold {fold+1}: n_test={len(te):,}  R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}")

# 汇总指标
all_true = df[TARGET].values
all_pred = df["cv_pred"].values
r2_all   = r2_score(all_true, all_pred)
rmse_all = float(np.sqrt(mean_squared_error(all_true, all_pred)))
mae_all  = float(mean_absolute_error(all_true, all_pred))
print(f"\n全局 CV 指标（所有折合并）:")
print(f"  R²={r2_all:.4f}  RMSE={rmse_all:.2f}  MAE={mae_all:.2f}")

# ── 网格级残差（同一网格取中位数）────────────────────────────────────────────
grid_res = (
    df.groupby(["grid_lat", "grid_lon"])["cv_residual"]
    .median()
    .reset_index()
    .rename(columns={"cv_residual": "residual"})
)
print(f"\n网格点数: {len(grid_res):,}")
print(f"残差统计: mean={grid_res.residual.mean():.2f}  std={grid_res.residual.std():.2f}")
print(f"  >+30 mW/m²: {(grid_res.residual > 30).sum()} 个网格（高估）")
print(f"  <-30 mW/m²: {(grid_res.residual < -30).sum()} 个网格（低估）")

# 分洋盆残差
if "basin" in df.columns:
    basin_stats = df.groupby("basin")["cv_residual"].agg(["mean", "std", "count"])
    print(f"\n分洋盆残差:")
    print(basin_stats.to_string())

# ── 绘图 ──────────────────────────────────────────────────────────────────────
vmax = np.percentile(np.abs(grid_res["residual"]), 95)
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig = plt.figure(figsize=(20, 10))
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.OCEAN,     facecolor="#eaf2fb", zorder=0)
ax.add_feature(cfeature.LAND,      facecolor="#d4d4d4", edgecolor="none", zorder=2)
ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=3)
ax.gridlines(linewidth=0.3, color="gray", alpha=0.4, linestyle="--")

sc = ax.scatter(
    grid_res["grid_lon"].values,
    grid_res["grid_lat"].values,
    c=grid_res["residual"].values,
    cmap="RdBu_r", norm=norm,
    s=12, marker="s", linewidths=0, alpha=0.9,
    transform=ccrs.PlateCarree(), zorder=4
)

cbar = plt.colorbar(sc, ax=ax, orientation="horizontal",
                    pad=0.04, shrink=0.6, aspect=40)
cbar.set_label(
    f"CV Residual (Obs − Pred, mW/m²)   "
    f"Blue = Underestimate  |  Red = Overestimate\n"
    f"Overall: R²={r2_all:.4f}  RMSE={rmse_all:.2f}  MAE={mae_all:.2f}",
    fontsize=10
)
cbar.ax.tick_params(labelsize=9)

ax.set_title(
    f"Spatial Distribution of Cross-Validation Residuals\n"
    f"(ExtraTrees, 5-Fold Spatial Block CV, 2°×2° blocks, n={len(grid_res):,} grids)",
    fontsize=13, fontweight="bold", pad=12
)

out_path = FIG_DIR / "step_cv_residual_map.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n图已保存: {out_path}")

# ── 额外输出：高残差区域列表 ──────────────────────────────────────────────────
print("\n低估最严重的20个网格（残差 < -30）:")
print(grid_res[grid_res.residual < -30].sort_values("residual").head(20).to_string(index=False))
print("\n高估最严重的20个网格（残差 > +30）:")
print(grid_res[grid_res.residual > 30].sort_values("residual", ascending=False).head(20).to_string(index=False))
