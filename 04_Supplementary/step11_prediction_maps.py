"""Generate global prediction maps and residual diagnostics from the Dataset D model."""

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


def calc_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    return r2, rmse, mae, bias

def base_map(ax, title, fontsize=12):
    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="#d8d3cc", zorder=2)
    ax.add_feature(cfeature.OCEAN,     facecolor="#eaf4fb", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#444444", zorder=3)
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray",
                 alpha=0.4, linestyle="--")
    ax.set_title(title, fontsize=fontsize, fontweight="bold", pad=8)


print("=" * 60)
print("加载观测数据...")
df = pd.read_csv(DATA_PATH)
print(f"  记录数: {len(df):,}  网格数: {df[['grid_lat','grid_lon']].drop_duplicates().shape[0]:,}")

print("空间分组 2°×2° 划分训练/测试集...")
df["block_id"] = ((df["grid_lat"] // 2) * 2).astype(str) + "_" +\
                 ((df["grid_lon"] // 2) * 2).astype(str)
valid_blocks = df["block_id"].value_counts()
valid_blocks = valid_blocks[valid_blocks >= 3].index
df2 = df[df["block_id"].isin(valid_blocks)].copy()

rng = np.random.default_rng(42)
blocks = df2["block_id"].unique()
rng.shuffle(blocks)
test_blocks = set(blocks[:int(len(blocks) * 0.3)])

train_df = df2[~df2["block_id"].isin(test_blocks)].copy()
test_df  = df2[ df2["block_id"].isin(test_blocks)].copy()
print(f"  训练集: {len(train_df):,}  测试集: {len(test_df):,}  blocks: {len(blocks)}")


print("训练 ExtraTrees（空间分组训练集）...")
model_cv = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model_cv.fit(train_df[FEATURE_COLS].values, train_df[TARGET].values)

test_df = test_df.copy()
test_df["q_pred"]   = model_cv.predict(test_df[FEATURE_COLS].values)
test_df["residual"] = test_df["q_pred"] - test_df[TARGET]
test_df["abs_resid"]= test_df["residual"].abs()

r2, rmse, mae, bias = calc_metrics(test_df[TARGET], test_df["q_pred"])
print(f"  测试集: R²={r2:.4f}  RMSE={rmse:.2f}  MAE={mae:.2f}  Bias={bias:.2f}")


save_cols = ["lat_NS", "long_EW", "grid_lat", "grid_lon", "basin", "qc_m",
             TARGET, "q_pred", "residual", "abs_resid"]
test_df[save_cols].to_csv(OUT_DIR / "step11_test_predictions.csv", index=False)
print(f"  已保存: outputs/step11_test_predictions.csv")


print("全量训练 ExtraTrees（全部 28,642 条记录）...")
model_full = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model_full.fit(df[FEATURE_COLS].values, df[TARGET].values)

print("加载全球网格特征文件...")
globe = pd.read_csv(GLOBE_PATH, usecols=["lon", "lat"] + FEATURE_COLS)
globe = globe.dropna(subset=FEATURE_COLS)
print(f"  全球网格: {len(globe):,} 个")

globe["q_pred"] = model_full.predict(globe[FEATURE_COLS].values)
globe_out = globe[["lon", "lat", "q_pred"]].copy()
globe_out.columns = ["lon", "lat", "q_pred_mWm2"]
globe_out.to_csv(OUT_DIR / "step11_global_predictions.csv", index=False)
print(f"  已保存: outputs/step11_global_predictions.csv")
print(f"  预测值范围: {globe_out['q_pred_mWm2'].min():.1f} ~ {globe_out['q_pred_mWm2'].max():.1f} mW/m²")


print("\n绘制图1：全球海洋热流预测图...")

fig, ax = plt.subplots(figsize=(18, 9),
                       subplot_kw={"projection": ccrs.Robinson()})
base_map(ax, "Global Ocean Heat Flow Prediction (ExtraTrees, Full Training Set)\n"
             f"n={len(globe_out):,} grids  |  0.5° resolution")

cmap_hf = plt.cm.get_cmap("RdYlBu_r")
norm_hf = mcolors.Normalize(vmin=20, vmax=180)

sc = ax.scatter(
    globe_out["lon"], globe_out["lat"],
    c=globe_out["q_pred_mWm2"], cmap=cmap_hf, norm=norm_hf,
    s=1.5, alpha=0.85, linewidths=0,
    transform=ccrs.PlateCarree(), zorder=4
)
cbar = plt.colorbar(sc, ax=ax, orientation="horizontal",
                    pad=0.04, fraction=0.025, aspect=50, extend="both")
cbar.set_label("Predicted Heat Flow (mW/m²)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / "step11_global_pred_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: step11_global_pred_map.png")


print("绘制图2：测试集预测值空间分布...")

fig, ax = plt.subplots(figsize=(16, 8),
                       subplot_kw={"projection": ccrs.Robinson()})
base_map(ax, f"Test Set — Predicted Heat Flow (2°×2° Spatial Block)\n"
             f"n={len(test_df):,}  R²={r2:.3f}  RMSE={rmse:.1f}  MAE={mae:.1f} mW/m²")

sc2 = ax.scatter(
    test_df["long_EW"], test_df["lat_NS"],
    c=test_df["q_pred"], cmap=cmap_hf, norm=norm_hf,
    s=5, alpha=0.8, linewidths=0,
    transform=ccrs.PlateCarree(), zorder=4
)
cbar2 = plt.colorbar(sc2, ax=ax, orientation="horizontal",
                     pad=0.04, fraction=0.025, aspect=50, extend="both")
cbar2.set_label("Predicted Heat Flow (mW/m²)", fontsize=11)
cbar2.ax.tick_params(labelsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / "step11_test_pred_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: step11_test_pred_map.png")


print("绘制图3：测试集残差空间分布...")

fig, ax = plt.subplots(figsize=(16, 8),
                       subplot_kw={"projection": ccrs.Robinson()})
base_map(ax, f"Test Set — Residuals (Predicted − Observed)\n"
             f"Bias={bias:.2f} mW/m²  |  Red = overestimate, Blue = underestimate")

norm_res = mcolors.TwoSlopeNorm(vmin=-80, vcenter=0, vmax=80)
sc3 = ax.scatter(
    test_df["long_EW"], test_df["lat_NS"],
    c=test_df["residual"], cmap="RdBu_r", norm=norm_res,
    s=5, alpha=0.8, linewidths=0,
    transform=ccrs.PlateCarree(), zorder=4
)
cbar3 = plt.colorbar(sc3, ax=ax, orientation="horizontal",
                     pad=0.04, fraction=0.025, aspect=50, extend="both")
cbar3.set_label("Residual (mW/m²)", fontsize=11)
cbar3.ax.tick_params(labelsize=9)

plt.tight_layout()
fig.savefig(FIG_DIR / "step11_test_residual_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: step11_test_residual_map.png")


print("绘制图4：各洋盆残差直方图...")

basins = ["Pacific", "Atlantic", "Indian"]
basin_colors = {"Pacific": "#3a7ebf", "Atlantic": "#e06c3a", "Indian": "#4caf7d"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for i, basin in enumerate(basins):
    sub = test_df[test_df["basin"] == basin]
    r2_b, rmse_b, mae_b, bias_b = calc_metrics(sub[TARGET], sub["q_pred"])

    axes[i].hist(sub["residual"], bins=50, range=(-150, 150),
                 color=basin_colors[basin], alpha=0.8, edgecolor="none")
    axes[i].axvline(0,        color="black",  linewidth=1.5, linestyle="--", label="zero")
    axes[i].axvline(bias_b,   color="#cc3333", linewidth=1.5, linestyle="-",
                    label=f"Bias={bias_b:.1f}")

    axes[i].set_title(f"{basin} Ocean\nn={len(sub):,}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel("Residual (mW/m²)", fontsize=10)
    axes[i].set_ylabel("Count", fontsize=10)
    axes[i].legend(fontsize=9)
    axes[i].spines[["top", "right"]].set_visible(False)


    axes[i].text(0.97, 0.95,
                 f"R²={r2_b:.3f}\nRMSE={rmse_b:.1f}\nMAE={mae_b:.1f}",
                 transform=axes[i].transAxes, fontsize=9,
                 va="top", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.suptitle("Residual Distribution by Ocean Basin", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step11_basin_residual_hist.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: step11_basin_residual_hist.png")


print("绘制图5：大残差点空间分布（|residual| > 30）...")

large = test_df[test_df["abs_resid"] > 30].copy()
small = test_df[test_df["abs_resid"] <= 30].copy()
n_large = len(large)
pct = 100 * n_large / len(test_df)

fig, ax = plt.subplots(figsize=(16, 8),
                       subplot_kw={"projection": ccrs.Robinson()})
base_map(ax, f"Large Residuals  |residual| > 30 mW/m²\n"
             f"n={n_large:,} / {len(test_df):,} ({pct:.1f}%)  "
             f"|  Red = overestimate, Blue = underestimate")


ax.scatter(small["long_EW"], small["lat_NS"],
           c="#bbbbbb", s=2.5, alpha=0.3, linewidths=0,
           transform=ccrs.PlateCarree(), zorder=4)


def sscale(v, vmin=30, s0=15, s1=120):
    vmax = max(v.max(), vmin + 1)
    return s0 + (v - vmin) / (vmax - vmin) * (s1 - s0)

over  = large[large["residual"] >  0]
under = large[large["residual"] <= 0]

if len(over) > 0:
    ax.scatter(over["long_EW"], over["lat_NS"],
               c=over["residual"], cmap="Reds",
               norm=mcolors.Normalize(vmin=30, vmax=120),
               s=sscale(over["abs_resid"]),
               alpha=0.85, linewidths=0.2, edgecolors="#7a0000",
               transform=ccrs.PlateCarree(), zorder=5,
               label=f"Overestimate  n={len(over):,}")

if len(under) > 0:
    ax.scatter(under["long_EW"], under["lat_NS"],
               c=under["abs_resid"], cmap="Blues",
               norm=mcolors.Normalize(vmin=30, vmax=120),
               s=sscale(under["abs_resid"]),
               alpha=0.85, linewidths=0.2, edgecolors="#00205a",
               transform=ccrs.PlateCarree(), zorder=5,
               label=f"Underestimate  n={len(under):,}")


for cmap_name, label, pad in [("Reds", "Overestimate (mW/m²)", 0.01),
                                ("Blues", "Underestimate (mW/m²)", 0.06)]:
    sm = plt.cm.ScalarMappable(cmap=cmap_name,
                                norm=mcolors.Normalize(vmin=30, vmax=120))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation="vertical",
                      pad=pad, fraction=0.016, aspect=28, extend="max")
    cb.set_label(label, fontsize=9)
    cb.ax.tick_params(labelsize=8)

ax.legend(loc="lower left", fontsize=9, framealpha=0.85,
          markerscale=1.0, scatterpoints=1)

plt.tight_layout()
fig.savefig(FIG_DIR / "step11_large_residual_map.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: step11_large_residual_map.png")


print("绘制图6：观测 vs 预测散点图...")

fig, ax = plt.subplots(figsize=(7, 7))

for basin, color in basin_colors.items():
    sub = test_df[test_df["basin"] == basin]
    ax.scatter(sub[TARGET], sub["q_pred"],
               c=color, s=4, alpha=0.45, linewidths=0, label=basin)

lim = (0, 250)
ax.plot(lim, lim, "k--", linewidth=1.2, label="1:1 line")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xlabel("Observed Heat Flow (mW/m²)", fontsize=12)
ax.set_ylabel("Predicted Heat Flow (mW/m²)", fontsize=12)
ax.set_title(f"Observed vs Predicted  (Test Set)\n"
             f"R²={r2:.3f}  RMSE={rmse:.1f}  MAE={mae:.1f}  Bias={bias:.2f} mW/m²",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10, markerscale=2)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
fig.savefig(FIG_DIR / "step11_scatter_obs_pred.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"  已保存: step11_scatter_obs_pred.png")


print()
print("=" * 60)
print("全部完成！输出文件：")
print(f"  CSV  outputs/step11_global_predictions.csv   ({len(globe_out):,} 行)")
print(f"  CSV  outputs/step11_test_predictions.csv     ({len(test_df):,} 行)")
print(f"  图1  step11_global_pred_map.png")
print(f"  图2  step11_test_pred_map.png")
print(f"  图3  step11_test_residual_map.png")
print(f"  图4  step11_basin_residual_hist.png")
print(f"  图5  step11_large_residual_map.png")
print(f"  图6  step11_scatter_obs_pred.png")
print("=" * 60)
