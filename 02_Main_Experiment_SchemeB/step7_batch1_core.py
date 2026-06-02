"""
第一批核心实验：6模型对比 + 验证策略对比 + 空间自相关分析
对应论文表6、表7、表8、图4、图5
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                             median_absolute_error)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/dataset_D_no_aggregation.csv"
FIG_DIR = ROOT / "outputs/figures"
OUT_DIR = ROOT / "outputs"
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

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=FEATURE_COLS + [TARGET])
print(f"Dataset D: {len(df):,} records, {df[['grid_lat','grid_lon']].drop_duplicates().shape[0]:,} grids\n")

MODELS = {
    "LinearReg":  lambda: LinearRegression(),
    "RF":         lambda: RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    "ExtraTrees": lambda: ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    "GBDT":       lambda: GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
    "XGBoost":    lambda: XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0, n_jobs=-1),
    "LightGBM":   lambda: LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1, n_jobs=-1),
}

def calc_full_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))
    # NSE = 1 - sum((y-yhat)^2) / sum((y-ymean)^2)
    nse  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    # MAPE: 排除 y_true==0
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "MedAE": medae,
            "MAPE": mape, "Bias": bias, "NSE": nse}

# ── 空间分组工具函数 ──
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
    return tr, te, len(blocks)

# ── Moran's I 计算（K近邻） ──
def calc_moran_knn(coords, values, k=8):
    """手动计算 K近邻 Moran's I，避免大数据量下 libpysal 内存问题"""
    from scipy.spatial import cKDTree
    n = len(values)
    z = values - np.mean(values)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k+1)  # +1 因为包含自身
    numerator = 0.0
    W = 0.0
    for i in range(n):
        for j_idx in range(1, k+1):  # 跳过自身
            j = indices[i, j_idx]
            numerator += z[i] * z[j]
            W += 1.0
    denom = np.sum(z**2)
    I = (n / W) * (numerator / denom)
    return I

# ── Moran's I 距离阈值 ──
def calc_moran_distance(coords, values, threshold_km):
    """距离阈值 Moran's I（采样计算，避免内存爆炸）"""
    from scipy.spatial import cKDTree
    # 将度转换为近似km（赤道附近 1度≈111km）
    coords_km = coords.copy()
    coords_km[:, 0] *= 111.0  # lat
    coords_km[:, 1] *= 111.0 * np.cos(np.radians(np.mean(coords[:, 0])))  # lon
    n = len(values)
    z = values - np.mean(values)
    tree = cKDTree(coords_km)
    numerator = 0.0
    W = 0.0
    for i in range(n):
        neighbors = tree.query_ball_point(coords_km[i], threshold_km)
        for j in neighbors:
            if i != j:
                numerator += z[i] * z[j]
                W += 1.0
    denom = np.sum(z**2)
    if W == 0:
        return np.nan
    I = (n / W) * (numerator / denom)
    return I

# ── Geary's C ──
def calc_geary_c(coords, values, k=8):
    from scipy.spatial import cKDTree
    n = len(values)
    z = values - np.mean(values)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k+1)
    numerator = 0.0
    W = 0.0
    for i in range(n):
        for j_idx in range(1, k+1):
            j = indices[i, j_idx]
            numerator += (values[i] - values[j])**2
            W += 1.0
    denom = np.sum(z**2)
    C = ((n - 1) / (2 * W)) * (numerator / denom)
    return C

# ═══════════════════════════════════════════════════════════════════
# 实验1：6模型对比（空间分组2°×2°）—— 对应论文表6
# ═══════════════════════════════════════════════════════════════════
print("=" * 80)
print("实验1：6模型对比（空间分组 2°×2°）—— 对应论文表6")
print("=" * 80)

tr, te, n_blocks = spatial_block_split(df)
print(f"训练集: {len(tr):,}, 测试集: {len(te):,}, blocks: {n_blocks}\n")

header = f"{'模型':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'MedAE':>8} {'MAPE%':>8} {'Bias':>8} {'NSE':>8} {'耗时(s)':>8}"
print(header)
print("-" * len(header))

exp1_results = []
exp1_models = {}  # 保存训练好的模型
for mname, mfunc in MODELS.items():
    model = mfunc()
    t0 = time.time()
    model.fit(tr[FEATURE_COLS].values, tr[TARGET].values)
    train_time = time.time() - t0
    pred = model.predict(te[FEATURE_COLS].values)
    m = calc_full_metrics(te[TARGET].values, pred)
    m["time"] = train_time
    exp1_results.append((mname, m))
    exp1_models[mname] = (model, pred)
    print(f"{mname:<12} {m['R2']:>8.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f} "
          f"{m['MedAE']:>8.2f} {m['MAPE']:>8.2f} {m['Bias']:>8.2f} "
          f"{m['NSE']:>8.4f} {m['time']:>8.2f}")

# 6模型散点图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
for idx, (mname, m) in enumerate(exp1_results):
    ax = axes[idx // 3, idx % 3]
    _, pred = exp1_models[mname]
    y_true = te[TARGET].values
    ax.scatter(y_true, pred, s=1, alpha=0.2, c="steelblue")
    ax.plot([0, 250], [0, 250], "r--", lw=1)
    ax.set_xlim(0, 260); ax.set_ylim(0, 260)
    ax.set_aspect("equal")
    ax.set_title(f"{mname} (R²={m['R2']:.4f})", fontsize=11)
    ax.set_xlabel("Observed (mW/m²)")
    ax.set_ylabel("Predicted (mW/m²)")
plt.suptitle("6 Models - Spatial Block 2°×2° Validation", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step7_scatter_6models.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\n  图已保存: {FIG_DIR / 'step7_scatter_6models.png'}")

# ═══════════════════════════════════════════════════════════════════
# 实验2：三种验证策略对比（ExtraTrees）—— 对应论文表7
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("实验2：三种验证策略对比（ExtraTrees）—— 对应论文表7")
print("=" * 80)

exp2_results = []

# 方案1：随机划分
X_all, y_all = df[FEATURE_COLS].values, df[TARGET].values
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
et = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
et.fit(X_tr, y_tr)
pred_random = et.predict(X_te)
m_random = calc_full_metrics(y_te, pred_random)
# Moran's I（随机划分的测试集坐标）
te_idx = df.index[np.isin(np.arange(len(df)),
    train_test_split(np.arange(len(df)), test_size=0.3, random_state=42)[1])]
coords_random = df.loc[te_idx, ["grid_lat", "grid_lon"]].values
residuals_random = y_te - pred_random
moran_random = calc_moran_knn(coords_random, residuals_random, k=8)
exp2_results.append(("随机划分", m_random, moran_random, len(y_te)))

# 方案2：空间分组 2°×2°
tr2, te2, _ = spatial_block_split(df)
et2 = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
et2.fit(tr2[FEATURE_COLS].values, tr2[TARGET].values)
pred_spatial = et2.predict(te2[FEATURE_COLS].values)
m_spatial = calc_full_metrics(te2[TARGET].values, pred_spatial)
coords_spatial = te2[["grid_lat", "grid_lon"]].values
residuals_spatial = te2[TARGET].values - pred_spatial
moran_spatial = calc_moran_knn(coords_spatial, residuals_spatial, k=8)
exp2_results.append(("空间分组2°×2°", m_spatial, moran_spatial, len(te2)))

# 方案3：跨洋盆
for basin in ["Pacific", "Atlantic", "Indian"]:
    tr_b = df[df["basin"] != basin]
    te_b = df[df["basin"] == basin]
    et3 = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et3.fit(tr_b[FEATURE_COLS].values, tr_b[TARGET].values)
    pred_b = et3.predict(te_b[FEATURE_COLS].values)
    m_b = calc_full_metrics(te_b[TARGET].values, pred_b)
    coords_b = te_b[["grid_lat", "grid_lon"]].values
    res_b = te_b[TARGET].values - pred_b
    moran_b = calc_moran_knn(coords_b, res_b, k=8)
    exp2_results.append((f"跨盆-{basin}", m_b, moran_b, len(te_b)))

print(f"\n{'方案':<16} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8} {'Moran I':>8}")
print("-" * 72)
for label, m, mi, n in exp2_results:
    print(f"{label:<16} {n:>8,} {m['R2']:>8.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f} "
          f"{m['Bias']:>8.2f} {mi:>8.4f}")

# ═══════════════════════════════════════════════════════════════════
# 实验3：空间自相关深度分析（空间分组2°×2°残差）—— 对应论文表8、图4、图5
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("实验3：空间自相关深度分析 —— 对应论文表8、图4、图5")
print("=" * 80)

# 3a: 多种方法计算 Moran's I / Geary's C
print("\n--- 3a: 空间自相关指标（空间分组2°×2°残差）---")
print(f"  K近邻 Moran's I (k=8): {moran_spatial:.4f}")

# 距离阈值 Moran's I（采样加速：随机取5000个点）
n_sample = min(5000, len(residuals_spatial))
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(residuals_spatial), n_sample, replace=False)
coords_sample = coords_spatial[sample_idx]
res_sample = residuals_spatial[sample_idx]

for d_km in [200, 400, 600]:
    mi_d = calc_moran_distance(coords_sample, res_sample, d_km)
    print(f"  距离阈值 Moran's I ({d_km}km, n={n_sample}): {mi_d:.4f}")

gc = calc_geary_c(coords_spatial, residuals_spatial, k=8)
print(f"  Geary's C (k=8): {gc:.4f}")

# 3b: Variogram
print("\n--- 3b: Variogram 分析 ---")
try:
    from skgstat import Variogram
    # 采样（variogram 对大数据量很慢）
    n_vario = min(3000, len(residuals_spatial))
    vidx = rng.choice(len(residuals_spatial), n_vario, replace=False)
    v_coords = coords_spatial[vidx] * [111.0, 111.0 * np.cos(np.radians(np.mean(coords_spatial[:, 0])))]
    v_values = residuals_spatial[vidx]
    V = Variogram(v_coords, v_values, model="spherical", n_lags=25, maxlag=5000)
    print(f"  Variogram 模型: spherical")
    print(f"  Range (空间相关长度): {V.parameters[0]:.0f} km")
    print(f"  Sill: {V.parameters[1]:.2f}")
    print(f"  Nugget: {V.parameters[2]:.2f}")

    # 绘制 variogram
    fig_v, ax_v = plt.subplots(figsize=(8, 5))
    V.plot(axes=ax_v, show=False)
    ax_v.set_xlabel("Distance (km)", fontsize=12)
    ax_v.set_ylabel("Semivariance", fontsize=12)
    ax_v.set_title("Variogram of Prediction Residuals (Spatial Block 2°×2°)", fontsize=13)
    fig_v.savefig(FIG_DIR / "step7_variogram.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {FIG_DIR / 'step7_variogram.png'}")
except Exception as e:
    print(f"  Variogram 计算失败: {e}")

# 3c: 残差空间分布图
print("\n--- 3c: 残差空间分布图 ---")
fig_r = plt.figure(figsize=(18, 9))
ax_r = fig_r.add_subplot(1, 1, 1, projection=ccrs.Robinson())
ax_r.set_global()
ax_r.add_feature(cfeature.LAND, facecolor="0.85", edgecolor="none", zorder=2)
ax_r.add_feature(cfeature.COASTLINE, linewidth=0.3, zorder=3)

# 用网格级别的残差中位数绘图（同一网格多条记录取中位数）
te2_copy = te2.copy()
te2_copy["residual"] = residuals_spatial
grid_res = te2_copy.groupby(["grid_lat", "grid_lon"])["residual"].median().reset_index()

vmax_r = np.percentile(np.abs(grid_res["residual"]), 95)
norm_r = mcolors.TwoSlopeNorm(vmin=-vmax_r, vcenter=0, vmax=vmax_r)

sc_r = ax_r.scatter(
    grid_res["grid_lon"].values, grid_res["grid_lat"].values,
    c=grid_res["residual"].values,
    cmap="RdBu_r", norm=norm_r, s=1.5, marker="s", linewidths=0,
    transform=ccrs.PlateCarree(), zorder=1
)
cbar_r = plt.colorbar(sc_r, ax=ax_r, orientation="horizontal", pad=0.05, shrink=0.6, aspect=40)
cbar_r.set_label("Prediction Residual (mW/m²)", fontsize=12)
ax_r.set_title("Spatial Distribution of Prediction Residuals (Block 2°×2°, ExtraTrees)",
               fontsize=14, fontweight="bold", pad=15)
fig_r.savefig(FIG_DIR / "step7_residual_spatial.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  图已保存: {FIG_DIR / 'step7_residual_spatial.png'}")

print("\n" + "=" * 80)
print("第一批实验全部完成!")
print("=" * 80)
