"""
新数据实验：不聚合热流 + 质量分级 + 6模型 × 3验证方案

数据源: IHFC_2024_GHFDB_v.2026.03.txt (新版GHFDB)
策略: 保留每条原始热流记录，特征取所在0.5°网格的值
质量分级: 按 Quality_Score_Parent 的 M 等级构建 A/B/C/D 四个数据集
"""

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import netCDF4 as nc
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestRegressor,
                              GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH  = ROOT / "data/raw/IHFC_2024_GHFDB_v.2026.03.txt"
FEAT_PATH = ROOT / "data/features/Ocean_HeatFlow_Prediction_Data_with_Age.csv"
NC_PATH   = ROOT / "data/raw/Muller_etal_2019_Tectonics_v2.0_PresentDay_AgeGrid.nc"
NE10_PATH = ROOT / "data/natural_earth/ne_10m_land.shp"

RES = 0.5
Q_MIN, Q_MAX = 0.0, 250.0

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
]
TARGET = "q"

# ═══════════════════════════════════════════════════════════════════
# Step A: 读取新数据 + 基础过滤
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("Step A: 读取新数据 + 基础过滤")
print("=" * 70)

df = pd.read_csv(RAW_PATH, sep="\t", skiprows=12, encoding="latin-1",
                 low_memory=False)
print(f"  原始总行数: {len(df):,}")

# 筛选海洋记录
marine_env = ["[offshore (continental)]", "[offshore (marine)]"]
df = df[df["environment"].isin(marine_env)].copy()
print(f"  保留 offshore: {len(df):,}")

# 数值转换 + 过滤异常值
df["q"] = pd.to_numeric(df["q"], errors="coerce")
df["lat_NS"] = pd.to_numeric(df["lat_NS"], errors="coerce")
df["long_EW"] = pd.to_numeric(df["long_EW"], errors="coerce")
df = df.dropna(subset=["q", "lat_NS", "long_EW"])
df = df[(df["q"] > Q_MIN) & (df["q"] <= Q_MAX)]
df = df[(df["lat_NS"] >= -90) & (df["lat_NS"] <= 90)]
df = df[(df["long_EW"] >= -180) & (df["long_EW"] <= 180)]
print(f"  过滤异常值后: {len(df):,}")

# 解析质量等级
df["qc_u"] = df["Quality_Score_Parent"].astype(str).str.extract(r"^(U[0-9x])")[0]
df["qc_m"] = df["Quality_Score_Parent"].astype(str).str.extract(r"\.(M[0-9x]+)\.")[0]
print(f"  M等级分布:")
for m in ["M1", "M1x", "M2", "M2x", "M3", "M3x", "M4", "M4x", "Mx"]:
    n = (df["qc_m"] == m).sum()
    if n > 0:
        print(f"    {m}: {n:,}")

# ═══════════════════════════════════════════════════════════════════
# Step B: 分配网格 + 匹配特征（不聚合热流）
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("Step B: 分配网格 + 匹配特征")
print("=" * 70)

# 分配 0.5° 网格
df["grid_lat"] = (np.floor(df["lat_NS"] / RES) * RES + RES / 2).round(2)
df["grid_lon"] = (np.floor(df["long_EW"] / RES) * RES + RES / 2).round(2)

# 陆地掩膜
print("  加载陆地掩膜...")
land = gpd.read_file(NE10_PATH)
unique_grids = df[["grid_lat", "grid_lon"]].drop_duplicates()
geometry = [Point(row.grid_lon, row.grid_lat) for row in unique_grids.itertuples()]
grid_gdf = gpd.GeoDataFrame(unique_grids, geometry=geometry, crs="EPSG:4326")
joined = gpd.sjoin(grid_gdf, land[["geometry"]], how="left", predicate="within")
land_grids = set(zip(
    joined[joined["index_right"].notna()]["grid_lat"],
    joined[joined["index_right"].notna()]["grid_lon"]
))
before = len(df)
df["_grid_key"] = list(zip(df["grid_lat"], df["grid_lon"]))
df = df[~df["_grid_key"].isin(land_grids)].drop(columns=["_grid_key"])
print(f"  陆地掩膜: {before:,} -> {len(df):,} (去除 {before - len(df):,})")

# 匹配 13 个特征
print("  匹配特征...")
feat = pd.read_csv(FEAT_PATH, usecols=["lon", "lat"] + FEATURE_COLS)
feat["grid_lon"] = feat["lon"].round(2)
feat["grid_lat"] = feat["lat"].round(2)
feat = feat.drop(columns=["lon", "lat"])
feat = feat.drop_duplicates(subset=["grid_lon", "grid_lat"])

df = df.merge(feat, on=["grid_lon", "grid_lat"], how="left")
n_matched = df[FEATURE_COLS[0]].notna().sum()
print(f"  特征匹配: {n_matched:,} / {len(df):,}")

# 提取洋壳年龄
print("  提取洋壳年龄...")
ds = nc.Dataset(str(NC_PATH))
nc_lon = ds.variables["lon"][:]
nc_lat = ds.variables["lat"][:]
nc_age = ds.variables["z"][:]

unique_grids2 = df[["grid_lat", "grid_lon"]].drop_duplicates()
age_map = {}
half = RES / 2
for _, row in unique_grids2.iterrows():
    glat, glon = row["grid_lat"], row["grid_lon"]
    lat_mask = (nc_lat >= glat - half) & (nc_lat <= glat + half)
    lon_mask = (nc_lon >= glon - half) & (nc_lon <= glon + half)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    if len(lat_idx) > 0 and len(lon_idx) > 0:
        patch = nc_age[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        vals = patch.compressed() if hasattr(patch, "compressed") else patch[~np.isnan(patch)]
        if len(vals) > 0:
            age_map[(glat, glon)] = float(np.median(vals))
ds.close()

df["oceanic_crust_age_Ma"] = df.apply(
    lambda r: age_map.get((r["grid_lat"], r["grid_lon"]), np.nan), axis=1
)
n_age = df["oceanic_crust_age_Ma"].notna().sum()
print(f"  洋壳年龄匹配: {n_age:,} / {len(df):,}")

# 丢弃 13 个基础特征缺失的记录
ALL_FEAT = FEATURE_COLS + ["oceanic_crust_age_Ma"]
before = len(df)
df = df.dropna(subset=FEATURE_COLS).copy()
df["oceanic_crust_age_Ma"] = df["oceanic_crust_age_Ma"].fillna(-1.0)
print(f"  丢弃特征缺失: {before:,} -> {len(df):,}")

# 洋盆标签
def assign_basin(lon, lat):
    if lat < -60:
        return "Southern"
    if lon < -20 or lon > 140:
        return "Pacific"
    if -20 <= lon <= 20:
        return "Atlantic"
    if 20 < lon <= 140 and lat > 0:
        return "Indian"
    return "Indian"

df["basin"] = df.apply(lambda r: assign_basin(r["grid_lon"], r["grid_lat"]), axis=1)

print(f"\n  最终数据集: {len(df):,} 条记录, {df[['grid_lat','grid_lon']].drop_duplicates().shape[0]:,} 个网格")
print(f"  洋盆分布: {dict(df['basin'].value_counts())}")

# ═══════════════════════════════════════════════════════════════════
# Step C: 构建质量分级数据集
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("Step C: 质量分级数据集")
print("=" * 70)

DATASETS = {
    "A: M1+M1x":           ["M1", "M1x"],
    "B: M1-M2(含x)":       ["M1", "M1x", "M2", "M2x"],
    "C: M1-M3(含x)":       ["M1", "M1x", "M2", "M2x", "M3", "M3x"],
    "ALL: 排除Mx":         ["M1", "M1x", "M2", "M2x", "M3", "M3x", "M4", "M4x"],
    "D: 全部(含Mx)":       None,  # 不筛选
}

for label, m_levels in DATASETS.items():
    if m_levels is None:
        sub = df
    else:
        sub = df[df["qc_m"].isin(m_levels)]
    n_grids = sub[["grid_lat", "grid_lon"]].drop_duplicates().shape[0]
    print(f"  {label:<20} 记录={len(sub):>6,}  网格={n_grids:>5,}  "
          f"q均值={sub['q'].mean():.1f}  q中位数={sub['q'].median():.1f}  q标准差={sub['q'].std():.1f}")

# ═══════════════════════════════════════════════════════════════════
# Step D: 6模型 × 3验证方案 × 质量分级
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("Step D: 模型训练与验证")
print("=" * 70)

MODELS = {
    "LinearReg":   lambda: LinearRegression(),
    "RF":          lambda: RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    "ExtraTrees":  lambda: ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    "GBDT":        lambda: GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42),
    "XGBoost":     lambda: XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0, n_jobs=-1),
    "LightGBM":    lambda: LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1, n_jobs=-1),
}

def calc_metrics(y_true, y_pred):
    return (r2_score(y_true, y_pred),
            float(np.sqrt(mean_squared_error(y_true, y_pred))),
            float(mean_absolute_error(y_true, y_pred)),
            float(np.mean(y_pred - y_true)))

results = []

for ds_label, m_levels in DATASETS.items():
    if m_levels is None:
        sub = df.copy()
    else:
        sub = df[df["qc_m"].isin(m_levels)].copy()

    if len(sub) < 50:
        print(f"\n  [{ds_label}] 样本不足({len(sub)}), 跳过")
        continue

    X = sub[ALL_FEAT].values
    y = sub[TARGET].values

    print(f"\n{'='*70}")
    print(f"数据集: {ds_label}  ({len(sub):,} 条记录)")
    print(f"{'='*70}")

    # ── 方案1: 随机划分 7:3 ──
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"\n  方案1: 随机划分 (train={len(y_tr):,}, test={len(y_te):,})")
    print(f"  {'模型':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8}")
    print("  " + "-" * 48)
    for mname, mfunc in MODELS.items():
        model = mfunc()
        model.fit(X_tr, y_tr)
        r2, rmse, mae, bias = calc_metrics(y_te, model.predict(X_te))
        print(f"  {mname:<12} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f} {bias:>8.2f}")
        results.append((ds_label, "随机划分", mname, len(sub), len(y_te), r2, rmse, mae, bias))

    # ── 方案2: 空间分组 2°×2° (同网格记录不分离) ──
    sub2 = sub.copy()
    sub2["block_id"] = ((sub2["grid_lat"] // 2) * 2).astype(str) + "_" + \
                       ((sub2["grid_lon"] // 2) * 2).astype(str)
    block_counts = sub2["block_id"].value_counts()
    valid_blocks = block_counts[block_counts >= 3].index
    sub2 = sub2[sub2["block_id"].isin(valid_blocks)]

    if len(sub2) > 50:
        rng = np.random.default_rng(42)
        blocks = sub2["block_id"].unique()
        rng.shuffle(blocks)
        test_blocks = set(blocks[:int(len(blocks) * 0.3)])
        tr = sub2[~sub2["block_id"].isin(test_blocks)]
        te = sub2[sub2["block_id"].isin(test_blocks)]

        print(f"\n  方案2: 空间分组 2°×2° (train={len(tr):,}, test={len(te):,}, "
              f"blocks={len(blocks)})")
        print(f"  {'模型':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8}")
        print("  " + "-" * 48)
        for mname, mfunc in MODELS.items():
            model = mfunc()
            model.fit(tr[ALL_FEAT].values, tr[TARGET].values)
            r2, rmse, mae, bias = calc_metrics(te[TARGET].values,
                                               model.predict(te[ALL_FEAT].values))
            print(f"  {mname:<12} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f} {bias:>8.2f}")
            results.append((ds_label, "空间分组", mname, len(sub), len(te), r2, rmse, mae, bias))
    else:
        print(f"\n  方案2: 空间分组 - 样本不足, 跳过")

    # ── 方案3: 跨洋盆验证 (ExtraTrees) ──
    print(f"\n  方案3: 跨洋盆验证 (ExtraTrees)")
    print(f"  {'洋盆':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8}")
    print("  " + "-" * 48)
    for basin in ["Pacific", "Atlantic", "Indian"]:
        tr_b = sub[sub["basin"] != basin]
        te_b = sub[sub["basin"] == basin]
        if len(te_b) > 10 and len(tr_b) > 10:
            et = ExtraTreesRegressor(n_estimators=100, max_depth=20,
                                     random_state=42, n_jobs=-1)
            et.fit(tr_b[ALL_FEAT].values, tr_b[TARGET].values)
            r2, rmse, mae, bias = calc_metrics(te_b[TARGET].values,
                                               et.predict(te_b[ALL_FEAT].values))
            print(f"  {basin:<12} {r2:>8.4f} {rmse:>8.2f} {mae:>8.2f} {bias:>8.2f}")
            results.append((ds_label, f"跨盆-{basin}", "ExtraTrees", len(sub), len(te_b), r2, rmse, mae, bias))
        else:
            print(f"  {basin:<12} 样本不足")

# ═══════════════════════════════════════════════════════════════════
# Step E: 汇总表
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 90)
print("汇总: ExtraTrees 在各数据集 × 各验证方案下的 R²")
print("=" * 90)
print(f"{'数据集':<20} {'验证方案':<16} {'n_total':>8} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 82)
for r in results:
    if r[2] == "ExtraTrees":
        print(f"{r[0]:<20} {r[1]:<16} {r[3]:>8,} {r[4]:>8,} {r[5]:>8.4f} {r[6]:>8.2f} {r[7]:>8.2f}")
