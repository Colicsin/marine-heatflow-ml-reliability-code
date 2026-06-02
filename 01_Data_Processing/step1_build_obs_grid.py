"""
Step 1: 从 IHFC_2024_GHFDB.csv 构建真实观测格点数据集

过滤规则：
  - 保留 Domain == 'marine' (包含 offshore continental + offshore marine)
  - 剔除 q <= 0 或 q > 250 mW/m²
  - 聚合到 0.5° 格点，取中位数，记录 count/std

输出：
  data/processed/obs_grid_0.5deg.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH     = ROOT / "data/raw/IHFC_2024_GHFDB.csv"
NE10_PATH    = ROOT / "data/natural_earth/ne_10m_land.shp"
OUT_DIR      = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RES = 0.5
Q_MIN, Q_MAX = 0.0, 250.0

# ── 1. 读取原始数据 ───────────────────────────────────────────────
print("读取 IHFC_2024_GHFDB.csv ...")
df = pd.read_csv(RAW_PATH, encoding="latin-1",
                 usecols=["q", "lat_NS", "long_EW", "Domain", "environment"])
print(f"  原始总行数: {len(df):,}")

# ── 2. 过滤：保留 Domain == marine ───────────────────────────────
# marine 域包含了 offshore (continental) 和 offshore (marine) 两类
marine_mask = df["Domain"] == "marine"
df = df[marine_mask].copy()
print(f"  保留 Domain==marine: {len(df):,}")
print(f"    其中 offshore(continental): {df['environment'].str.contains('continental', na=False).sum():,}")
print(f"    其中 offshore(marine):      {df['environment'].str.contains('offshore.*marine', na=False).sum():,}")

# ── 3. 过滤：剔除无效坐标和异常热流值 ────────────────────────────
df = df.dropna(subset=["q", "lat_NS", "long_EW"])
df = df[(df["q"] > Q_MIN) & (df["q"] <= Q_MAX)]
df = df[(df["lat_NS"] >= -90) & (df["lat_NS"] <= 90)]
df = df[(df["long_EW"] >= -180) & (df["long_EW"] <= 180)]
print(f"  过滤异常值后: {len(df):,}")

# ── 4. 分配到 0.5° 格点 ──────────────────────────────────────────
df["grid_lat"] = (np.floor(df["lat_NS"]  / RES) * RES + RES / 2).round(2)
df["grid_lon"] = (np.floor(df["long_EW"] / RES) * RES + RES / 2).round(2)

# ── 5. 用 ne_10m 陆地掩膜过滤格点中心落在陆地的格点 ──────────────
print("加载 ne_10m 陆地掩膜 ...")
land = gpd.read_file(NE10_PATH)

unique_grids = df[["grid_lon", "grid_lat"]].drop_duplicates().copy()
print(f"  唯一格点数: {len(unique_grids):,}")

# 判断格点中心是否在陆地
geometry = [Point(row.grid_lon, row.grid_lat) for row in unique_grids.itertuples()]
grid_gdf  = gpd.GeoDataFrame(unique_grids, geometry=geometry, crs="EPSG:4326")
joined    = gpd.sjoin(grid_gdf, land[["geometry"]], how="left", predicate="within")
land_idx  = set(joined[joined["index_right"].notna()].index)
ocean_idx = set(unique_grids.index) - land_idx

ocean_grids = unique_grids.loc[list(ocean_idx)]
print(f"  格点中心在陆地（剔除）: {len(land_idx):,}")
print(f"  格点中心在海洋（保留）: {len(ocean_idx):,}")

df = df.merge(ocean_grids[["grid_lon", "grid_lat"]], on=["grid_lon", "grid_lat"], how="inner")
print(f"  过滤后观测点数: {len(df):,}")

# ── 6. 聚合到格点 ────────────────────────────────────────────────
print("聚合到 0.5° 格点 ...")
agg = df.groupby(["grid_lat", "grid_lon"]).agg(
    median_q  = ("q", "median"),
    mean_q    = ("q", "mean"),
    std_q     = ("q", "std"),
    count     = ("q", "count"),
    min_q     = ("q", "min"),
    max_q     = ("q", "max"),
).reset_index()

agg["std_q"] = agg["std_q"].fillna(0.0)  # count=1 时 std 为 NaN，填 0

print(f"\n最终格点数: {len(agg):,}")
print(f"热流值范围: {agg['median_q'].min():.1f} ~ {agg['median_q'].max():.1f} mW/m²")
print(f"均值={agg['median_q'].mean():.1f}  标准差={agg['median_q'].std():.1f}")
print(f"\n观测数分布:")
for label, lo, hi in [("1",1,2),("2",2,3),("3",3,4),("4",4,5),("5-9",5,10),("10-49",10,50),("≥50",50,99999)]:
    n = int(((agg["count"] >= lo) & (agg["count"] < hi)).sum())
    print(f"  count={label}: {n:,} ({n/len(agg)*100:.1f}%)")

# ── 7. 保存 ──────────────────────────────────────────────────────
out_path = OUT_DIR / "obs_grid_0.5deg.csv"
agg.to_csv(out_path, index=False)
print(f"\n保存至: {out_path}")
