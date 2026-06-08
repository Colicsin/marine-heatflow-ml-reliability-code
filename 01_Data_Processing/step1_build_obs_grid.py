"""Build a 0.5-degree gridded marine heat-flow observation dataset from GHFDB records."""

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


print("读取 IHFC_2024_GHFDB.csv ...")
df = pd.read_csv(RAW_PATH, encoding="latin-1",
                 usecols=["q", "lat_NS", "long_EW", "Domain", "environment"])
print(f"  原始总行数: {len(df):,}")


marine_mask = df["Domain"] == "marine"
df = df[marine_mask].copy()
print(f"  保留 Domain==marine: {len(df):,}")
print(f"    其中 offshore(continental): {df['environment'].str.contains('continental', na=False).sum():,}")
print(f"    其中 offshore(marine):      {df['environment'].str.contains('offshore.*marine', na=False).sum():,}")


df = df.dropna(subset=["q", "lat_NS", "long_EW"])
df = df[(df["q"] > Q_MIN) & (df["q"] <= Q_MAX)]
df = df[(df["lat_NS"] >= -90) & (df["lat_NS"] <= 90)]
df = df[(df["long_EW"] >= -180) & (df["long_EW"] <= 180)]
print(f"  过滤异常值后: {len(df):,}")


df["grid_lat"] = (np.floor(df["lat_NS"]  / RES) * RES + RES / 2).round(2)
df["grid_lon"] = (np.floor(df["long_EW"] / RES) * RES + RES / 2).round(2)


print("加载 ne_10m 陆地掩膜 ...")
land = gpd.read_file(NE10_PATH)

unique_grids = df[["grid_lon", "grid_lat"]].drop_duplicates().copy()
print(f"  唯一格点数: {len(unique_grids):,}")


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


print("聚合到 0.5° 格点 ...")
agg = df.groupby(["grid_lat", "grid_lon"]).agg(
    median_q  = ("q", "median"),
    mean_q    = ("q", "mean"),
    std_q     = ("q", "std"),
    count     = ("q", "count"),
    min_q     = ("q", "min"),
    max_q     = ("q", "max"),
).reset_index()

agg["std_q"] = agg["std_q"].fillna(0.0)

print(f"\n最终格点数: {len(agg):,}")
print(f"热流值范围: {agg['median_q'].min():.1f} ~ {agg['median_q'].max():.1f} mW/m²")
print(f"均值={agg['median_q'].mean():.1f}  标准差={agg['median_q'].std():.1f}")
print(f"\n观测数分布:")
for label, lo, hi in [("1",1,2),("2",2,3),("3",3,4),("4",4,5),("5-9",5,10),("10-49",10,50),("≥50",50,99999)]:
    n = int(((agg["count"] >= lo) & (agg["count"] < hi)).sum())
    print(f"  count={label}: {n:,} ({n/len(agg)*100:.1f}%)")


out_path = OUT_DIR / "obs_grid_0.5deg.csv"
agg.to_csv(out_path, index=False)
print(f"\n保存至: {out_path}")
