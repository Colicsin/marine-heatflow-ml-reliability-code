"""
Step 2+3: 特征对齐 + 洋壳年龄提取

Step 2: 从 Ocean_HeatFlow_Prediction_Data_with_Age.csv 提取13个特征（去掉热流和洋壳年龄列）
        按坐标 join 到 obs_grid_0.5deg.csv

Step 3: 从 Muller nc 文件提取洋壳年龄
        对每个格点，在 ±0.25° 窗口内取中位数（对应0.5°格点范围）
        无值的格点（大陆架）保留为 NaN

输出：
  data/processed/dataset_with_features.csv
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc

ROOT      = Path(__file__).resolve().parents[1]
OBS_PATH  = ROOT / "data/processed/obs_grid_0.5deg.csv"
FEAT_PATH = ROOT / "data/features/Ocean_HeatFlow_Prediction_Data_with_Age.csv"
NC_PATH   = ROOT / "data/raw/Muller_etal_2019_Tectonics_v2.0_PresentDay_AgeGrid.nc"
OUT_DIR   = ROOT / "data/processed"

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

# ── Step 2: 特征对齐 ─────────────────────────────────────────────
print("=" * 55)
print("Step 2: 特征对齐")
obs = pd.read_csv(OBS_PATH)
print(f"  obs_grid 格点数: {len(obs):,}")

feat = pd.read_csv(FEAT_PATH, usecols=["lon", "lat"] + FEATURE_COLS)
feat["grid_lon"] = feat["lon"].round(2)
feat["grid_lat"] = feat["lat"].round(2)
feat = feat.drop(columns=["lon", "lat"])
feat = feat.drop_duplicates(subset=["grid_lon", "grid_lat"])
print(f"  特征数据行数: {len(feat):,}")

merged = obs.merge(feat, on=["grid_lon", "grid_lat"], how="left")
n_feat_matched = merged[FEATURE_COLS[0]].notna().sum()
print(f"  特征匹配成功: {n_feat_matched:,} / {len(merged):,} 格点")
n_feat_missing = len(merged) - n_feat_matched
if n_feat_missing > 0:
    print(f"  特征缺失格点: {n_feat_missing:,} (这些格点不在旧数据集覆盖范围内)")

# ── Step 3: 洋壳年龄提取 ─────────────────────────────────────────
print("\n" + "=" * 55)
print("Step 3: 洋壳年龄提取 (Muller 2019, 0.1°)")

f   = nc.Dataset(NC_PATH)
nc_lon = np.array(f.variables["lon"][:])   # (3601,) -180~180
nc_lat = np.array(f.variables["lat"][:])   # (1801,) -90~90
nc_z   = np.array(f.variables["z"][:])     # (1801, 3601) float32, NaN=大陆
f.close()

# nc_lon 步长 0.1°，预先计算索引范围更快
lon_step = float(nc_lon[1] - nc_lon[0])  # 0.1
lat_step = float(nc_lat[1] - nc_lat[0])  # 0.1
half_win = 0.25  # 0.5° 格点的半窗口

def extract_age(grid_lon, grid_lat):
    """在 ±0.25° 窗口内取洋壳年龄中位数，全NaN返回NaN"""
    lon_lo = grid_lon - half_win
    lon_hi = grid_lon + half_win
    lat_lo = grid_lat - half_win
    lat_hi = grid_lat + half_win

    # 转为索引范围
    i_lo = max(0, int(np.floor((lat_lo - nc_lat[0]) / lat_step)))
    i_hi = min(len(nc_lat)-1, int(np.ceil((lat_hi - nc_lat[0]) / lat_step)))
    j_lo = max(0, int(np.floor((lon_lo - nc_lon[0]) / lon_step)))
    j_hi = min(len(nc_lon)-1, int(np.ceil((lon_hi - nc_lon[0]) / lon_step)))

    patch = nc_z[i_lo:i_hi+1, j_lo:j_hi+1]
    valid = patch[~np.isnan(patch)]
    if len(valid) == 0:
        return np.nan
    return float(np.median(valid))

print(f"  提取 {len(merged):,} 个格点的洋壳年龄 ...")
ages = []
for idx, row in enumerate(merged.itertuples()):
    ages.append(extract_age(row.grid_lon, row.grid_lat))
    if (idx + 1) % 1000 == 0:
        print(f"    {idx+1:,} / {len(merged):,} ...")

merged["oceanic_crust_age_Ma"] = ages

n_age_valid = merged["oceanic_crust_age_Ma"].notna().sum()
n_age_nan   = merged["oceanic_crust_age_Ma"].isna().sum()
print(f"  有洋壳年龄: {n_age_valid:,} ({n_age_valid/len(merged)*100:.1f}%)")
print(f"  NaN(大陆架/无洋壳): {n_age_nan:,} ({n_age_nan/len(merged)*100:.1f}%)")
age_vals = merged["oceanic_crust_age_Ma"].dropna()
print(f"  年龄范围: {age_vals.min():.1f} ~ {age_vals.max():.1f} Ma  "
      f"中位数={age_vals.median():.1f} Ma")

# ── 汇总缺失情况 ─────────────────────────────────────────────────
print("\n" + "=" * 55)
print("各特征缺失统计:")
all_feat_cols = FEATURE_COLS + ["oceanic_crust_age_Ma"]
for col in all_feat_cols:
    n_nan = merged[col].isna().sum()
    print(f"  {col:<45} NaN={n_nan:,} ({n_nan/len(merged)*100:.1f}%)")

# 完整特征的格点数（所有特征都不为NaN）
complete_mask = merged[all_feat_cols].notna().all(axis=1)
print(f"\n所有14个特征均完整的格点: {complete_mask.sum():,} / {len(merged):,}")
print(f"至少有13个特征（允许洋壳年龄NaN）的格点: "
      f"{merged[FEATURE_COLS].notna().all(axis=1).sum():,}")

# ── 保存 ─────────────────────────────────────────────────────────
out_path = OUT_DIR / "dataset_with_features.csv"
merged.to_csv(out_path, index=False)
print(f"\n保存至: {out_path}")
print(f"列: {list(merged.columns)}")
