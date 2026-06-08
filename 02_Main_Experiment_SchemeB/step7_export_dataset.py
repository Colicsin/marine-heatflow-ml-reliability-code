"""Export the complete Dataset D used by the main experiments."""
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import netCDF4 as nc

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH  = ROOT / "data/raw/IHFC_2024_GHFDB_v.2026.03.txt"
FEAT_PATH = ROOT / "data/features/Ocean_HeatFlow_Prediction_Data_with_Age.csv"
NC_PATH   = ROOT / "data/raw/Muller_etal_2019_Tectonics_v2.0_PresentDay_AgeGrid.nc"
NE10_PATH = ROOT / "data/natural_earth/ne_10m_land.shp"
OUT_PATH  = ROOT / "data/processed/dataset_D_no_aggregation.csv"

RES = 0.5
FEATURE_COLS = [
    "CRUST1.0_moho_depth_0.5deg", "CRUST1.0_upper_crust_thickness_0.5deg",
    "CRUST1.0_mid_crust_thickness_0.5deg", "CRUST1.0_mantle_rho_0.5deg",
    "hotspot_min_hotspot_distance_km", "volcano_latest_vocano_dist",
    "topo_topo_mean", "topo_topo_diff", "topo_topo_median",
    "EMAG2_sealevel", "EMAG2_upcont", "LITH_IDW_lab", "LITH_IDW_moho",
]


df = pd.read_csv(RAW_PATH, sep="\t", skiprows=12, encoding="latin-1", low_memory=False)
df = df[df["environment"].isin(["[offshore (continental)]", "[offshore (marine)]"])].copy()
df["q"] = pd.to_numeric(df["q"], errors="coerce")
df["lat_NS"] = pd.to_numeric(df["lat_NS"], errors="coerce")
df["long_EW"] = pd.to_numeric(df["long_EW"], errors="coerce")
df = df.dropna(subset=["q", "lat_NS", "long_EW"])
df = df[(df["q"] > 0) & (df["q"] <= 250)]
df = df[(df["lat_NS"] >= -90) & (df["lat_NS"] <= 90)]
df = df[(df["long_EW"] >= -180) & (df["long_EW"] <= 180)]


df["qc_u"] = df["Quality_Score_Parent"].astype(str).str.extract(r"^(U[0-9x])")[0]
df["qc_m"] = df["Quality_Score_Parent"].astype(str).str.extract(r"\.(M[0-9x]+)\.")[0]


df["grid_lat"] = (np.floor(df["lat_NS"] / RES) * RES + RES / 2).round(2)
df["grid_lon"] = (np.floor(df["long_EW"] / RES) * RES + RES / 2).round(2)


land = gpd.read_file(NE10_PATH)
ug = df[["grid_lat", "grid_lon"]].drop_duplicates()
geom = [Point(r.grid_lon, r.grid_lat) for r in ug.itertuples()]
gg = gpd.GeoDataFrame(ug, geometry=geom, crs="EPSG:4326")
jn = gpd.sjoin(gg, land[["geometry"]], how="left", predicate="within")
land_set = set(zip(jn[jn["index_right"].notna()]["grid_lat"], jn[jn["index_right"].notna()]["grid_lon"]))
df["_gk"] = list(zip(df["grid_lat"], df["grid_lon"]))
df = df[~df["_gk"].isin(land_set)].drop(columns=["_gk"])


feat = pd.read_csv(FEAT_PATH, usecols=["lon", "lat"] + FEATURE_COLS)
feat["grid_lon"] = feat["lon"].round(2)
feat["grid_lat"] = feat["lat"].round(2)
feat = feat.drop(columns=["lon", "lat"]).drop_duplicates(subset=["grid_lon", "grid_lat"])
df = df.merge(feat, on=["grid_lon", "grid_lat"], how="left")


ds = nc.Dataset(str(NC_PATH))
nc_lon, nc_lat, nc_age = ds.variables["lon"][:], ds.variables["lat"][:], ds.variables["z"][:]
ug2 = df[["grid_lat", "grid_lon"]].drop_duplicates()
age_map = {}
half = RES / 2
for _, row in ug2.iterrows():
    glat, glon = row["grid_lat"], row["grid_lon"]
    li = np.where((nc_lat >= glat - half) & (nc_lat <= glat + half))[0]
    lo = np.where((nc_lon >= glon - half) & (nc_lon <= glon + half))[0]
    if len(li) > 0 and len(lo) > 0:
        patch = nc_age[li[0]:li[-1]+1, lo[0]:lo[-1]+1]
        vals = patch.compressed() if hasattr(patch, "compressed") else patch[~np.isnan(patch)]
        if len(vals) > 0:
            age_map[(glat, glon)] = float(np.median(vals))
ds.close()
df["oceanic_crust_age_Ma"] = df.apply(lambda r: age_map.get((r["grid_lat"], r["grid_lon"]), np.nan), axis=1)


df = df.dropna(subset=FEATURE_COLS).copy()
df["oceanic_crust_age_Ma"] = df["oceanic_crust_age_Ma"].fillna(-1.0)


def assign_basin(lon, lat):
    if lat < -60: return "Southern"
    if lon < -20 or lon > 140: return "Pacific"
    if -20 <= lon <= 20: return "Atlantic"
    return "Indian"
df["basin"] = df.apply(lambda r: assign_basin(r["grid_lon"], r["grid_lat"]), axis=1)


out_cols = ["q", "lat_NS", "long_EW", "grid_lat", "grid_lon"] + FEATURE_COLS + ["oceanic_crust_age_Ma", "basin", "qc_u", "qc_m"]
df[out_cols].to_csv(OUT_PATH, index=False)
print(f"导出完成: {OUT_PATH}")
print(f"  记录数: {len(df):,}, 网格数: {df[['grid_lat','grid_lon']].drop_duplicates().shape[0]:,}")
