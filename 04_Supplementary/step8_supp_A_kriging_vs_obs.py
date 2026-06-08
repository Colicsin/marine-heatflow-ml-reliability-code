"""Compare Kriging-derived labels with observed labels in supplementary validation tests."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
FEAT_PATH = ROOT / "data/features/Ocean_HeatFlow_Prediction_Data_with_Age.csv"
OBS_PATH  = ROOT / "data/processed/dataset_D_no_aggregation.csv"

FEATURE_COLS = [
    "CRUST1.0_moho_depth_0.5deg", "CRUST1.0_upper_crust_thickness_0.5deg",
    "CRUST1.0_mid_crust_thickness_0.5deg", "CRUST1.0_mantle_rho_0.5deg",
    "hotspot_min_hotspot_distance_km", "volcano_latest_vocano_dist",
    "topo_topo_mean", "topo_topo_diff", "topo_topo_median",
    "EMAG2_sealevel", "EMAG2_upcont", "LITH_IDW_lab", "LITH_IDW_moho",
    "oceanic_crust_age_Ma",
]

def spatial_block_split(data, lat_col, lon_col, block_size=2.0,
                        test_ratio=0.3, seed=42, min_per_block=3):
    d = data.copy()
    d["_bid"] = ((d[lat_col] // block_size) * block_size).astype(str) + "_" +\
                ((d[lon_col] // block_size) * block_size).astype(str)
    bc = d["_bid"].value_counts()
    d = d[d["_bid"].isin(bc[bc >= min_per_block].index)]
    rng = np.random.default_rng(seed)
    blocks = d["_bid"].unique()
    rng.shuffle(blocks)
    n_test = int(len(blocks) * test_ratio)
    test_set = set(blocks[:n_test])
    tr = d[~d["_bid"].isin(test_set)].drop(columns=["_bid"])
    te = d[d["_bid"].isin(test_set)].drop(columns=["_bid"])
    return tr, te, len(blocks)

def calc_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }

def assign_basin(lon, lat):
    if lat < -60: return "Southern"
    if lon < -20 or lon > 140: return "Pacific"
    if -20 <= lon <= 20: return "Atlantic"
    return "Indian"


print("=" * 80)
print("Supplementary experiment A: Kriging-derived labels vs observed labels")
print("=" * 80)

kriging = pd.read_csv(FEAT_PATH)
kriging = kriging.rename(columns={"heatflow_heatflow-value": "q",
                                   "lon": "grid_lon", "lat": "grid_lat"})
kriging["grid_lon"] = kriging["grid_lon"].round(2)
kriging["grid_lat"] = kriging["grid_lat"].round(2)
kriging["oceanic_crust_age_Ma"] = kriging["oceanic_crust_age_Ma"].fillna(-1.0)
kriging = kriging.dropna(subset=FEATURE_COLS + ["q"])
kriging["basin"] = kriging.apply(
    lambda r: assign_basin(r["grid_lon"], r["grid_lat"]), axis=1)
print(f"\nKriging-label dataset: {len(kriging):,} grid cells")
print(f"  q: mean={kriging['q'].mean():.1f}, std={kriging['q'].std():.1f}")


obs = pd.read_csv(OBS_PATH)
obs = obs.dropna(subset=FEATURE_COLS + ["q"])
print(f"\nobserved dataset: {len(obs):,} records, "
      f"{obs[['grid_lat','grid_lon']].drop_duplicates().shape[0]:,} grid cells")
print(f"  q: mean={obs['q'].mean():.1f}, std={obs['q'].std():.1f}")


datasets = {
    "Kriging labels (123k grid cells)": (kriging, FEATURE_COLS, "q", "grid_lat", "grid_lon"),
    "Observed values (28k records)":     (obs,     FEATURE_COLS, "q", "grid_lat", "grid_lon"),
}

results = []

for ds_name, (data, feat_cols, target, lat_col, lon_col) in datasets.items():
    X = data[feat_cols].values
    y = data[target].values

    print(f"\n{'='*70}")
    print(f"dataset: {ds_name}")
    print(f"{'='*70}")


    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    et = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et.fit(X_tr, y_tr)
    m = calc_metrics(y_te, et.predict(X_te))
    results.append((ds_name, "random split", len(y_te), m))
    print(f"  random split:     n_test={len(y_te):>7,}  R²={m['R2']:.4f}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}")


    tr, te, n_blocks = spatial_block_split(data, lat_col, lon_col)
    et2 = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    et2.fit(tr[feat_cols].values, tr[target].values)
    m2 = calc_metrics(te[target].values, et2.predict(te[feat_cols].values))
    results.append((ds_name, "2°x2° spatial block split", len(te), m2))
    print(f"  2°x2° spatial block split: n_test={len(te):>7,}  R²={m2['R2']:.4f}  RMSE={m2['RMSE']:.2f}  MAE={m2['MAE']:.2f}")


    for basin in ["Pacific", "Atlantic", "Indian"]:
        tr_b = data[data["basin"] != basin]
        te_b = data[data["basin"] == basin]
        if len(te_b) > 10 and len(tr_b) > 10:
            et3 = ExtraTreesRegressor(n_estimators=100, max_depth=20,
                                       random_state=42, n_jobs=-1)
            et3.fit(tr_b[feat_cols].values, tr_b[target].values)
            m3 = calc_metrics(te_b[target].values, et3.predict(te_b[feat_cols].values))
            results.append((ds_name, f"cross-basin-{basin}", len(te_b), m3))
            print(f"  cross-basin-{basin:<9} n_test={len(te_b):>7,}  R²={m3['R2']:.4f}  RMSE={m3['RMSE']:.2f}  MAE={m3['MAE']:.2f}")


print("\n" + "=" * 90)
print("Summary comparison table: Kriging labels vs observed labels")
print("=" * 90)
print(f"{'dataset':<22} {'validation scheme':<16} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'Bias':>8}")
print("-" * 82)
for ds_name, scheme, n, m in results:
    tag = "K" if "Kriging" in ds_name else "O"
    print(f"[{tag}] {scheme:<18} {n:>8,} {m['R2']:>8.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f} {m['Bias']:>8.2f}")


print("\n--- R² difference (Kriging - observed) ---")
kriging_results = {r[1]: r[3] for r in results if "Kriging" in r[0]}
obs_results = {r[1]: r[3] for r in results if "Observed" in r[0]}
for scheme in kriging_results:
    if scheme in obs_results:
        diff = kriging_results[scheme]["R2"] - obs_results[scheme]["R2"]
        print(f"  {scheme:<18} Kriging R²={kriging_results[scheme]['R2']:.4f}  "
              f"observed R²={obs_results[scheme]['R2']:.4f}  difference={diff:+.4f}")

print("\nsupplementary experiment A completed!")
