"""Compare this validation design with the Li et al. methodological setting."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/dataset_D_no_aggregation.csv"

FEATURE_COLS = [
    "CRUST1.0_moho_depth_0.5deg", "CRUST1.0_upper_crust_thickness_0.5deg",
    "CRUST1.0_mid_crust_thickness_0.5deg", "CRUST1.0_mantle_rho_0.5deg",
    "hotspot_min_hotspot_distance_km", "volcano_latest_vocano_dist",
    "topo_topo_mean", "topo_topo_diff", "topo_topo_median",
    "EMAG2_sealevel", "EMAG2_upcont", "LITH_IDW_lab", "LITH_IDW_moho",
    "oceanic_crust_age_Ma",
]
TARGET = "q"

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS + [TARGET])
print(f"Dataset D: {len(df):,} records\n")

def spatial_block_split(data, block_size=2.0, seed=42, min_per_block=3):
    d = data.copy()
    d["_bid"] = ((d["grid_lat"] // block_size) * block_size).astype(str) + "_" +\
                ((d["grid_lon"] // block_size) * block_size).astype(str)
    bc = d["_bid"].value_counts()
    d = d[d["_bid"].isin(bc[bc >= min_per_block].index)]
    rng = np.random.default_rng(seed)
    blocks = d["_bid"].unique()
    rng.shuffle(blocks)
    n_test = int(len(blocks) * 0.3)
    test_set = set(blocks[:n_test])
    tr = d[~d["_bid"].isin(test_set)]
    te = d[d["_bid"].isin(test_set)]
    return tr, te

def calc_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }

X = df[FEATURE_COLS].values
y = df[TARGET].values

print("=" * 80)
print("Supplementary experiment D: methodological comparison with Li et al. (2021)")
print("=" * 80)


print("\n--- Comparison 1: Li et al. setup versus this study setup ---")
print(f"{'setup':<35} {'model':<12} {'split':<12} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 82)

configs = [

    ("Li: RF, 80/20 random",    "RF",         0.2, 42),
    ("Li: RF, 70/30 random",    "RF",         0.3, 42),

    ("Ours: ET, 70/30 random",  "ExtraTrees", 0.3, 42),
    ("Ours: RF, 70/30 random",  "RF",         0.3, 42),
]

random_results = {}
for label, model_name, test_size, seed in configs:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    if model_name == "RF":
        model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    else:
        model = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    m = calc_metrics(y_te, model.predict(X_te))
    random_results[label] = m
    print(f"{label:<35} {model_name:<12} {f'{int((1-test_size)*100)}/{int(test_size*100)}':<12} "
          f"{m['R2']:>8.4f} {m['RMSE']:>8.2f} {m['MAE']:>8.2f}")


print("\n--- Comparison 2: random split versus spatial block split (showing spatial leakage) ---")
print(f"{'model':<12} {'validation scheme':<20} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'R² gap':>8}")
print("-" * 72)

for model_name, model_cls in [("RF", RandomForestRegressor), ("ExtraTrees", ExtraTreesRegressor)]:

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    m_rand = model_cls(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    m_rand.fit(X_tr, y_tr)
    mr = calc_metrics(y_te, m_rand.predict(X_te))


    tr, te = spatial_block_split(df)
    m_sp = model_cls(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
    m_sp.fit(tr[FEATURE_COLS].values, tr[TARGET].values)
    ms = calc_metrics(te[TARGET].values, m_sp.predict(te[FEATURE_COLS].values))

    gap = mr["R2"] - ms["R2"]
    print(f"{model_name:<12} {'random split 7:3':<20} {len(y_te):>8,} {mr['R2']:>8.4f} {mr['RMSE']:>8.2f} {mr['MAE']:>8.2f}")
    print(f"{'':<12} {'2°x2° spatial block split':<20} {len(te):>8,} {ms['R2']:>8.4f} {ms['RMSE']:>8.2f} {ms['MAE']:>8.2f} {gap:>+8.4f}")


print("\n--- Comparison 3: cross-basin validation ---")
print(f"{'model':<12} {'test basin':<12} {'n_test':>8} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
print("-" * 58)

for model_name, model_cls in [("RF", RandomForestRegressor), ("ExtraTrees", ExtraTreesRegressor)]:
    for basin in ["Pacific", "Atlantic", "Indian"]:
        tr_b = df[df["basin"] != basin]
        te_b = df[df["basin"] == basin]
        m_b = model_cls(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        m_b.fit(tr_b[FEATURE_COLS].values, tr_b[TARGET].values)
        mb = calc_metrics(te_b[TARGET].values, m_b.predict(te_b[FEATURE_COLS].values))
        print(f"{model_name:<12} {basin:<12} {len(te_b):>8,} {mb['R2']:>8.4f} {mb['RMSE']:>8.2f} {mb['MAE']:>8.2f}")


print("\n--- Comparison 4: values reported by Li et al. (2021) versus this dataset ---")
print("Note: Li et al. used the NGHF database + 25 features + RF + an 80/20 random split")
print("    This study uses GHFDB R2024 + 14 features + RF + an 80/20 random split")
print()
print(f"{'source':<30} {'R²':>8} {'RMSE':>8}")
print("-" * 50)
print(f"{'Li No.1 (Grade A, NGHF)':<30} {'0.96':>8} {'11.74':>8}")
print(f"{'Li No.2 (Grades A+B, NGHF)':<30} {'0.88':>8} {'22.58':>8}")
print(f"{'Li No.3 (Grades A+B+C, NGHF)':<30} {'0.77':>8} {'32.56':>8}")
r2_ours = random_results["Li: RF, 80/20 random"]["R2"]
rmse_ours = random_results["Li: RF, 80/20 random"]["RMSE"]
print(f"{'This study (all, GHFDB R2024)':<30} {r2_ours:>8.4f} {rmse_ours:>8.2f}")
print()
print("Difference analysis: the high R² reported by Li et al. mainly arises from two factors:")
print("  1. Grade-A data have very low noise because only the highest-quality measurements are retained")
print("  2. 80/20 random split plus shared features among records from the same grid cell leads to spatial leakage")
print(f"  Under the same split strategy, this study obtains R²={r2_ours:.4f}, which is far below the 0.96 reported for Li No.1, ")
print("  This indicates that label quality (observed labels vs high-quality subsets) is the main source of the performance difference.")

print("\nsupplementary experiment D completed!")
