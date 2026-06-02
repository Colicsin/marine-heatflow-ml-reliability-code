"""
实验4：SHAP 特征重要性 + 交互分析
对应论文表9、表10、图6
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import shap
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/dataset_D_no_aggregation.csv"
FIG_DIR = ROOT / "outputs/figures"
OUT_DIR = ROOT / "outputs"

FEATURE_COLS = [
    "CRUST1.0_moho_depth_0.5deg", "CRUST1.0_upper_crust_thickness_0.5deg",
    "CRUST1.0_mid_crust_thickness_0.5deg", "CRUST1.0_mantle_rho_0.5deg",
    "hotspot_min_hotspot_distance_km", "volcano_latest_vocano_dist",
    "topo_topo_mean", "topo_topo_diff", "topo_topo_median",
    "EMAG2_sealevel", "EMAG2_upcont", "LITH_IDW_lab", "LITH_IDW_moho",
    "oceanic_crust_age_Ma",
]
SHORT_NAMES = [
    "moho_depth", "upper_crust", "mid_crust", "mantle_rho",
    "hotspot_dist", "volcano_dist", "topo_mean", "topo_diff",
    "topo_median", "EMAG2_sea", "EMAG2_up", "LITH_lab", "LITH_moho",
    "crust_age",
]
TARGET = "q"

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS + [TARGET])
X = df[FEATURE_COLS].values
y = df[TARGET].values

print("训练 ExtraTrees（全量数据）...")
model = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X, y)

# SHAP 主效应（采样3000）
print("计算 SHAP 值（采样3000）...")
rng = np.random.default_rng(42)
idx_shap = rng.choice(len(X), 3000, replace=False)
X_shap = X[idx_shap]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# SHAP 重要性 Top
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
importance_order = np.argsort(mean_abs_shap)[::-1]

print("\n" + "=" * 60)
print("SHAP 特征重要性排序 —— 对应论文表9")
print("=" * 60)
print(f"{'排名':<4} {'特征':<30} {'平均|SHAP|':>12}")
print("-" * 50)
for rank, i in enumerate(importance_order):
    print(f"{rank+1:<4} {SHORT_NAMES[i]:<30} {mean_abs_shap[i]:>12.3f}")

# SHAP 交互值（采样500）
print("\n计算 SHAP 交互值（采样500）...")
idx_inter = rng.choice(len(X), 500, replace=False)
X_inter = X[idx_inter]
shap_interaction = explainer.shap_interaction_values(X_inter)

# 交互强度矩阵
inter_matrix = np.mean(np.abs(shap_interaction), axis=0)
# 去掉对角线（主效应）
np.fill_diagonal(inter_matrix, 0)

# Top 5 交互对
print("\n" + "=" * 60)
print("SHAP 交互对 Top 5 —— 对应论文表10")
print("=" * 60)
pairs = []
for i in range(len(FEATURE_COLS)):
    for j in range(i+1, len(FEATURE_COLS)):
        pairs.append((i, j, inter_matrix[i, j]))
pairs.sort(key=lambda x: x[2], reverse=True)

print(f"{'排名':<4} {'特征对':<45} {'交互强度':>10}")
print("-" * 62)
for rank, (i, j, val) in enumerate(pairs[:5]):
    print(f"{rank+1:<4} {SHORT_NAMES[i]} × {SHORT_NAMES[j]:<30} {val:>10.3f}")

# 绘制交互强度矩阵热力图 —— 对应论文图6
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(inter_matrix, cmap="YlOrRd", aspect="equal")
ax.set_xticks(range(len(SHORT_NAMES)))
ax.set_yticks(range(len(SHORT_NAMES)))
ax.set_xticklabels(SHORT_NAMES, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(SHORT_NAMES, fontsize=8)
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Mean |SHAP interaction|", fontsize=11)
ax.set_title("SHAP Feature Interaction Matrix (ExtraTrees, Dataset D)", fontsize=13)
plt.tight_layout()
fig.savefig(FIG_DIR / "step7_shap_interaction_matrix.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"\n图已保存: {FIG_DIR / 'step7_shap_interaction_matrix.png'}")

# SHAP summary plot
fig2, ax2 = plt.subplots(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, feature_names=SHORT_NAMES, show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "step7_shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"图已保存: {FIG_DIR / 'step7_shap_summary.png'}")

print("\nSHAP 分析完成!")
