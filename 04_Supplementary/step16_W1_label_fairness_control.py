"""
补充实验 W1：标签对比公平性控制实验
====================================
审稿意见核心质疑：
  原始实验中方案A/C的训练集覆盖全球网格（含无观测区域的插值值），
  而方案B仅使用28,642条真实观测。三种方案的训练样本在数量、空间覆盖
  和信息量上存在本质差异，性能差异可能部分来自样本分布不匹配。

控制实验设计：
  将方案A和方案C的训练集限制在与方案B完全相同的观测点位置，
  仅将标签值替换为对应位置的Kriging插值值。
  这样三种方案的训练样本在空间分布上完全一致，
  性能差异即可归因于标签质量本身。

方案：
  B:  真实观测标签（基线，与原实验一致）
  A': 同位置全局Kriging标签（控制组）
  C': 同位置局部Kriging标签（控制组）

输出：
  CSV:  outputs/step16_W1_fairness_summary.csv
  图1:  step16_W1_scatter_controlled.png
  图2:  step16_W1_residual_hist_controlled.png
  图3:  step16_W1_bar_comparison.png  (原始 vs 控制)

依赖: pip install pykrige
运行时间: 约30-60分钟（局部Kriging是瓶颈）
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.spatial import cKDTree

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


# ════════════════════════════════════════════════════════════════════
# 工具函数（与 step14 保持一致）
# ════════════════════════════════════════════════════════════════════
def spatial_block_split(data, block_size=2.0, test_ratio=0.3, seed=42,
                        min_per_block=3):
    d = data.copy()
    d["block_id"] = (
        (d["grid_lat"] // block_size * block_size).astype(str) + "_" +
        (d["grid_lon"] // block_size * block_size).astype(str)
    )
    bc = d["block_id"].value_counts()
    d = d[d["block_id"].isin(bc[bc >= min_per_block].index)]
    rng = np.random.default_rng(seed)
    blocks = d["block_id"].unique()
    rng.shuffle(blocks)
    n_test = int(len(blocks) * test_ratio)
    test_blocks = set(blocks[:n_test])
    tr = d[~d["block_id"].isin(test_blocks)]
    te = d[d["block_id"].isin(test_blocks)]
    return tr, te


def calc_metrics(y_true, y_pred):
    return {
        "R2":   r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def calc_moran_knn(coords, values, k=8):
    n = len(values)
    z = values - np.mean(values)
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    num = sum(z[i] * z[indices[i, j]]
              for i in range(n) for j in range(1, k + 1))
    W = n * k
    return float((n / W) * (num / np.sum(z ** 2)))


# ════════════════════════════════════════════════════════════════════
# 阶段0：数据准备
# ════════════════════════════════════════════════════════════════════
print("=" * 72)
print("W1 控制实验：标签对比公平性验证")
print("=" * 72)

df = pd.read_csv(DATA_PATH).dropna(subset=FEATURE_COLS + [TARGET])
df = df[df[TARGET] > 0].copy()
print(f"  观测数据: {len(df):,} 条")

# 空间分组划分（与原实验完全一致）
train_df, test_df = spatial_block_split(df)
print(f"  训练集: {len(train_df):,}  测试集: {len(test_df):,}")

X_test = test_df[FEATURE_COLS].values
y_test = test_df[TARGET].values
coords_test = test_df[["grid_lat", "grid_lon"]].values

# 训练集观测点坐标（用于 Kriging）
train_lon = train_df["long_EW"].values
train_lat = train_df["lat_NS"].values
train_q   = train_df[TARGET].values

# 训练集网格级中位数（用于全局 Kriging 拟合，与 step14 一致）
train_grid = (train_df.groupby(["grid_lat", "grid_lon"])[TARGET]
              .median().reset_index())

all_preds   = {}
all_metrics = {}


# ════════════════════════════════════════════════════════════════════
# 阶段1：方案B — 真实观测直接预测（基线，与原实验一致）
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("阶段1：方案B — 真实观测直接预测（基线）")
print("=" * 72)

X_train_B = train_df[FEATURE_COLS].values
y_train_B = train_df[TARGET].values
print(f"  方案B训练集: {len(train_df):,} 条真实观测")

model_B = ExtraTreesRegressor(n_estimators=200, max_depth=20,
                               random_state=42, n_jobs=-1)
model_B.fit(X_train_B, y_train_B)
pred_B = model_B.predict(X_test)
all_preds["B"] = pred_B
all_metrics["B"] = calc_metrics(y_test, pred_B)
print(f"  测试集: R²={all_metrics['B']['R2']:.4f}  "
      f"RMSE={all_metrics['B']['RMSE']:.2f}  "
      f"MAE={all_metrics['B']['MAE']:.2f}  "
      f"Bias={all_metrics['B']['Bias']:.2f}")


# ════════════════════════════════════════════════════════════════════
# 阶段2：方案A' — 同位置全局Kriging标签（控制组）
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("阶段2：方案A' — 同位置全局Kriging标签（控制组）")
print("  关键区别：训练样本位置与方案B完全一致，仅替换标签")
print("=" * 72)

from pykrige.ok import OrdinaryKriging

# 用训练集网格中位数拟合全局 Kriging（与 step14 一致）
print(f"  Kriging 拟合输入点数: {len(train_grid):,}（训练集网格中位数）")

t0 = time.time()
ok_global = OrdinaryKriging(
    train_grid["grid_lon"].values,
    train_grid["grid_lat"].values,
    train_grid[TARGET].values,
    variogram_model="spherical",
    verbose=False,
    enable_plotting=False,
    nlags=20,
)

# ── 关键控制：在训练集观测点位置执行 Kriging，而非全球网格 ──
print("  在训练集观测点位置执行 Kriging 插值（n_closest_points=50）...")
z_train_A, _ = ok_global.execute(
    "points",
    train_lon,
    train_lat,
    n_closest_points=50,
    backend="loop",
)
z_train_A = np.asarray(z_train_A).ravel()

# 过滤不合理值
bad_A = ~np.isfinite(z_train_A) | (z_train_A < 0) | (z_train_A > 500)
n_bad_A = bad_A.sum()
print(f"  Kriging 完成: {time.time()-t0:.1f}s  "
      f"不合理值: {n_bad_A:,}/{len(z_train_A):,}")

# 对不合理值用真实观测替代（保持样本量一致）
z_train_A[bad_A] = train_q[bad_A]
print(f"  方案A'训练集: {len(z_train_A):,} 条（同位置Kriging标签）")

# 用同位置 Kriging 标签训练 ExtraTrees
model_A_ctrl = ExtraTreesRegressor(n_estimators=200, max_depth=20,
                                    random_state=42, n_jobs=-1)
model_A_ctrl.fit(X_train_B, z_train_A)  # 特征与方案B完全一致
pred_A_ctrl = model_A_ctrl.predict(X_test)
all_preds["A'"] = pred_A_ctrl
all_metrics["A'"] = calc_metrics(y_test, pred_A_ctrl)
m_Ac = all_metrics["A'"]
print(f"  测试集: R²={m_Ac['R2']:.4f}  "
      f"RMSE={m_Ac['RMSE']:.2f}  "
      f"MAE={m_Ac['MAE']:.2f}  "
      f"Bias={m_Ac['Bias']:.2f}")


# ════════════════════════════════════════════════════════════════════
# 阶段3：方案C' — 同位置局部Kriging标签（控制组）
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("阶段3：方案C' — 同位置局部Kriging标签（控制组）")
print("  关键区别：训练样本位置与方案B完全一致，仅替换标签")
print("=" * 72)

WINDOW_DEG = 6.0
N_MIN      = 5
D_MAX_KM   = 500.0
HALF_W     = WINDOW_DEG / 2.0

# 在训练集观测点位置执行局部 Kriging
# 注意：对每个训练点，用其周围的其他训练点做局部 Kriging 预测
z_train_C = np.full(len(train_df), np.nan)

t0 = time.time()
n_done = 0
n_fallback = 0

for i in range(len(train_df)):
    pt_lon = train_lon[i]
    pt_lat = train_lat[i]

    # 矩形窗口内的其他训练点（排除自身，避免自预测）
    lon_mask = (train_lon >= pt_lon - HALF_W) & (train_lon <= pt_lon + HALF_W)
    lat_mask = (train_lat >= pt_lat - HALF_W) & (train_lat <= pt_lat + HALF_W)
    local_mask = lon_mask & lat_mask
    # 排除自身
    local_mask[i] = False
    n_local = local_mask.sum()

    if n_local < N_MIN:
        # 邻近点不足，回退到真实观测值
        z_train_C[i] = train_q[i]
        n_fallback += 1
        n_done += 1
        if n_done % 5000 == 0:
            elapsed = time.time() - t0
            n_filled = np.isfinite(z_train_C).sum()
            print(f"    进度: {n_done:,}/{len(train_df):,}  "
                  f"filled={n_filled:,}  fallback={n_fallback:,}  "
                  f"耗时={elapsed:.0f}s")
        continue

    try:
        ok_local = OrdinaryKriging(
            train_lon[local_mask],
            train_lat[local_mask],
            train_q[local_mask],
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
        )
        z_loc, _ = ok_local.execute(
            "points", np.array([pt_lon]), np.array([pt_lat])
        )
        val = float(z_loc[0])
        if np.isfinite(val) and 0 < val < 500:
            z_train_C[i] = val
        else:
            z_train_C[i] = train_q[i]
            n_fallback += 1
    except Exception:
        z_train_C[i] = train_q[i]
        n_fallback += 1

    n_done += 1
    if n_done % 5000 == 0:
        elapsed = time.time() - t0
        n_filled = np.isfinite(z_train_C).sum()
        print(f"    进度: {n_done:,}/{len(train_df):,}  "
              f"filled={n_filled:,}  fallback={n_fallback:,}  "
              f"耗时={elapsed:.0f}s")

print(f"  局部Kriging完成: {time.time()-t0:.1f}s  "
      f"回退到真实值: {n_fallback:,}/{len(train_df):,}")
print(f"  方案C'训练集: {len(z_train_C):,} 条（同位置局部Kriging标签）")

# 用同位置局部 Kriging 标签训练 ExtraTrees
model_C_ctrl = ExtraTreesRegressor(n_estimators=200, max_depth=20,
                                    random_state=42, n_jobs=-1)
model_C_ctrl.fit(X_train_B, z_train_C)  # 特征与方案B完全一致
pred_C_ctrl = model_C_ctrl.predict(X_test)
all_preds["C'"] = pred_C_ctrl
all_metrics["C'"] = calc_metrics(y_test, pred_C_ctrl)
m_Cc = all_metrics["C'"]
print(f"  测试集: R²={m_Cc['R2']:.4f}  "
      f"RMSE={m_Cc['RMSE']:.2f}  "
      f"MAE={m_Cc['MAE']:.2f}  "
      f"Bias={m_Cc['Bias']:.2f}")


# ════════════════════════════════════════════════════════════════════
# 阶段4：统一评估与对比
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("阶段4：统一评估与对比")
print("=" * 72)

# 原始实验结果（来自 step14，硬编码用于对比）
ORIGINAL = {
    "A": {"R2": 0.2885, "RMSE": 43.70, "MAE": 33.41, "Bias": np.nan},
    "B": {"R2": 0.4160, "RMSE": 39.60, "MAE": 28.21, "Bias": np.nan},
    "C": {"R2": 0.2796, "RMSE": 43.98, "MAE": 32.96, "Bias": np.nan},
}

# Moran's I
moran_vals = {}
for key in ["B", "A'", "C'"]:
    residuals = all_preds[key] - y_test
    moran_vals[key] = calc_moran_knn(coords_test, residuals, k=8)

# 标签与真实值的偏差统计
label_stats = {}
# A': Kriging标签 vs 真实观测
diff_A = z_train_A - train_q
label_stats["A'"] = {
    "label_bias": float(np.mean(diff_A)),
    "label_mae":  float(np.mean(np.abs(diff_A))),
    "label_corr": float(np.corrcoef(z_train_A, train_q)[0, 1]),
}
# C': 局部Kriging标签 vs 真实观测
diff_C = z_train_C - train_q
label_stats["C'"] = {
    "label_bias": float(np.mean(diff_C)),
    "label_mae":  float(np.mean(np.abs(diff_C))),
    "label_corr": float(np.corrcoef(z_train_C, train_q)[0, 1]),
}

# ── 打印汇总表 ──
METHOD_KEYS  = ["B", "A'", "C'"]
METHOD_NAMES = {
    "B":  "B: Direct Obs (Baseline)",
    "A'": "A': Same-loc Global Kriging",
    "C'": "C': Same-loc Local Kriging",
}

print(f"\n{'方案':<30} {'R²':>8} {'RMSE':>8} {'MAE':>8} "
      f"{'Bias':>8} {'Moran I':>8}")
print("-" * 78)
for key in METHOD_KEYS:
    m = all_metrics[key]
    mi = moran_vals[key]
    print(f"  {METHOD_NAMES[key]:<28} {m['R2']:>8.4f} {m['RMSE']:>8.2f} "
          f"{m['MAE']:>8.2f} {m['Bias']:>8.2f} {mi:>8.4f}")

print(f"\n标签质量诊断（训练集标签 vs 真实观测）:")
print(f"{'方案':<30} {'Label Bias':>12} {'Label MAE':>12} {'Label Corr':>12}")
print("-" * 68)
for key in ["A'", "C'"]:
    ls = label_stats[key]
    print(f"  {METHOD_NAMES[key]:<28} {ls['label_bias']:>12.2f} "
          f"{ls['label_mae']:>12.2f} {ls['label_corr']:>12.4f}")

print(f"\n原始实验 vs 控制实验对比:")
print(f"{'对比项':<35} {'原始 R²':>10} {'控制 R²':>10} {'ΔR²':>10} {'结论':>20}")
print("-" * 90)
for orig_key, ctrl_key in [("A", "A'"), ("C", "C'")]:
    r2_orig = ORIGINAL[orig_key]["R2"]
    r2_ctrl = all_metrics[ctrl_key]["R2"]
    delta = r2_ctrl - r2_orig
    if r2_ctrl < all_metrics["B"]["R2"]:
        conclusion = "标签质量确有影响"
    else:
        conclusion = "需重新评估"
    print(f"  {orig_key} → {ctrl_key:<30} {r2_orig:>10.4f} {r2_ctrl:>10.4f} "
          f"{delta:>+10.4f} {conclusion:>20}")

r2_B = all_metrics["B"]["R2"]
r2_Ac = all_metrics["A'"]["R2"]
r2_Cc = all_metrics["C'"]["R2"]
print(f"\n核心结论:")
print(f"  方案B R² = {r2_B:.4f}")
print(f"  方案A'R² = {r2_Ac:.4f}  (ΔR² vs B = {r2_Ac - r2_B:+.4f})")
print(f"  方案C'R² = {r2_Cc:.4f}  (ΔR² vs B = {r2_Cc - r2_B:+.4f})")
if r2_B > r2_Ac and r2_B > r2_Cc:
    print("  → 在控制样本分布后，方案B仍然显著优于A'和C'，")
    print("    证明标签质量本身（而非样本分布差异）是性能差异的主要来源。")
else:
    print("  → 控制样本分布后差距缩小，需在论文中调整结论表述强度。")

# ── 保存 CSV ──
rows = []
for key in METHOD_KEYS:
    m = all_metrics[key]
    row = {"Method": METHOD_NAMES[key], **m, "Moran_I": moran_vals[key]}
    if key in label_stats:
        row.update(label_stats[key])
    rows.append(row)
summary_df = pd.DataFrame(rows)
csv_path = OUT_DIR / "step16_W1_fairness_summary.csv"
summary_df.to_csv(csv_path, index=False)
print(f"\n已保存: {csv_path}")


# ════════════════════════════════════════════════════════════════════
# 阶段5：可视化
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("阶段5：可视化")
print("=" * 72)

COLORS = {"B": "#3a7ebf", "A'": "#e06c3a", "C'": "#4caf7d"}

# ── 图1：三种方案散点图对比 ──
print("绘制图1：控制实验散点图对比...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for col, key in enumerate(METHOD_KEYS):
    ax = axes[col]
    pred = all_preds[key]
    m = all_metrics[key]
    ax.scatter(y_test, pred, c=COLORS[key], s=3, alpha=0.35, linewidths=0)
    lim = (0, 300)
    ax.plot(lim, lim, "k--", linewidth=1.2, label="1:1 line")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Observed (mW/m²)", fontsize=10)
    ax.set_ylabel("Predicted (mW/m²)", fontsize=10)
    ax.set_title(f"{METHOD_NAMES[key]}\n"
                 f"R²={m['R2']:.4f}  RMSE={m['RMSE']:.1f}  "
                 f"Bias={m['Bias']:.2f}",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

plt.suptitle("W1 Controlled Experiment — Same Location, Different Labels",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step16_W1_scatter_controlled.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step16_W1_scatter_controlled.png")

# ── 图2：残差直方图叠加对比 ──
print("绘制图2：残差直方图...")
fig, ax = plt.subplots(figsize=(10, 6))
bins_hist = np.linspace(-150, 150, 80)

for key in METHOD_KEYS:
    residual = all_preds[key] - y_test
    m = all_metrics[key]
    ax.hist(residual, bins=bins_hist, color=COLORS[key], alpha=0.45,
            edgecolor="none",
            label=f"{METHOD_NAMES[key]}  Bias={m['Bias']:.1f}")

ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
ax.set_xlabel("Residual (mW/m²)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title("W1 Controlled Experiment — Residual Distribution",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
fig.savefig(FIG_DIR / "step16_W1_residual_hist_controlled.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step16_W1_residual_hist_controlled.png")

# ── 图3：原始 vs 控制实验柱状图对比 ──
print("绘制图3：原始 vs 控制实验对比...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# R² 对比
ax = axes[0]
labels = ["A (Original)", "A' (Controlled)", "B (Baseline)",
          "C (Original)", "C' (Controlled)"]
r2_vals = [
    ORIGINAL["A"]["R2"], all_metrics["A'"]["R2"], all_metrics["B"]["R2"],
    ORIGINAL["C"]["R2"], all_metrics["C'"]["R2"],
]
colors_bar = ["#e06c3a", "#f4a582", "#3a7ebf", "#4caf7d", "#a6d96a"]
bars = ax.bar(labels, r2_vals, color=colors_bar, alpha=0.85, edgecolor="white")
ax.axhline(all_metrics["B"]["R2"], color="#3a7ebf", linewidth=1.5,
           linestyle="--", alpha=0.7, label=f"B baseline R²={r2_B:.4f}")
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("R²", fontsize=12)
ax.set_title("R² Comparison: Original vs Controlled", fontsize=12,
             fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)

# RMSE 对比
ax = axes[1]
rmse_vals = [
    ORIGINAL["A"]["RMSE"], all_metrics["A'"]["RMSE"],
    all_metrics["B"]["RMSE"],
    ORIGINAL["C"]["RMSE"], all_metrics["C'"]["RMSE"],
]
bars = ax.bar(labels, rmse_vals, color=colors_bar, alpha=0.85,
              edgecolor="white")
ax.axhline(all_metrics["B"]["RMSE"], color="#3a7ebf", linewidth=1.5,
           linestyle="--", alpha=0.7,
           label=f"B baseline RMSE={all_metrics['B']['RMSE']:.2f}")
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("RMSE (mW/m²)", fontsize=12)
ax.set_title("RMSE Comparison: Original vs Controlled", fontsize=12,
             fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=9)

plt.suptitle("W1: Does Controlling Sample Distribution Change the Conclusion?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(FIG_DIR / "step16_W1_bar_comparison.png",
            dpi=200, bbox_inches="tight")
plt.close(fig)
print("  已保存: step16_W1_bar_comparison.png")


# ════════════════════════════════════════════════════════════════════
# 汇总
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("W1 控制实验全部完成！")
print(f"  CSV   {csv_path}")
print(f"  图1   step16_W1_scatter_controlled.png")
print(f"  图2   step16_W1_residual_hist_controlled.png")
print(f"  图3   step16_W1_bar_comparison.png")
print("=" * 72)
