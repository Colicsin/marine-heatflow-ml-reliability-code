"""
Step 4: 数据最终整合 + 质量评估 + 可视化

处理规则：
  - 丢弃13个基础特征有缺失的897个格点
  - 洋壳年龄NaN填 -1（大陆架标记值）
  - 输出最终可用数据集 final_dataset.csv
  - 生成完整的质量评估可视化
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geopandas as gpd

ROOT     = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/dataset_with_features.csv"
NE10_PATH = ROOT / "data/natural_earth/ne_10m_land.shp"
OUT_DIR   = ROOT / "data/processed"
FIG_DIR   = ROOT / "outputs/figures"
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

# ── 1. 读取并整合 ─────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"原始格点数: {len(df):,}")

# 丢弃13个基础特征有缺失的格点
base_cols = FEATURE_COLS[:-1]  # 不含洋壳年龄
df = df.dropna(subset=base_cols).copy()
print(f"丢弃特征缺失格点后: {len(df):,}")

# 洋壳年龄NaN填-1
n_age_nan = df["oceanic_crust_age_Ma"].isna().sum()
df["oceanic_crust_age_Ma"] = df["oceanic_crust_age_Ma"].fillna(-1.0)
print(f"洋壳年龄NaN填-1: {n_age_nan:,} 个格点")

# 确认无缺失
assert df[FEATURE_COLS].isna().sum().sum() == 0, "仍有缺失值！"
assert df["median_q"].isna().sum() == 0

# 洋盆标签（从旧数据集复用逻辑：简单按经纬度判断）
def assign_basin(lon, lat):
    if lat < -60:
        return "Southern"
    if lon < -20 or lon > 140:
        return "Pacific"
    if -20 <= lon <= 20:
        return "Atlantic"
    if 20 < lon <= 140 and lat > 0:
        return "Indian" if lon > 60 else "Atlantic"
    if 20 < lon <= 140 and lat <= 0:
        return "Indian"
    return "Other"

if "basin" not in df.columns:
    df["basin"] = [assign_basin(r.grid_lon, r.grid_lat) for r in df.itertuples()]

print(f"\n最终数据集: {len(df):,} 个格点，{len(FEATURE_COLS)} 个特征")
print(f"热流值: {df['median_q'].min():.1f} ~ {df['median_q'].max():.1f} mW/m²  "
      f"均值={df['median_q'].mean():.1f}  std={df['median_q'].std():.1f}")
print(f"\n洋盆分布:")
print(df["basin"].value_counts().to_string())
print(f"\n洋壳年龄分布:")
print(f"  有洋壳年龄(>0): {(df['oceanic_crust_age_Ma']>0).sum():,}")
print(f"  大陆架(-1):     {(df['oceanic_crust_age_Ma']==-1).sum():,}")

# 保存最终数据集
out_path = OUT_DIR / "final_dataset.csv"
df.to_csv(out_path, index=False)
print(f"\n保存至: {out_path}")

# ── 2. 可视化 ─────────────────────────────────────────────────────
print("\n生成可视化...")
world = gpd.read_file(NE10_PATH)
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor('#0d1117')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# ── 图1：全球分布（按热流值着色）────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#0a1628')
world.plot(ax=ax1, color='#2a2a3a', edgecolor='#555', linewidth=0.3)
sc = ax1.scatter(df["grid_lon"], df["grid_lat"],
                 c=df["median_q"], cmap="plasma",
                 vmin=0, vmax=200, s=6, alpha=0.85, linewidths=0, zorder=4)
cbar = plt.colorbar(sc, ax=ax1, orientation="vertical", pad=0.01, shrink=0.8)
cbar.set_label("Heat Flow (mW/m²)", color="white", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
ax1.set_xlim(-180, 180); ax1.set_ylim(-90, 90)
ax1.set_title(f"Final Dataset — {len(df):,} grid cells (0.5°×0.5°) with real observations",
              color="white", fontsize=13)
ax1.set_xlabel("Longitude", color="white"); ax1.set_ylabel("Latitude", color="white")
ax1.tick_params(colors="white")
ax1.grid(color="#222", linewidth=0.3, alpha=0.5)
for sp in ax1.spines.values(): sp.set_edgecolor("#444")

# ── 图2：热流值分布直方图 ────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#0d1117')
ax2.hist(df["median_q"], bins=60, color="#ff6b35", alpha=0.8, edgecolor="#333", linewidth=0.3)
ax2.axvline(df["median_q"].median(), color="white", linestyle="--", linewidth=1.5,
            label=f"Median={df['median_q'].median():.0f}")
ax2.axvline(df["median_q"].mean(), color="#4af", linestyle="--", linewidth=1.5,
            label=f"Mean={df['median_q'].mean():.0f}")
ax2.set_xlabel("Heat Flow (mW/m²)", color="white"); ax2.set_ylabel("Count", color="white")
ax2.set_title("Heat Flow Distribution", color="white", fontsize=11)
ax2.tick_params(colors="white"); ax2.grid(color="#222", linewidth=0.3, alpha=0.5)
ax2.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax2.spines.values(): sp.set_edgecolor("#444")

# ── 图3：观测数量分布 ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#0d1117')
count_bins = [1,2,3,5,10,50,9999,99999]
count_labels = ["1","2","3","4","5-9","10-49","≥50"]
count_vals = [int(((df["count"]>=count_bins[i])&(df["count"]<count_bins[i+1])).sum())
              for i in range(len(count_labels))]
colors = plt.cm.YlOrRd(np.linspace(0.2, 0.9, len(count_labels)))
bars = ax3.bar(range(len(count_labels)), count_vals, color=colors, edgecolor="#333", linewidth=0.4)
for bar, val in zip(bars, count_vals):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
             f"{val:,}\n({val/len(df)*100:.0f}%)", ha="center", va="bottom",
             color="white", fontsize=7)
ax3.set_xticks(range(len(count_labels)))
ax3.set_xticklabels([f"{l}" for l in count_labels], color="white", fontsize=9)
ax3.set_xlabel("Observations per grid cell", color="white"); ax3.set_ylabel("Count", color="white")
ax3.set_title("Observation Count per Grid Cell", color="white", fontsize=11)
ax3.tick_params(colors="white"); ax3.grid(axis="y", color="#222", linewidth=0.3, alpha=0.5)
for sp in ax3.spines.values(): sp.set_edgecolor("#444")

# ── 图4：洋壳年龄分布 ────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor('#0d1117')
age_valid = df[df["oceanic_crust_age_Ma"] > 0]["oceanic_crust_age_Ma"]
ax4.hist(age_valid, bins=50, color="#4a9eff", alpha=0.8, edgecolor="#333", linewidth=0.3)
ax4.axvline(age_valid.median(), color="white", linestyle="--", linewidth=1.5,
            label=f"Median={age_valid.median():.0f} Ma")
n_continental = int((df["oceanic_crust_age_Ma"]==-1).sum())
ax4.set_xlabel("Oceanic Crust Age (Ma)", color="white"); ax4.set_ylabel("Count", color="white")
ax4.set_title(f"Oceanic Crust Age Distribution\n(continental shelf: {n_continental:,} grids set to -1)",
              color="white", fontsize=10)
ax4.tick_params(colors="white"); ax4.grid(color="#222", linewidth=0.3, alpha=0.5)
ax4.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
for sp in ax4.spines.values(): sp.set_edgecolor("#444")

# ── 图5：各洋盆热流箱线图 ────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_facecolor('#0d1117')
basins_order = ["Pacific", "Atlantic", "Indian", "Southern", "Other"]
basin_data = [df[df["basin"]==b]["median_q"].values for b in basins_order]
basin_data = [d for d in basin_data if len(d) > 0]
basin_labels = [b for b, d in zip(basins_order, [df[df["basin"]==b]["median_q"].values
                for b in basins_order]) if len(d) > 0]
bp = ax5.boxplot(basin_data, patch_artist=True, notch=False,
                 medianprops=dict(color="white", linewidth=2))
colors_bp = ["#ff6b35","#4a9eff","#4aff91","#ffcc00","#cc88ff"]
for patch, color in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(color); patch.set_alpha(0.7)
for element in ["whiskers","caps","fliers"]:
    for item in bp[element]: item.set_color("#888")
ax5.set_xticklabels([f"{b}\n(n={len(df[df['basin']==b]):,})" for b in basin_labels],
                    color="white", fontsize=8)
ax5.set_ylabel("Heat Flow (mW/m²)", color="white")
ax5.set_title("Heat Flow by Basin", color="white", fontsize=11)
ax5.tick_params(colors="white"); ax5.grid(axis="y", color="#222", linewidth=0.3, alpha=0.5)
for sp in ax5.spines.values(): sp.set_edgecolor("#444")

# ── 图6：热流 vs 洋壳年龄散点 ────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 1])
ax6.set_facecolor('#0d1117')
df_age = df[df["oceanic_crust_age_Ma"] > 0]
ax6.scatter(df_age["oceanic_crust_age_Ma"], df_age["median_q"],
            c=df_age["median_q"], cmap="plasma", vmin=0, vmax=200,
            s=3, alpha=0.5, linewidths=0)
ax6.set_xlabel("Oceanic Crust Age (Ma)", color="white")
ax6.set_ylabel("Heat Flow (mW/m²)", color="white")
ax6.set_title("Heat Flow vs Oceanic Crust Age\n(plate cooling trend expected)", color="white", fontsize=10)
ax6.tick_params(colors="white"); ax6.grid(color="#222", linewidth=0.3, alpha=0.5)
for sp in ax6.spines.values(): sp.set_edgecolor("#444")

# ── 图7：纬度带分布 ──────────────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
ax7.set_facecolor('#0d1117')
lat_bins = np.arange(-90, 91, 10)
lat_hist, _ = np.histogram(df["grid_lat"], bins=lat_bins)
lat_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
ax7.barh(lat_centers, lat_hist, height=9, color="#4a9eff", alpha=0.8, edgecolor="#333", linewidth=0.3)
ax7.set_xlabel("Number of Grid Cells", color="white"); ax7.set_ylabel("Latitude (°)", color="white")
ax7.set_title("Latitudinal Distribution\n(10° bands)", color="white", fontsize=11)
ax7.set_ylim(-90, 90); ax7.axhline(0, color="#555", linewidth=0.8, linestyle=":")
ax7.tick_params(colors="white"); ax7.grid(axis="x", color="#222", linewidth=0.3, alpha=0.5)
for sp in ax7.spines.values(): sp.set_edgecolor("#444")

fig_path = FIG_DIR / "step4_dataset_quality.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"图保存至: {fig_path}")
plt.close()
