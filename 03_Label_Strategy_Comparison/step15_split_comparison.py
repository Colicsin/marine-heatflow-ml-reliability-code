"""Compare random and spatial-block validation results from label-strategy self-evaluation."""

from pathlib import Path
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SPATIAL_PATH = OUT_DIR / "step15_self_eval_summary.csv"
RANDOM_PATH = OUT_DIR / "step15_random_self_eval_summary.csv"
CSV_OUT = OUT_DIR / "step15_split_comparison.csv"
FIG_OUT = FIG_DIR / "step15_split_comparison.png"


def main():
    spatial = pd.read_csv(SPATIAL_PATH).copy()
    random = pd.read_csv(RANDOM_PATH).copy()

    spatial["Split"] = "Spatial"
    random["Split"] = "Random"

    merged = random.merge(
        spatial,
        on=["Method", "Ground_Truth"],
        suffixes=("_random", "_spatial"),
    )

    merged["R2_gap_random_minus_spatial"] = merged["R2_random"] - merged["R2_spatial"]
    merged["RMSE_gap_random_minus_spatial"] = merged["RMSE_random"] - merged["RMSE_spatial"]
    merged["MAE_gap_random_minus_spatial"] = merged["MAE_random"] - merged["MAE_spatial"]
    merged["Bias_gap_random_minus_spatial"] = merged["Bias_random"] - merged["Bias_spatial"]
    merged["Moran_gap_random_minus_spatial"] = (
        merged["Moran_I_random"] - merged["Moran_I_spatial"]
    )

    cols = [
        "Method",
        "Ground_Truth",
        "N_Test_random",
        "N_Test_spatial",
        "R2_random",
        "R2_spatial",
        "R2_gap_random_minus_spatial",
        "RMSE_random",
        "RMSE_spatial",
        "RMSE_gap_random_minus_spatial",
        "MAE_random",
        "MAE_spatial",
        "MAE_gap_random_minus_spatial",
        "Bias_random",
        "Bias_spatial",
        "Bias_gap_random_minus_spatial",
        "Moran_I_random",
        "Moran_I_spatial",
        "Moran_gap_random_minus_spatial",
    ]
    merged = merged[cols]
    merged.to_csv(CSV_OUT, index=False)

    print("=" * 78)
    print("Step15 划分方式对比：随机划分 vs 空间分组")
    print("=" * 78)
    print(
        f"{'Method':<22} {'R2(random)':>11} {'R2(spatial)':>12} {'Delta':>9} "
        f"{'RMSE(random)':>13} {'RMSE(spatial)':>14} {'Delta':>9}"
    )
    print("-" * 92)
    for row in merged.itertuples(index=False):
        print(
            f"{row.Method:<22} {row.R2_random:>11.4f} {row.R2_spatial:>12.4f} "
            f"{row.R2_gap_random_minus_spatial:>+9.4f} {row.RMSE_random:>13.2f} "
            f"{row.RMSE_spatial:>14.2f} {row.RMSE_gap_random_minus_spatial:>+9.2f}"
        )

    methods = merged["Method"].tolist()
    x = np.arange(len(methods))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(x - width / 2, merged["R2_random"], width, label="Random", color="#3a7ebf")
    ax.bar(x + width / 2, merged["R2_spatial"], width, label="Spatial", color="#e06c3a")
    ax.set_ylabel("R2")
    ax.set_title("Step15 Self-Eval: R2 by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color="black", linewidth=0.8)

    ax = axes[1]
    ax.bar(x - width / 2, merged["RMSE_random"], width, label="Random", color="#3a7ebf")
    ax.bar(x + width / 2, merged["RMSE_spatial"], width, label="Spatial", color="#e06c3a")
    ax.set_ylabel("RMSE (mW/m2)")
    ax.set_title("Step15 Self-Eval: RMSE by Split")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Step15 Random vs Spatial Split Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved CSV: {CSV_OUT}")
    print(f"Saved Fig: {FIG_OUT}")


if __name__ == "__main__":
    main()
