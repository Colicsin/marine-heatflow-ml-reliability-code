from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "final_dataset.csv"
LAND_SHP = ROOT / "data" / "natural_earth" / "ne_110m_land.shp"
OUT_DIR = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "step4_observed_heatflow_robinson.png"


# Robinson projection lookup table (Snyder, 1993), 5-degree intervals.
ROBINSON_X = np.array([
    1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216,
    0.8962, 0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322
])
ROBINSON_Y = np.array([
    0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958,
    0.5571, 0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000
])
ROBINSON_X_SCALE = 0.8487
ROBINSON_Y_SCALE = 1.3523
XMAX = ROBINSON_X_SCALE * np.pi
YMAX = ROBINSON_Y_SCALE


def robinson_project(lon, lat):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    abslat = np.clip(np.abs(lat), 0, 90)
    idx = np.minimum((abslat // 5).astype(int), 17)
    frac = (abslat - idx * 5) / 5.0

    xcoef = ROBINSON_X[idx] + frac * (ROBINSON_X[idx + 1] - ROBINSON_X[idx])
    ycoef = ROBINSON_Y[idx] + frac * (ROBINSON_Y[idx + 1] - ROBINSON_Y[idx])

    x = ROBINSON_X_SCALE * np.radians(lon) * xcoef
    y = ROBINSON_Y_SCALE * np.sign(lat) * ycoef
    return x, y


def read_polygon_parts(shp_path: Path):
    with shp_path.open("rb") as f:
        data = f.read()

    pos = 100  # skip shapefile header
    polygons = []

    while pos + 8 <= len(data):
        _, rec_len_words = struct.unpack(">2i", data[pos:pos + 8])
        pos += 8
        rec_len = rec_len_words * 2
        rec = data[pos:pos + rec_len]
        pos += rec_len

        if len(rec) < 44:
            continue

        shape_type = struct.unpack("<i", rec[:4])[0]
        if shape_type == 0:
            continue
        if shape_type != 5:
            continue

        num_parts, num_points = struct.unpack("<2i", rec[36:44])
        parts = struct.unpack("<" + "i" * num_parts, rec[44:44 + 4 * num_parts])
        points_offset = 44 + 4 * num_parts
        points = np.frombuffer(
            rec,
            dtype="<f8",
            count=num_points * 2,
            offset=points_offset,
        ).reshape(num_points, 2)

        for i, start in enumerate(parts):
            end = parts[i + 1] if i + 1 < num_parts else num_points
            part = points[start:end]
            if len(part) >= 3:
                polygons.append(part.copy())

    return polygons


def build_cmap():
    colors = [
        "#4b5ab8",
        "#6daed6",
        "#c9e4f2",
        "#f7f7d9",
        "#fee08b",
        "#fdae61",
        "#f46d43",
        "#d73027",
        "#a50026",
    ]
    return LinearSegmentedColormap.from_list("obs_heatflow_paper", colors, N=256)


def main():
    df = pd.read_csv(DATA_PATH)
    polygons = read_polygon_parts(LAND_SHP)

    x, y = robinson_project(df["grid_lon"].values, df["grid_lat"].values)

    fig = plt.figure(figsize=(12.8, 7.6), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("white")

    globe = Ellipse(
        (0, 0),
        width=2 * XMAX,
        height=2 * YMAX,
        facecolor="white",
        edgecolor="#3a3a3a",
        linewidth=1.0,
        zorder=0,
    )
    ax.add_patch(globe)

    for poly in polygons:
        px, py = robinson_project(poly[:, 0], poly[:, 1])
        patch = plt.Polygon(
            np.column_stack([px, py]),
            facecolor="#d0d0d0",
            edgecolor="#9a9a9a",
            linewidth=0.45,
            zorder=3,
        )
        patch.set_clip_path(globe)
        ax.add_patch(patch)

    cmap = build_cmap()
    norm = mcolors.Normalize(vmin=0, vmax=200)
    sc = ax.scatter(
        x,
        y,
        c=df["median_q"].clip(0, 200),
        cmap=cmap,
        norm=norm,
        s=6,
        marker="s",
        linewidths=0,
        alpha=0.95,
        zorder=2,
        rasterized=True,
    )
    sc.set_clip_path(globe)

    ax.set_xlim(-XMAX * 1.02, XMAX * 1.02)
    ax.set_ylim(-YMAX * 1.08, YMAX * 1.08)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.set_title(
        "Observed Oceanic Heat Flow (7,218 grid cells, 0.5° × 0.5°)",
        fontsize=17,
        fontweight="bold",
        pad=12,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        pad=0.055,
        fraction=0.04,
        aspect=42,
        extend="max",
    )
    cbar.set_label("Observed Heat Flow (mW/m²)", fontsize=15)
    cbar.ax.tick_params(labelsize=11)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.1)
    fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved figure to: {OUT_PATH}")


if __name__ == "__main__":
    main()
