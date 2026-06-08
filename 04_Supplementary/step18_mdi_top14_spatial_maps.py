"""Generate global spatial maps for MDI-pruned features and full-data predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "dataset_D_no_aggregation.csv"
GRID_PATH = ROOT / "data" / "features" / "Ocean_HeatFlow_Prediction_Data_with_Age.csv"
LAND_SHP = ROOT / "data" / "natural_earth" / "ne_110m_land.shp"
OUT_DIR = ROOT / "outputs" / "figures" / "mdi_top14_spatial_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "q"
EPS = 1e-3
MODEL = ExtraTreesRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)

BASE_COLS = [
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

TOP14 = [
    "CRUST1.0_mantle_rho_0.5deg",
    "LITH_IDW_lab",
    "log_hotspot_dist",
    "topo_topo_diff",
    "hotspot_mantle_inter",
    "hotspot_min_hotspot_distance_km",
    "log_volcano_dist",
    "volcano_latest_vocano_dist",
    "mantle_volcano_inter",
    "topo_topo_mean",
    "inv_age",
    "inv_upper_crust",
    "LITH_IDW_moho",
    "CRUST1.0_mid_crust_thickness_0.5deg",
]


@dataclass(frozen=True)
class MapSpec:
    col: str
    title: str
    colorbar: str
    cmap: str
    vmin: float | None = None
    vmax: float | None = None
    caption: str | None = None


MAP_SPECS = [
    MapSpec("q_pred_mWm2", "Global Ocean Heat Flow Prediction (MDI Top14)", "Predicted heat flow (mW m$^{-2}$)", "RdYlBu_r", 20, 180, "Predicted heat flow"),
    MapSpec("CRUST1.0_mantle_rho_0.5deg", "Mantle Density", "Mantle density (g cm$^{-3}$)", "viridis", caption="Mantle density"),
    MapSpec("LITH_IDW_lab", "Lithosphere-Asthenosphere Boundary Depth", "LAB depth (m)", "viridis_r", caption="LAB depth"),
    MapSpec("log_hotspot_dist", "Log Hotspot Distance", "log(Hotspot distance + 1)", "magma", caption="Log hotspot distance"),
    MapSpec("topo_topo_diff", "Topographic Relief", "Relief (m)", "terrain", caption="Topographic relief"),
    MapSpec("hotspot_mantle_inter", "Hotspot Distance x Mantle Density", "km x g cm$^{-3}$", "plasma", caption="Hotspot distance x mantle density"),
    MapSpec("hotspot_min_hotspot_distance_km", "Hotspot Distance", "Distance to hotspot (km)", "magma", caption="Hotspot distance"),
    MapSpec("log_volcano_dist", "Log Volcano Distance", "log(Volcano distance + 1)", "inferno", caption="Log volcano distance"),
    MapSpec("volcano_latest_vocano_dist", "Volcano Distance", "Distance to volcano (km)", "inferno", caption="Volcano distance"),
    MapSpec("mantle_volcano_inter", "Mantle Density x Volcano Distance", "g cm$^{-3}$ x km", "plasma", caption="Mantle density x volcano distance"),
    MapSpec("topo_topo_mean", "Mean Topography", "Mean topography (m)", "terrain", caption="Mean topography"),
    MapSpec("inv_age", "Inverse Oceanic Crust Age", "1 / (Age + 0.001)", "cividis", caption="Inverse crustal age"),
    MapSpec("inv_upper_crust", "Inverse Upper-Crust Thickness", "1 / (Thickness + 0.001)", "cividis", caption="Inverse upper-crust thickness"),
    MapSpec("LITH_IDW_moho", "Moho Depth", "Moho depth (m)", "viridis_r", caption="Moho depth"),
    MapSpec("CRUST1.0_mid_crust_thickness_0.5deg", "Middle-Crust Thickness", "Middle-crust thickness (km)", "viridis", caption="Middle-crust thickness"),
]


def add_engineered_features(df: pd.DataFrame, fill_age_nan: bool = True) -> pd.DataFrame:
    out = df.copy()
    if fill_age_nan:
        out["oceanic_crust_age_Ma"] = out["oceanic_crust_age_Ma"].fillna(-1.0)
    out["hotspot_mantle_inter"] = (
        out["hotspot_min_hotspot_distance_km"] * out["CRUST1.0_mantle_rho_0.5deg"]
    )
    out["mantle_volcano_inter"] = (
        out["CRUST1.0_mantle_rho_0.5deg"] * out["volcano_latest_vocano_dist"]
    )
    out["log_hotspot_dist"] = np.log1p(out["hotspot_min_hotspot_distance_km"])
    out["log_volcano_dist"] = np.log1p(out["volcano_latest_vocano_dist"])
    out["inv_age"] = 1.0 / (out["oceanic_crust_age_Ma"].clip(lower=0) + EPS)
    out["inv_upper_crust"] = 1.0 / (out["CRUST1.0_upper_crust_thickness_0.5deg"] + EPS)
    return out.replace([np.inf, -np.inf], np.nan)


def robinson(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_tab = np.array(
        [1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427, 0.9216,
         0.8962, 0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732, 0.6213, 0.5722, 0.5322]
    )
    y_tab = np.array(
        [0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340, 0.4958,
         0.5571, 0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936, 0.9394, 0.9761, 1.0000]
    )
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    abs_lat = np.clip(np.abs(lat_arr), 0, 90)
    steps = np.arange(0, 95, 5)
    x_coeff = np.interp(abs_lat, steps, x_tab)
    y_coeff = np.interp(abs_lat, steps, y_tab)
    x = 0.8487 * np.deg2rad(lon_arr) * x_coeff
    y = 1.3523 * np.sign(lat_arr) * y_coeff
    return x, y


def read_polygon_parts(shp_path: Path) -> list[np.ndarray]:
    parts_out: list[np.ndarray] = []
    with shp_path.open("rb") as f:
        f.read(100)
        while True:
            rec_header = f.read(8)
            if len(rec_header) < 8:
                break
            _, rec_len_words = struct.unpack(">2i", rec_header)
            content = f.read(rec_len_words * 2)
            if len(content) < 44:
                continue
            shape_type = struct.unpack("<i", content[:4])[0]
            if shape_type not in (5, 15):
                continue
            num_parts, num_points = struct.unpack("<2i", content[36:44])
            part_offset = 44
            point_offset = part_offset + 4 * num_parts
            if num_parts <= 0 or num_points <= 0:
                continue
            part_idx = list(struct.unpack(f"<{num_parts}i", content[part_offset:point_offset]))
            point_data = np.frombuffer(content[point_offset:point_offset + num_points * 16], dtype="<f8")
            points = point_data.reshape(-1, 2)
            bounds = part_idx + [num_points]
            for start, end in zip(bounds[:-1], bounds[1:]):
                ring = points[start:end]
                if len(ring) >= 3:
                    parts_out.extend(split_dateline(ring))
    return parts_out


def split_dateline(ring: np.ndarray) -> list[np.ndarray]:
    if len(ring) < 3:
        return []
    lon = ring[:, 0]
    jumps = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
    if len(jumps) == 0:
        return [ring]
    chunks = np.split(ring, jumps)
    return [chunk for chunk in chunks if len(chunk) >= 3]


def world_boundary_patch() -> PathPatch:
    lat_edge = np.linspace(-90, 90, 181)
    lon_edge = np.linspace(-180, 180, 361)
    left = np.column_stack(robinson(np.full_like(lat_edge, -180), lat_edge))
    top = np.column_stack(robinson(lon_edge, np.full_like(lon_edge, 90)))
    right = np.column_stack(robinson(np.full_like(lat_edge, 180), lat_edge[::-1]))
    bottom = np.column_stack(robinson(lon_edge[::-1], np.full_like(lon_edge, -90)))
    verts = np.vstack([left, top, right, bottom, left[:1]])
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(verts) - 2) + [MplPath.CLOSEPOLY]
    return PathPatch(MplPath(verts, codes), facecolor="#eaf4fb", edgecolor="#3c3c3c", lw=0.7)


def projected_land_collection(parts: list[np.ndarray]) -> PolyCollection:
    polys = []
    for ring in parts:
        x, y = robinson(ring[:, 0], ring[:, 1])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() >= 3:
            polys.append(np.column_stack([x[ok], y[ok]]))
    return PolyCollection(
        polys,
        facecolors="#d8d3cc",
        edgecolors="#444444",
        linewidths=0.25,
        zorder=3,
    )


def graticule_collection() -> LineCollection:
    lines = []
    for lon in np.arange(-180, 181, 60):
        lat = np.linspace(-90, 90, 361)
        x, y = robinson(np.full_like(lat, lon), lat)
        lines.append(np.column_stack([x, y]))
    for lat in np.arange(-60, 61, 30):
        lon = np.linspace(-180, 180, 721)
        x, y = robinson(lon, np.full_like(lon, lat))
        lines.append(np.column_stack([x, y]))
    return LineCollection(lines, colors="#8a8a8a", linewidths=0.28, linestyles="--", alpha=0.45, zorder=2)


def finite_scale(values: pd.Series, col: str, fixed_min: float | None, fixed_max: float | None) -> tuple[float, float]:
    data = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if col == "inv_age":
        data = data[data < 999]
    if fixed_min is not None and fixed_max is not None:
        return fixed_min, fixed_max
    if data.empty:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(data.to_numpy(), [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo = float(data.min())
        hi = float(data.max())
    if np.isclose(lo, hi):
        hi = lo + 1.0
    return float(lo), float(hi)


def safe_name(text: str) -> str:
    return (
        text.replace("CRUST1.0", "CRUST1_0")
        .replace(".5deg", "_5deg")
        .replace(".", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def draw_map(
    df: pd.DataFrame,
    spec: MapSpec,
    index: int,
    land_parts: list[np.ndarray],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> tuple[Path, Path, dict[str, float]]:
    vmin, vmax = finite_scale(df[spec.col], spec.col, spec.vmin, spec.vmax)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    cmap = plt.get_cmap(spec.cmap).copy()
    cmap.set_bad("#f1f1f1")

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    boundary = world_boundary_patch()
    ax.add_patch(boundary)
    boundary_path = boundary.get_path()
    boundary_transform = boundary.get_transform()

    gridlines = graticule_collection()
    gridlines.set_clip_path(boundary_path, boundary_transform)
    ax.add_collection(gridlines)

    values = pd.to_numeric(df[spec.col], errors="coerce").to_numpy()
    valid = np.isfinite(values)
    scatter = ax.scatter(
        grid_x[valid],
        grid_y[valid],
        c=values[valid],
        s=1.3,
        cmap=cmap,
        norm=norm,
        alpha=0.88,
        linewidths=0,
        zorder=4,
        rasterized=True,
    )
    scatter.set_clip_path(boundary_path, boundary_transform)

    land = projected_land_collection(land_parts)
    land.set_clip_path(boundary_path, boundary_transform)
    ax.add_collection(land)

    cbar = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.035, fraction=0.045, aspect=45, extend="both")
    cbar.set_label(spec.colorbar, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(spec.title, fontsize=13, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-1.43, 1.43)
    ax.axis("off")
    fig.tight_layout(pad=0.6)

    file_stem = f"map_{index:02d}_{safe_name(spec.col)}"
    png_path = OUT_DIR / f"{file_stem}.png"
    svg_path = OUT_DIR / f"{file_stem}.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    return png_path, svg_path, {
        "index": index,
        "feature": spec.col,
        "n_points": int(valid.sum()),
        "vmin": vmin,
        "vmax": vmax,
        "min": float(np.nanmin(values[valid])),
        "max": float(np.nanmax(values[valid])),
        "mean": float(np.nanmean(values[valid])),
    }


def draw_panel(
    df: pd.DataFrame,
    specs: list[MapSpec],
    land_parts: list[np.ndarray],
    grid_x: np.ndarray,
    grid_y: np.ndarray,
) -> tuple[Path, Path]:
    nrows, ncols = 5, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 28), dpi=300)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    boundary = world_boundary_patch()

    for ax, spec in zip(axes, specs):
        vmin, vmax = finite_scale(df[spec.col], spec.col, spec.vmin, spec.vmax)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        cmap = plt.get_cmap(spec.cmap).copy()
        cmap.set_bad("#f1f1f1")

        ax.set_facecolor("white")
        boundary = world_boundary_patch()
        ax.add_patch(boundary)
        boundary_path = boundary.get_path()
        boundary_transform = boundary.get_transform()

        gridlines = graticule_collection()
        gridlines.set_clip_path(boundary_path, boundary_transform)
        ax.add_collection(gridlines)

        values = pd.to_numeric(df[spec.col], errors="coerce").to_numpy()
        valid = np.isfinite(values)
        sc = ax.scatter(
            grid_x[valid],
            grid_y[valid],
            c=values[valid],
            s=0.7,
            cmap=cmap,
            norm=norm,
            alpha=0.88,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
        sc.set_clip_path(boundary_path, boundary_transform)

        land = projected_land_collection(land_parts)
        land.set_clip_path(boundary_path, boundary_transform)
        ax.add_collection(land)

        cbar = fig.colorbar(
            sc,
            ax=ax,
            orientation="horizontal",
            pad=0.04,
            fraction=0.06,
            aspect=28,
            extend="both",
        )
        cbar.set_ticks([])
        cbar.outline.set_linewidth(0.4)
        cbar.ax.tick_params(length=0)
        if spec.caption:
            cbar.ax.text(
                0.5,
                -1.65,
                spec.caption,
                transform=cbar.ax.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                fontweight="normal",
                clip_on=False,
            )

        ax.set_aspect("equal")
        ax.set_xlim(-2.75, 2.75)
        ax.set_ylim(-1.43, 1.43)
        ax.axis("off")

    for ax in axes[len(specs):]:
        ax.axis("off")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.995, bottom=0.035, wspace=0.02, hspace=0.12)
    png_path = OUT_DIR / "mdi_top14_spatial_panel.png"
    svg_path = OUT_DIR / "mdi_top14_spatial_panel.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.10)
    fig.savefig(svg_path, format="svg", dpi=300, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)
    return png_path, svg_path


def main() -> None:
    print("Loading training data...")
    train = pd.read_csv(DATA_PATH).dropna(subset=BASE_COLS + [TARGET])
    train = add_engineered_features(train, fill_age_nan=True).dropna(subset=TOP14 + [TARGET])
    print(f"  Training records: {len(train):,}")

    print("Training ExtraTrees on MDI Top14...")
    MODEL.fit(train[TOP14].to_numpy(), train[TARGET].to_numpy())

    print("Loading global grid...")
    usecols = ["lon", "lat"] + BASE_COLS
    grid = pd.read_csv(GRID_PATH, usecols=usecols)
    n_age_nan = int(grid["oceanic_crust_age_Ma"].isna().sum())
    grid = add_engineered_features(grid, fill_age_nan=True)
    grid = grid.dropna(subset=TOP14).copy()
    print(f"  Global grid records: {len(grid):,}")
    print(f"  Age NaNs encoded as -1 before transform: {n_age_nan:,}")

    grid["q_pred_mWm2"] = MODEL.predict(grid[TOP14].to_numpy())
    prediction_csv = OUT_DIR / "mdi_top14_prediction_grid.csv"
    grid[["lon", "lat", "q_pred_mWm2"] + TOP14].to_csv(prediction_csv, index=False)
    print(f"  Saved prediction grid: {prediction_csv}")

    print("Preparing Robinson projection and land polygons...")
    grid_x, grid_y = robinson(grid["lon"].to_numpy(), grid["lat"].to_numpy())
    land_parts = read_polygon_parts(LAND_SHP)
    print(f"  Land polygon parts: {len(land_parts):,}")

    summary_rows = []
    for idx, spec in enumerate(MAP_SPECS):
        print(f"Drawing {idx:02d}: {spec.col}")
        png, svg, stats = draw_map(grid, spec, idx, land_parts, grid_x, grid_y)
        stats["png"] = str(png)
        stats["svg"] = str(svg)
        summary_rows.append(stats)

    print("Drawing panel figure...")
    panel_png, panel_svg = draw_panel(grid, MAP_SPECS, land_parts, grid_x, grid_y)
    print(f"  Saved panel PNG: {panel_png}")
    print(f"  Saved panel SVG: {panel_svg}")

    summary_csv = OUT_DIR / "mdi_top14_spatial_map_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
