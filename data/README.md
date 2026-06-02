# Data Files

This code repository does not include raw or processed data. The analysis requires third-party geoscience datasets that should be obtained from their original sources and placed in this directory.

## Expected Directory Layout

```text
data/
├── raw/
│   ├── IHFC_2024_GHFDB.csv
│   ├── IHFC_2024_GHFDB_v.2026.03.txt
│   ├── Muller_etal_2019_Tectonics_v2.0_PresentDay_AgeGrid.nc
│   ├── PB2002_boundaries.json
│   └── PB2002_orogens.json
├── features/
│   ├── Ocean_HeatFlow_Prediction_Data_with_Age.csv
│   ├── Ocean_HeatFlow_Prediction_Data_1x1deg.csv
│   └── Ocean_HeatFlow_Prediction_Data_0.25x0.25deg.csv
├── processed/
│   └── dataset_D_no_aggregation.csv
└── natural_earth/
    ├── ne_10m_land.shp
    └── ne_110m_land.shp
```

The `.shp` files require their companion files (`.dbf`, `.shx`, `.prj`, etc.) in the same folder.

## Data Sources to Confirm

- IHFC/GHFDB heat-flow observations.
- Muller et al. present-day oceanic crustal age grid.
- Natural Earth land polygons for land/ocean masking.
- Geophysical feature grids derived from CRUST1.0, LITHO1.0, EMAG2, topography, volcano/hotspot distance data, and related sources.

The processed feature grids are not regenerated from every original upstream dataset in this code-only release. Authors/users should cite and follow each data provider's license and redistribution terms.
