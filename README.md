# Marine Heat Flow ML Reliability Code

This repository contains the analysis code for the manuscript:

**Assessing the reliability of machine learning-based global marine heat flow prediction with real observations and spatial validation**

The code evaluates machine-learning predictions of global marine heat flow with an emphasis on real observation labels versus Kriging-derived labels, random versus spatial validation, cross-basin validation, residual spatial autocorrelation, SHAP interpretation, and feature-sensitivity experiments.

## What Is Included

- `01_Data_Processing/`: scripts for filtering IHFC/GHFDB heat-flow observations, applying a land mask, gridding observations, aligning geophysical features, and constructing the main observation-level dataset.
- `02_Main_Experiment_SchemeB/`: main real-observation-label experiments, including model comparison, random split, spatial block validation, cross-basin validation, residual autocorrelation, SHAP, feature importance, and feature sensitivity.
- `03_Label_Strategy_Comparison/`: comparison of global Kriging labels, direct observation labels, and local Kriging labels.
- `04_Supplementary/`: supplementary analyses for resolution, prediction maps, uncertainty, label fairness control, block robustness, cross-validation residual mapping, and spatial maps of important features.
- `config/example_config.yml`: documented paths and key parameters.
- `data/README.md`: required data files and source notes.

No raw data, processed data, generated figures, result tables, manuscript drafts, or presentation files are included in this code-only release.


## Installation

Conda is recommended because geospatial packages such as `cartopy` and `geopandas` can be difficult to install with pip alone.

```bash
conda env create -f environment.yml
conda activate marine-heatflow-ml
```

Alternatively:

```bash
python -m pip install -r requirements.txt
```

## Data Setup

Place required input files under `data/` as described in `data/README.md`.

Expected examples:

```text
data/raw/IHFC_2024_GHFDB.csv
data/raw/IHFC_2024_GHFDB_v.2026.03.txt
data/raw/Muller_etal_2019_Tectonics_v2.0_PresentDay_AgeGrid.nc
data/features/Ocean_HeatFlow_Prediction_Data_with_Age.csv
data/natural_earth/ne_10m_land.shp
```

Third-party datasets must be obtained from their original sources. Redistribution permissions were not assumed.

## Main Workflow

Run from the repository root.

```bash
python 01_Data_Processing/step1_build_obs_grid.py
python 01_Data_Processing/step2_3_align_features.py
python 02_Main_Experiment_SchemeB/step7_export_dataset.py

python 02_Main_Experiment_SchemeB/step7_batch1_core.py
python 02_Main_Experiment_SchemeB/step7_batch2_feat_eng.py
python 02_Main_Experiment_SchemeB/step7_batch2_shap.py

python 03_Label_Strategy_Comparison/step14_label_strategy_comparison.py
python 03_Label_Strategy_Comparison/step15_self_evaluation.py --split spatial
python 03_Label_Strategy_Comparison/step15_self_evaluation.py --split random
python 03_Label_Strategy_Comparison/step15_split_comparison.py
```

Selected supplementary scripts can be run independently after `data/processed/dataset_D_no_aggregation.csv` is available.

```bash
python 04_Supplementary/step10_resolution_comparison.py --res 1.0
python 04_Supplementary/step11_prediction_maps.py
python 04_Supplementary/step13_uncertainty.py
python 04_Supplementary/step16_W1_label_fairness_control.py
python 04_Supplementary/step17_W2_block_robustness_extended.py
python 04_Supplementary/step18_mdi_top14_spatial_maps.py
python 04_Supplementary/step_cv_residual_map.py
```

## Reproducibility Scope

This is a code-only public release. It supports transparent inspection and rerunning of the main analyses after the required third-party data files are obtained and placed in the documented paths. It is not a one-command reproduction package because the data are not redistributed.

## License

Code is released under the MIT License. Data files retain the licenses and terms of their original providers.
