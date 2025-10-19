# Thyroid Analysis Dashbaord
This repository contains exploration, cleaning, visualization, and a Streamlit dashboard built around the UCI "Thyroid Disease" dataset (most-extensive variant). 
## Scope
Understanding thyroid disease through data analysis, with a focus on:
- Data cleaning and preprocessing techniques for clinical datasets.
- Exploratory data analysis (EDA) with missing data considerations.
- Interactive visualization of hormone levels against clinical reference ranges.
- Predictive insights for individual patients based on lab results.


## Dataset (source)
- Name: Thyroid Disease
- Source: UCI Machine Learning Repository (Quinlan, Ross, 1986)
- Records: ~9k (original archive: 9172 records spanning 1984–1987)
- Attributes: 29 fields including clinical flags (t/f), lab measurements (TSH, T3, TT4, T4U, FTI, TBG), demographics (age, sex), referral source, and raw diagnosis codes.
- Missing values: Unknowns are encoded as "?" in the raw file and are parsed to NaN here.
- Known issues observed: extreme age outliers (e.g., 65526), many measured-flag redundancy columns, class imbalance in diagnoses.

## What is useful in this repo
- Preprocessing and cleaning pipeline (notebook): parsing diagnosis codes, converting boolean flags, dropping implausible ages, and exporting cleaned CSV.
- Missing-data exploration: counts, percentages, a missingness matrix, and missingness-correlation analyses (interactive Plotly and static/matplotlib approaches).
- Lab reference table for common thyroid tests (TSH, T3, TT4, T4U, FTI, TBG) used for clinical-range annotations.
- Streamlit dashboard to explore distributions, compare hormone levels to reference ranges, and perform per-patient interpretation.

## Key files/folders
- app.py — Streamlit dashboard (entrypoint)
- requirements.txt — Python dependencies for the app
- LICENSE — project license (MIT)
- data/ — packaged CSVs used by the app:
  - thyroid_data.csv (cleaned main dataset)
  - condition_codes.csv (mapping of diagnosis codes)
  - lab_reference_intervals.csv (clinical reference ranges)
- source/ — app helper modules (config, sidebar, utils, tabs)
- .streamlit/, assets/ — app config and static assets


## Data processing highlights
- Diagnosis parsing: raw diagnosis string split into primary and secondary codes and patient_id extracted.
- Boolean flags: converted 't'/'f' to True/False for easier filtering.
- Age cleaning: rows with age > 100 are examined and removed by default as likely data entry errors; see notebook to change policy.
- Missingness strategies shown: threshold-based masking, percentile-based masking, tolerance-based approximate divisibility for floats, and sampled visualizations for large datasets.


## Deployment
- Live demo: <[Streamlit Cloud](https://juliab-thyroid-data-analysis.streamlit.app/)>

## Screenshots
Include screenshots of the dashboard here. Example markdown placeholders:
- IDA Tab:
  ![IDA Tab](thyroid-data-analysis\screenshots\ida-tab.png)
- EDA tab:
  ![EDA Tab](./docs/screenshots/eda_tab.png)
- Info Tab:
  ![Info Tab](./docs/screenshots/info_tab.png)

Create the docs/screenshots directory and add PNGs to display them here.

## Dataset Citation
If you use the dataset in publications, cite the original data as:
```bibtex
@misc{thyroid_disease_102,
  author       = {Quinlan, Ross},
  title        = {{Thyroid Disease}},
  year         = {1986},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5D010}
}
```

## License
MIT — see LICENSE file for details.