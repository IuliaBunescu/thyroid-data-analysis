# Thyroid Analysis Dashboard
This repository contains exploration, cleaning, visualization, and a Streamlit dashboard built around the UCI "Thyroid Disease" dataset (most extensive variant, made from multiple datasets). 

## Scope
Understanding thyroid disease through data analysis, with a focus on:
- Data cleaning and preprocessing techniques for clinical datasets with systematic missing data handling.
- Advanced imputation strategies comparing KNN, MICE, Mean, and Median methods based on correlation preservation.
- Exploratory data analysis (EDA) with comprehensive missing data visualization and analysis.
- Interactive visualization of hormone levels against clinical reference ranges.
- Feature selection and encoding optimized for medical data to avoid target leakage.
- Predictive insights for individual patients based on lab results.

## Dataset (source)
- Name: Thyroid Disease
- Source: UCI Machine Learning Repository (Quinlan, Ross, 1986)
- Records: ~9k (original archive: 9172 records spanning 1984–1987)
- Attributes: 29 fields including clinical flags (t/f), lab measurements (TSH, T3, TT4, T4U, FTI, TBG), demographics (age, sex), referral source, and raw diagnosis codes.
- Missing values: Unknowns are encoded as "?" in the raw file and are parsed to NaN here.
- Known issues observed: extreme age outliers (e.g., 65526), many measured-flag redundancy columns, class imbalance in diagnoses.

## Key files/folders
- app.py — Streamlit dashboard (entrypoint)
- requirements.txt — Python dependencies for the app
- LICENSE — project license (MIT)
- data/ — packaged CSVs used by the app:
  - thyroid_data.csv (cleaned main dataset)
  - condition_codes.csv (mapping of diagnosis codes)
  - lab_reference_intervals.csv (clinical reference ranges)
  - thyroid_data_target.csv (simplified target categories)
- source/ — app helper modules (config, sidebar, utils, tabs)
- .streamlit/, assets/ — app config and static assets

## Data processing highlights
- **Diagnosis parsing**: Raw diagnosis string split into primary and secondary codes, and patient_id extracted.
- **Boolean flags**: Converted 't'/'f' to True/False for easier filtering.
- **Age cleaning**: Rows with age > 100 are examined and removed by default as likely data entry errors.
- **Advanced imputation strategy**:
  - TBG: Dropped due to >90% missing data and MNAR nature, retained TBG_measured flag
  - Sex: Imputed with most frequent category
  - Secondary condition: Filled with '-' (no secondary condition)
  - Numerical blood features (TSH, T3, TT4, T4U, FTI): Compared KNN vs Iterative (MICE) vs Mean vs Median imputation and selected method with smallest correlation structure change
- **Missing data analysis**: Interactive visualizations including missing counts, percentages, missingness matrices, and correlation patterns between missing values.
- **Target engineering**: Created simplified diagnostic categories from complex condition codes for balanced modeling.
- **Feature selection**: Systematic removal of features causing target leakage (diagnosis codes, categories) and redundant measured flags.
- **Feature encoding**: Boolean to binary conversion, categorical encoding while preserving medical interpretability.
- **Feature importance analysis**: Random Forest, Permutation Importance, and Mutual Information rankings with PCA visualization.

## App features
- **IDA Tab**: Interactive missing data analysis with Plotly visualizations, imputation comparison, and data quality assessment.
- **EDA Tab**: Comprehensive exploratory analysis including target distribution, correlation heatmaps, feature importance, and PCA with medical context.
- **Info Tab**: Dataset documentation, lab reference ranges, and methodology explanations.

## Deployment
- Live demo: [Streamlit Cloud](https://juliab-thyroid-data-analysis.streamlit.app/)

## Screenshots
- IDA Tab:
  ![IDA Tab](https://github.com/IuliaBunescu/thyroid-data-analysis/blob/main/screenshots/ida-tab.png)
- EDA tab:
  ![EDA Tab](https://github.com/IuliaBunescu/thyroid-data-analysis/blob/main/screenshots/eda-tab.png)
- Info Tab:
  ![Info Tab](./screenshots/info_tab.png)

## Quick start (local)
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

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