import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from source.config import (
    AXIS_TICK_FONT_SIZE,
    AXIS_TITLE_FONT_SIZE,
    CONTINUOUS_COLOR_SCALE,
    DISCRETE_COLOR_PALETTE,
    TITLE_FONT_SIZE,
)


def general_eda_structure(
    df: pd.DataFrame, target_df: pd.DataFrame, condition_codes: pd.DataFrame = None
):
    """
    General structure for the EDA tab
    Inputs:
    - df: DataFrame containing thyroid data
    """
    st.header("Target Analysis")
    target_exploration(df, target_df, condition_codes)
    st.markdown("---")

    st.header("Multivariate Analysis")
    multivariate_analysis(df, target_df)
    st.markdown("---")

    st.header("Correlation Analysis")
    correlation_analysis(df, target_df)
    st.markdown("---")

    st.header("Imputation")
    df_imputed = imputation(df)

    st.markdown("---")
    st.header("Encoding & Feature Selection")
    feature_selection_and_encoding(df_imputed, target_df)


def target_exploration(
    df: pd.DataFrame, target_df: pd.DataFrame, condition_codes: pd.DataFrame = None
):
    """
    Explore the target variable in the dataset
    Inputs:
    - df: DataFrame containing thyroid data
    - target_df: DataFrame containing target variable
    """
    st.subheader("Understanding Thyroid Primary Conditions")
    col1, col2 = st.columns([2, 3])

    col1.dataframe(condition_codes)

    with col2:

        st.subheader("Replacement vs Antithyroid Treatment Diagnoses")

        cat = df["Category"].astype(str)

        rep_mask = cat.str.contains("Replacement", case=False, na=False)
        anti_mask = cat.str.contains("Antithyroid", case=False, na=False)

        rep_count = int(rep_mask.sum())
        anti_count = int(anti_mask.sum())
        total = rep_count + anti_count
        rep_pct = (rep_count / total * 100) if total > 0 else 0.0
        anti_pct = (anti_count / total * 100) if total > 0 else 0.0

        mcol1, mcol2 = st.columns([1, 1])
        mcol1.metric(
            "Replacement ",
            rep_count,
            f"{rep_pct:.1f}% of dataset",
            border=True,
            help="How many patients with a treatment diagnosis requiring hormone replacement.",
        )
        mcol2.metric(
            "Antithyroid ",
            anti_count,
            f"{anti_pct:.1f}% of dataset",
            border=True,
            help="How many patients diagnosed with conditions requiring antithyroid treatment.",
        )

        st.subheader("Belief vs Actual Hypothyroid/ Hyperthyroid Diagnosis")

        hypo_mask = cat.str.contains("Hypothyroid", case=False, na=False)
        hyper_mask = cat.str.contains("Hyperthyroid", case=False, na=False)

        hypo_count = int(hypo_mask.sum())
        hyper_count = int(hyper_mask.sum())

        # Belief counts (safe checks for boolean query columns)
        belief_hypo_count = 0
        belief_hyper_count = 0
        if "query_hypothyroid" in df.columns:
            belief_hypo_count = int(df["query_hypothyroid"].astype(bool).sum())
        if "query_hyperthyroid" in df.columns:
            belief_hyper_count = int(df["query_hyperthyroid"].astype(bool).sum())

        # Differences: Belief − Actual
        diff_hypo = belief_hypo_count - hypo_count
        diff_hypo_pct = (diff_hypo / hypo_count * 100) if hypo_count > 0 else 0.0

        diff_hyper = belief_hyper_count - hyper_count
        diff_hyper_pct = (diff_hyper / hyper_count * 100) if hyper_count > 0 else 0.0

        dcol1, dcol2 = st.columns([1, 1])
        dcol1.metric(
            "Hypothyroid: Belief − Actual ",
            diff_hypo,
            f"{diff_hypo_pct:.1f}% of data",
            border=True,
            help="How many of the patients believed to have hypothyroidism actually have it according to the data",
        )
        dcol2.metric(
            "Hyperthyroid: Belief − Actual ",
            diff_hyper,
            f"{diff_hyper_pct:.1f}% of data",
            border=True,
            help="How many of the patients believed to have hyperthyroidism actually have it according to the data",
        )

    st.info(
        "There are quite a few different thyroid conditions represented in the dataset, and even the *Category* feature has maybe values we are not necessarily interested in for this analysis."
        " For simplicity, we will group them into broader categories and define a new **target** variable. The categories are:"
        "\n"
        "- Hyperthyroid: A, B, C, D\n"
        "- Hypothyroid: E, F, G, H\n"
        "- Binding Protein: I, J\n"
        "- General Health: K\n"
        "- Discordant Results: R\n"
        "- Elevated Hormones: S, T\n"
        "- Normal: None\n"
    )
    st.subheader("Target Variable Distribution")

    target_counts = target_df["target"].value_counts().reset_index()
    target_counts.columns = ["Target", "Count"]

    fig = px.bar(
        target_counts,
        x="Target",
        y="Count",
        title="Distribution of Target Variable",
        labels={"Count": "Number of Patients", "Target": "Thyroid Condition"},
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        hoverlabel=dict(font=dict(size=AXIS_TICK_FONT_SIZE)),
        xaxis=dict(
            title_font=dict(size=AXIS_TITLE_FONT_SIZE),
            tickfont=dict(size=AXIS_TICK_FONT_SIZE),
        ),
        yaxis=dict(
            title_font=dict(size=AXIS_TITLE_FONT_SIZE),
            tickfont=dict(size=AXIS_TICK_FONT_SIZE),
        ),
        title=dict(font=dict(size=TITLE_FONT_SIZE)),
    )
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "The target is visibly unbalanced, with most patients being labeled as *normal* and having no thyroid condition. To account for this, resampling techniques will be implemented when training the model. No changes will be made to the dataset to account for the imbalance at this step."
    )


@st.fragment
def multivariate_analysis(
    df: pd.DataFrame, target_df: pd.DataFrame = None, target_col: str = "target"
):
    """
    Perform multivariate analysis on the dataset and render several plots in Streamlit.
    - If target_df is provided (shares the same index), it will be joined into the plotting frame.
    - Automatically picks numeric features (top by variance) for plotting.
    - Creates: scatter matrix, pairwise scatter plots, violin distributions by target, and correlation heatmap.
    - Allows the user to pick a grouping (color) column: the target OR any boolean / categorical column.
    """
    # Only pairwise scatter plots: let user pick a single Y numeric feature to compare against others
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric features available for multivariate plots.")
        return

    # Prepare plotting frame and attach target if provided
    df_plot = df.copy()
    if target_df is not None:
        if isinstance(target_df, pd.Series):
            series = target_df.reindex(df.index)
        else:
            if target_col in target_df.columns:
                series = target_df[target_col].reindex(df.index)
            elif target_df.shape[1] == 1:
                series = target_df.iloc[:, 0].reindex(df.index)
            else:
                series = None
        if series is not None:
            df_plot[target_col] = series.astype(str)

    # Build list of candidate grouping columns: include target if present, booleans, and categoricals/objects
    candidates = []
    if target_col in df_plot.columns:
        candidates.append(target_col)
    for c in df_plot.columns:
        dt = df_plot[c].dtype
        if dt == bool or dt.name == "category" or dt == object:
            if c not in candidates:
                candidates.append(c)

    group_options = ["None"] + candidates
    # Default selection: target_col if available, else None
    default_idx = 1 if target_col in candidates else 0
    group_by = st.selectbox(
        "Group / color by (choose target or any boolean/categorical feature)",
        group_options,
        index=default_idx,
    )

    color_arg = group_by if group_by != "None" else None

    st.subheader("Pairwise Scatter Plots")

    numerical_pairwise_fragment(df_plot, color_arg, numeric_cols)


@st.fragment
def correlation_analysis(
    df: pd.DataFrame, target_df: pd.DataFrame = None, target_col: str = "target"
):
    """
    Perform correlation analysis on the dataset.
    Allows grouping by target or any boolean/categorical column, and optionally viewing per-group correlations.
    """
    df_plot = df.copy()
    if target_df is not None:
        if isinstance(target_df, pd.Series):
            series = target_df.reindex(df.index)
        else:
            if target_col in target_df.columns:
                series = target_df[target_col].reindex(df.index)
            elif target_df.shape[1] == 1:
                series = target_df.iloc[:, 0].reindex(df.index)
            else:
                series = None
        if series is not None:
            df_plot[target_col] = series

    numeric_cols = df_plot.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric features available for correlation analysis.")
        return

    # Candidate grouping columns (same logic as multivariate)
    candidates = []
    if target_col in df_plot.columns:
        candidates.append(target_col)
    for c in df_plot.columns:
        dt = df_plot[c].dtype
        if dt == bool or dt.name == "category" or dt == object:
            if c not in candidates:
                candidates.append(c)

    group_options = ["None"] + candidates
    default_idx = 1 if target_col in candidates else 0
    group_by = st.selectbox(
        "Group correlations by (None or choose a boolean/categorical column)",
        group_options,
        index=default_idx,
    )

    st.subheader("Correlation Matrix")

    # If no grouping, just show overall correlation
    if group_by == "None":
        _plot_corr(df_plot, " (overall)", numeric_cols)
        return

    # If grouping selected, show overall and provide option to inspect per-group correlations
    _plot_corr(df_plot, " (overall)", numeric_cols)

    # Show available group values (limit to reasonable count)
    unique_vals = df_plot[group_by].dropna().unique().tolist()
    if not unique_vals:
        st.info(f"No values found for grouping column '{group_by}'.")
        return

    choice_val = st.selectbox(
        f"Select value of '{group_by}' to view subgroup correlations",
        unique_vals,
        index=0,
    )
    sub_df = df_plot[df_plot[group_by] == choice_val]
    if sub_df.empty:
        st.info("No rows for the selected subgroup.")
        return
    _plot_corr(sub_df, f" ({group_by} = {choice_val})", numeric_cols)


def _plot_corr(sub_df, title_suffix="", numeric_cols=None):
    corr = sub_df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".3f",
        color_continuous_scale=CONTINUOUS_COLOR_SCALE,
        zmin=-1,
        zmax=1,
        labels=dict(x="Feature", y="Feature", color="Correlation"),
        title=f"Correlation matrix{title_suffix}",
        height=600,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis=dict(tickfont=dict(size=AXIS_TICK_FONT_SIZE)),
        yaxis=dict(tickfont=dict(size=AXIS_TICK_FONT_SIZE)),
        title=dict(font=dict(size=TITLE_FONT_SIZE)),
    )
    st.plotly_chart(fig, width="stretch")


def imputation(df: pd.DataFrame):
    """
    Imputation strategy for thyroid dataset:
    - TBG: Drop due to >90% missing data and MNAR nature, but keep TBG_measured flag
    - sex: Impute with most frequent category
    - condition_secondary: Impute with '-' (no secondary condition)
    - Numerical blood features (TSH, T3, TT4, T4U, FTI): Compare KNN vs Iterative (MICE)
      vs Mean vs Median imputation and choose the method that least alters the correlation structure

    Inputs:
        - df:

    """

    df_impute = df.copy()

    # 1. Handle TBG column - drop if >90% missing
    missing_pct = df_impute["TBG"].isna().mean() * 100
    df_impute = df_impute.drop(columns=["TBG"])

    # 2. Impute sex with mode
    mode_sex = df_impute["sex"].mode()
    mode_value = mode_sex.iloc[0]
    df_impute["sex"] = df_impute["sex"].fillna(mode_value)

    # 3. Impute condition_secondary with '-'
    df_impute["condition_secondary"] = df_impute["condition_secondary"].fillna("-")

    # 4. Handle numerical blood test features
    blood_features = ["TSH", "T3", "TT4", "T4U", "FTI"]

    # Calculate original correlation matrix for comparison
    original_corr = df_impute[blood_features].corr().fillna(0).values

    # Prepare results storage
    imputation_results = {}
    correlation_changes = {}

    # Try KNN Imputation
    try:
        knn_imputer = KNNImputer(n_neighbors=5)
        blood_data_knn = knn_imputer.fit_transform(df_impute[blood_features])
        knn_corr = pd.DataFrame(blood_data_knn, columns=blood_features).corr().values
        knn_change = np.linalg.norm(knn_corr - original_corr)

        imputation_results["KNN"] = blood_data_knn
        correlation_changes["KNN"] = knn_change
    except Exception as e:
        correlation_changes["KNN"] = np.inf

    # Try Iterative (MICE) Imputation
    try:
        mice_imputer = IterativeImputer(random_state=42, max_iter=30, tol=1e-3)
        blood_data_mice = mice_imputer.fit_transform(df_impute[blood_features])
        mice_corr = pd.DataFrame(blood_data_mice, columns=blood_features).corr().values
        mice_change = np.linalg.norm(mice_corr - original_corr)

        imputation_results["MICE"] = blood_data_mice
        correlation_changes["MICE"] = mice_change
    except Exception as e:
        correlation_changes["MICE"] = np.inf

    # Try Mean Imputation
    try:
        blood_data_mean = df_impute[blood_features].copy()
        for col in blood_features:
            mean_val = df_impute[col].mean()
            blood_data_mean[col] = df_impute[col].fillna(mean_val)

        mean_corr = blood_data_mean.corr().values
        mean_change = np.linalg.norm(mean_corr - original_corr)

        imputation_results["Mean"] = blood_data_mean.values
        correlation_changes["Mean"] = mean_change
    except Exception as e:
        correlation_changes["Mean"] = np.inf

    # Try Median Imputation
    try:
        blood_data_median = df_impute[blood_features].copy()
        for col in blood_features:
            median_val = df_impute[col].median()
            blood_data_median[col] = df_impute[col].fillna(median_val)

        median_corr = blood_data_median.corr().values
        median_change = np.linalg.norm(median_corr - original_corr)

        imputation_results["Median"] = blood_data_median.values
        correlation_changes["Median"] = median_change
    except Exception as e:
        correlation_changes["Median"] = np.inf

    # Choose best method
    valid_methods = {k: v for k, v in correlation_changes.items() if np.isfinite(v)}

    if not valid_methods:
        # Ultimate fallback - use mean imputation
        df_final = df_impute.copy()
        for col in blood_features:
            mean_val = df_impute[col].mean()
            df_final[col] = df_impute[col].fillna(mean_val)
        chosen_method = "Mean (fallback)"
        correlation_changes["Mean (fallback)"] = "N/A (ultimate fallback)"
    else:
        # Choose method with smallest correlation change
        chosen_method = min(valid_methods, key=valid_methods.get)
        chosen_data = imputation_results[chosen_method]

        df_final = df_impute.copy()
        df_final[blood_features] = chosen_data

    # Store results in session state
    st.session_state["imputed_df"] = df_final
    st.session_state["imputation_method"] = chosen_method
    st.session_state["correlation_changes"] = correlation_changes

    # Create comparison table for transparency
    st.subheader("Imputation Method Comparison (Continuous Numeric Features)")
    comparison_data = []
    for method, change in correlation_changes.items():
        if method == chosen_method:
            status = "✓ Selected"
        else:
            status = "Not selected"

        if isinstance(change, (int, float)) and np.isfinite(change):
            change_str = f"{change:.6f}"
        else:
            change_str = str(change)

        comparison_data.append(
            {
                "Method": method,
                "Pearson Correlation Change (Frobenius Norm)": change_str,
                "Status": status,
            }
        )

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, width="stretch")

    st.caption(
        "Lower correlation change indicates better preservation of the original correlation structure between blood test features."
    )

    # Final summary
    st.write(
        f"**Imputation Summary:** \n"
        "- **TBG**: Dropped due to MNAR nature and {missing_pct:.1f}% missing data. TBG_measured flag retained. \n"
        "- **Sex**: Filled missing values with most frequent category: '{mode_value}' \n"
        "- **Secondary condition**: Filled with '-' (no secondary condition). Note: this feature was not used for target creation and will be dropped later. \n"
        "- **Blood test features** (TSH, T3, TT4, T4U, FTI): Compared KNN vs Iterative (MICE) vs Mean vs Median imputation and selected **{chosen_method}** as it produced the smallest change in pairwise correlation structure."
    )

    return df_final


def numerical_pairwise_fragment(
    df_plot: pd.DataFrame, color_arg: str = None, numeric_cols: list = None
):
    selected_y = st.selectbox(
        "Select Y-axis numeric feature for comparison", numeric_cols, index=0
    )

    x_features = [c for c in numeric_cols if c != selected_y]
    if not x_features:
        st.info("No other numeric features to compare with the selected Y feature.")
        return

    # Limit number of comparison plots shown
    max_plots = min(6, len(x_features))
    x_features = x_features[:max_plots]

    # Render plots two per row
    for i in range(0, len(x_features), 2):
        row_cols = st.columns(2)
        for col_idx, x_feat in enumerate(x_features[i : i + 2]):
            fig = px.scatter(
                df_plot,
                x=x_feat,
                y=selected_y,
                color=color_arg,
                labels={x_feat: x_feat, selected_y: selected_y},
                title=f"{selected_y} vs {x_feat}",
                color_discrete_sequence=DISCRETE_COLOR_PALETTE,
                marginal_x="violin",
                marginal_y="violin",
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                hoverlabel=dict(font=dict(size=AXIS_TICK_FONT_SIZE)),
                xaxis=dict(
                    title_font=dict(size=AXIS_TITLE_FONT_SIZE),
                    tickfont=dict(size=AXIS_TICK_FONT_SIZE),
                ),
                yaxis=dict(
                    title_font=dict(size=AXIS_TITLE_FONT_SIZE),
                    tickfont=dict(size=AXIS_TICK_FONT_SIZE),
                ),
                title=dict(font=dict(size=TITLE_FONT_SIZE)),
                legend=dict(font=dict(size=AXIS_TICK_FONT_SIZE)),
            )
            row_cols[col_idx].plotly_chart(fig, width="stretch")


def feature_selection_and_encoding(df: pd.DataFrame, target_df: pd.DataFrame = None):
    """
    Simplified encoder:
    - Booleans -> 0/1
    - Categorical columns are reduced to at most 10 categories (top 9 + OTHER/MISSING),
      then one-hot encoded.
    After encoding show variance/PCA and (if target provided) simple supervised importances.
    Encoded dataframe is stored in st.session_state['encoded_df'] and offered as CSV.
    """
    st.subheader("Feature Selection")

    st.write(
        "Some of the features are useful for visualisations but might not be that useful for modeling directly. Because of this a few features have been dropped before encoding:\n"
        "- *Category*, *condition_primary*, *condition_secondary*: they would lead to target leakage if included in modeling.\n"
        "- *referral_source*: not very clear how this is medically relevant, safe to assume it is not important enough to be part of the modeling dataset.\n"
        "- *T3_measured*, *T4U_measured*, *FTI_measured*, *TSH_measured*, *TT4_measured*: these are just indicators of whether the corresponding test was performed, which is already captured by the presence of the actual test value. They might be useful for a more complex model ensemble approach, but for simplicity we drop them here.\n"
        " - *TBG*: dropped due to high missingness, however due to its MNAR nature *TBG_measured* is retained to indicate whether the test was performed."
    )
    df_dropped = df.copy()
    df_dropped = df.drop(
        columns=[
            "Category",
            "condition_primary",
            "condition_secondary",
            "referral_source",
            "T3_measured",
            "T4U_measured",
            "FTI_measured",
            "TSH_measured",
            "TBG",
            "TT4_measured",
        ],
        errors="ignore",
    )

    # Resolve target series if provided (aligned to df index)
    target_series = None
    if target_df is not None:
        if isinstance(target_df, pd.Series):
            target_series = target_df.reindex(df_dropped.index)
        else:
            if "target" in target_df.columns:
                target_series = target_df["target"].reindex(df_dropped.index)
            elif target_df.shape[1] == 1:
                target_series = target_df.iloc[:, 0].reindex(df_dropped.index)
            else:
                target_series = None

    df_encoded = df_dropped.copy()

    # Convert booleans to 0/1
    bool_cols = df_encoded.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    # If 'sex' exists, convert to binary: female -> 1, male -> 0 (robust to case/whitespace)
    if "sex" in df_encoded.columns:
        s = df_encoded["sex"].fillna("").astype(str).str.strip().str.lower()
        df_encoded["sex"] = np.where(
            s.str.startswith("f"), 1, np.where(s.str.startswith("m"), 0, np.nan)
        ).astype(float)

    # Save encoded df to session and preview
    st.session_state["encoded_df"] = df_encoded
    st.subheader("Encoded Data (preview)")
    st.write(
        "**Encoding Summary:**\n"
        "- Boolean features converted to 0/1\n"
        "- Sex feature converted to binary (1 for female, 0 for male)\n"
        "- Features causing target leakage or deemed not useful for modeling have been dropped\n"
        "- Missing values handled in previous imputation step"
    )

    st.dataframe(df_encoded.head(100))

    # Feature importance analysis (only if target is available)
    if target_series is not None:
        st.subheader("Feature Importance Analysis")

        # Align target with encoded features
        y = target_series.reindex(df_encoded.index)
        mask = y.notna()

        X = df_encoded.loc[mask].copy()
        y = y.loc[mask].copy()

        if len(X) == 0:
            st.warning("No valid samples with both features and target available.")
            return

        try:
            model = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
            model.fit(X.values, y.values)

            # Tree-based feature importances
            feat_imp = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=False)
            top_feat_imp = feat_imp.head(20).reset_index()
            top_feat_imp.columns = ["feature", "importance"]
            fig_imp = px.bar(
                top_feat_imp,
                x="feature",
                y="importance",
                title="Model Feature Importances (Tree-based)",
            )
            fig_imp.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_imp, width="stretch")

            # Permutation importance
            p_imp = permutation_importance(
                model, X.values, y.values, n_repeats=3, random_state=0, n_jobs=1
            )
            pser = pd.Series(p_imp.importances_mean, index=X.columns).sort_values(
                ascending=False
            )
            top_p = pser.head(20).reset_index()
            top_p.columns = ["feature", "perm_importance"]
            fig_perm = px.bar(
                top_p,
                x="feature",
                y="perm_importance",
                title="Permutation Importances (Mean)",
            )
            fig_perm.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_perm, width="stretch")

            # Mutual information
            try:
                mi = mutual_info_classif(
                    X.values, y.values, discrete_features="auto", random_state=0
                )
                mi_ser = pd.Series(mi, index=X.columns).sort_values(ascending=False)
                top_mi = mi_ser.head(20).reset_index()
                top_mi.columns = ["feature", "mutual_info"]
                fig_mi = px.bar(
                    top_mi,
                    x="feature",
                    y="mutual_info",
                    title="Mutual Information (Top Features)",
                )
                fig_mi.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_mi, width="stretch")
            except Exception:
                st.info("Mutual information could not be computed in this environment.")

        except Exception as e:
            st.warning(f"Supervised importance computation failed: {e}")

        # PCA visualization
        st.subheader("PCA Visualization")
        try:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X.values)
            pca = PCA(n_components=min(10, X_scaled.shape[1]))
            pcs = pca.fit_transform(X_scaled)
            evr = pca.explained_variance_ratio_

            # Scree plot
            scree = pd.DataFrame(
                {"pc": [f"PC{i+1}" for i in range(len(evr))], "explained_variance": evr}
            )
            fig_scree = px.bar(
                scree,
                x="pc",
                y="explained_variance",
                title="PCA Explained Variance (Scree Plot)",
            )
            st.plotly_chart(fig_scree, width="stretch")

            # PCA scatter plot
            pc_df = pd.DataFrame(pcs[:, :2], columns=["PC1", "PC2"], index=X.index)
            pc_df["target"] = y.astype(str).values
            fig_pca = px.scatter(
                pc_df,
                x="PC1",
                y="PC2",
                color="target",
                title="PCA (PC1 vs PC2) Colored by Target",
                color_discrete_sequence=DISCRETE_COLOR_PALETTE,
            )
            st.plotly_chart(fig_pca, width="stretch")

        except Exception as e:
            st.warning(f"PCA visualization failed: {e}")
    else:
        st.info("No target variable provided - skipping feature importance analysis.")

    # csv = df_encoded.to_csv(index=False)
    # st.download_button(
    #     "Download encoded dataframe (CSV)",
    #     data=csv,
    #     file_name="encoded_df.csv",
    #     mime="text/csv",
    # )
