import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from source.config import (
    CONTINUOUS_COLOR_SCALE,
    CUSTOM_DISCRETE_2VAR_COLOR_PALETTE,
    DISCRETE_COLOR_PALETTE,
    TITLE_FONT_SIZE,
)
from source.utils import apply_standard_layout


def general_eda_structure(
    df: pd.DataFrame, target_df: pd.DataFrame, condition_codes: pd.DataFrame = None
):
    """Render the Exploratory Data Analysis tab layout and content.

    Args:
        df (pandas.DataFrame): Dataset containing thyroid features.
        target_df (pandas.DataFrame): Encoded targets aligned to the dataset index.
        condition_codes (pandas.DataFrame | None): Optional lookup table for condition codes.

    Returns:
        None: Streamlit sections for EDA are rendered sequentially.
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
    st.header("Encoding")
    encoding(df_imputed, target_df)

    st.markdown("---")
    st.header("Feature Selection")
    feature_selection()


def target_exploration(
    df: pd.DataFrame, target_df: pd.DataFrame, condition_codes: pd.DataFrame = None
):
    """Explore the target distribution and related diagnostic insights.

    Args:
        df (pandas.DataFrame): Dataset containing original features and categories.
        target_df (pandas.DataFrame): Table or series describing the derived target labels.
        condition_codes (pandas.DataFrame | None): Optional mapping of condition codes to descriptions.

    Returns:
        None: Visualizations and narrative text are rendered in Streamlit.
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
    apply_standard_layout(fig)
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "The target is visibly unbalanced, with most patients being labeled as *normal* and having no thyroid condition. To account for this, resampling techniques will be implemented when training the model. No changes will be made to the dataset to account for the imbalance at this step."
    )


@st.fragment
def multivariate_analysis(
    df: pd.DataFrame, target_df: pd.DataFrame = None, target_col: str = "target"
):
    """Run interactive multivariate visualizations for numeric features.

    Args:
        df (pandas.DataFrame): Dataset whose features will be analysed pairwise.
        target_df (pandas.DataFrame | None): Optional target information to join for colouring.
        target_col (str): Column name containing the target labels once joined.

    Returns:
        None: Pairwise plots and selection controls are rendered in Streamlit.
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
    """Visualize correlation matrices overall and by subgroup.

    Args:
        df (pandas.DataFrame): Dataset used to compute correlation coefficients.
        target_df (pandas.DataFrame | None): Optional target information for grouping.
        target_col (str): Column name containing the target after merging.

    Returns:
        None: Correlation heatmaps and subgroup controls are rendered in Streamlit.
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

    st.subheader("Pearson Correlation Matrix")

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
    """Render a correlation heatmap for the provided subset of data.

    Args:
        sub_df (pandas.DataFrame): Subset of the dataset used to compute correlations.
        title_suffix (str): Text appended to the plot title to clarify context.
        numeric_cols (list[str] | None): Numeric columns to include in the correlation calculation.

    Returns:
        None: The correlation heatmap is displayed within Streamlit.
    """
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
    apply_standard_layout(fig)
    st.plotly_chart(fig, width="stretch")


def imputation(df: pd.DataFrame):
    """Impute missing values while monitoring correlation structure shifts.

    Args:
        df (pandas.DataFrame): Dataset requiring targeted imputation strategies.

    Returns:
        pandas.DataFrame: Copy of the input data with imputed values applied.
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
        f"- **TBG**: Dropped due to MNAR nature and {missing_pct:.1f}% missing data. TBG_measured flag retained. \n"
        f"- **Sex**: Filled missing values with most frequent category: '{mode_value}' \n"
        f"- **Secondary condition**: Filled with '-' (no secondary condition). Note: this feature was not used for target creation and will be dropped later. \n"
        f"- **Blood test features** (TSH, T3, TT4, T4U, FTI): Compared KNN vs Iterative (MICE) vs Mean vs Median imputation and selected **{chosen_method}** as it produced the smallest change in pairwise correlation structure."
    )

    return df_final


def numerical_pairwise_fragment(
    df_plot: pd.DataFrame, color_arg: str = None, numeric_cols: list = None
):
    """Render scatter comparisons between one selected target numeric and others.

    Args:
        df_plot (pandas.DataFrame): DataFrame used for plotting scatter comparisons.
        color_arg (str | None): Column name used for colouring points, if any.
        numeric_cols (list[str] | None): Numeric columns eligible for selection.

    Returns:
        None: Pairwise scatter plots are shown within Streamlit.
    """
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
            apply_standard_layout(fig)
            row_cols[col_idx].plotly_chart(fig, width="stretch")


@st.fragment
def pca_fragment(X: pd.DataFrame, y: pd.Series, n_components: int = None):
    """Visualize PCA scree plots and biplots for the supplied dataset.

    Args:
        X (pandas.DataFrame): Numeric feature matrix aligned to the observations.
        y (pandas.Series): Target series used for colouring in the biplot.
        n_components (int | None): Number of components to compute; defaults to min(10, n_features).

    Returns:
        None: PCA outputs and supporting tables are rendered in Streamlit.
    """
    if n_components is None:
        n_components = min(10, X.shape[1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(X_scaled)

    # Store PCA scores and model as simple numpy arrays (no extra index objects)
    st.session_state["pca_model"] = pca
    st.session_state["pca_scaled_X"] = pcs

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    svals = pca.singular_values_
    V = pca.components_.T

    # Raw loadings scaled by singular values for PC1/PC2
    loadings_raw = np.column_stack(
        [
            V[:, 0] * (svals[0] if len(svals) > 0 else 1.0),
            V[:, 1] * (svals[1] if len(svals) > 1 else 1.0),
        ]
    )

    # Automatic scale so loadings fit the score cloud
    scores = pcs[:, :2]
    score_max = np.max(np.abs(scores)) if scores.size > 0 else 1.0
    loading_max = np.max(np.abs(loadings_raw)) if loadings_raw.size > 0 else 1.0
    base_scale = 0.9 * score_max / loading_max if loading_max > 0 else 1.0

    # Scree figure (individual + cumulative)
    pcs_idx = list(range(1, len(evr) + 1))
    scree_fig = go.Figure()
    scree_fig.add_trace(
        go.Scatter(
            x=pcs_idx,
            y=evr,
            mode="lines+markers",
            name="Individual",
            marker=dict(color=CUSTOM_DISCRETE_2VAR_COLOR_PALETTE[1], size=8),
        )
    )
    scree_fig.add_trace(
        go.Scatter(
            x=pcs_idx,
            y=cum_evr,
            mode="lines+markers",
            name="Cumulative",
            marker=dict(color=CUSTOM_DISCRETE_2VAR_COLOR_PALETTE[0], size=8),
        )
    )
    scree_fig.update_xaxes(
        title_text="Principal Component", tickmode="array", tickvals=pcs_idx
    )
    scree_fig.update_yaxes(title_text="Proportion of Variance Explained")
    scree_fig.update_layout(title_text="PCA Scree Plot")

    st.plotly_chart(scree_fig, use_container_width=True)

    # Biplot: interactive

    # Slider for interactive scaling
    scale_mult = st.slider(
        "Loadings scale multiplier", min_value=0.1, max_value=5.0, value=1.0, step=0.1
    )
    scale = base_scale * scale_mult
    biplot_fig = go.Figure()
    labels = y.astype(str).values
    unique_labels = np.unique(labels)
    palette = (
        DISCRETE_COLOR_PALETTE if DISCRETE_COLOR_PALETTE else px.colors.qualitative.T10
    )
    for idx, lab in enumerate(unique_labels):
        mask = labels == lab
        biplot_fig.add_trace(
            go.Scatter(
                x=scores[mask, 0],
                y=scores[mask, 1],
                mode="markers",
                name=str(lab),
                marker=dict(size=7, color=palette[idx % len(palette)], opacity=0.8),
                hoverinfo="text",
                hovertext=[
                    f"{lab}<br>PC1: {x:.3f}<br>PC2: {y:.3f}"
                    for x, y in scores[mask, :2]
                ],
            )
        )

    feature_names = X.columns.tolist()
    # loading lines + hover markers (labels appear on hover only)
    for i, feature in enumerate(feature_names):
        lx = loadings_raw[i, 0] * scale
        ly = loadings_raw[i, 1] * scale
        biplot_fig.add_trace(
            go.Scatter(
                x=[0, lx],
                y=[0, ly],
                mode="lines",
                line=dict(color="red", width=2),
                showlegend=False,
                hoverinfo="text",
                hovertext=[
                    f"{feature} loading: ({loadings_raw[i,0]:.4f}, {loadings_raw[i,1]:.4f})"
                ],
            )
        )
        biplot_fig.add_trace(
            go.Scatter(
                x=[lx],
                y=[ly],
                mode="markers",
                marker=dict(size=6, color="red"),
                showlegend=False,
                hoverinfo="text",
                hovertext=[f"{feature}<br>loading (scaled): ({lx:.3f}, {ly:.3f})"],
            )
        )

    pc1_var = evr[0] if len(evr) > 0 else 0.0
    pc2_var = evr[1] if len(evr) > 1 else 0.0
    biplot_fig.update_xaxes(title_text=f"PC1 ({pc1_var:.1%} variance)")
    biplot_fig.update_yaxes(title_text=f"PC2 ({pc2_var:.1%} variance)")
    biplot_fig.update_layout(
        title_text="PCA Biplot (hover features to see loadings)",
        height=600,
    )

    st.plotly_chart(biplot_fig, use_container_width=True)
    # show loadings table concisely under the biplot
    loadings = {
        "feature": feature_names,
        "PC1": (loadings_raw[:, 0]).round(4),
        "PC2": (loadings_raw[:, 1]).round(4),
    }
    loadings_df = pd.DataFrame(loadings).set_index("feature")
    st.dataframe(loadings_df)


def encoding(df: pd.DataFrame, target_df: pd.DataFrame = None):
    """Encode features and targets for downstream modelling.

    Args:
        df (pandas.DataFrame): Prepared dataset to encode.
        target_df (pandas.DataFrame | None): Target values aligned to the dataset index.

    Returns:
        None: Encoded artifacts are stored in session state and summaries are displayed.
    """
    st.subheader("Feature Selection")

    st.write(
        "Some of the features are useful for visualization but might not be that useful for modeling. Because of this, a few features have been dropped before encoding:\n"
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
    st.subheader("Encoded Data (preview)")
    st.write(
        "**Encoding Summary:**\n"
        "- Boolean features converted to 0/1\n"
        "- Sex feature converted to binary (1 for female, 0 for male)\n"
        "- Features causing target leakage or deemed not useful for modeling have been dropped\n"
        "- Missing values handled in previous imputation step"
    )

    st.dataframe(df_encoded.head(100))

    # Align target with encoded features
    y = target_series.reindex(df_encoded.index)
    mask = y.notna()

    X = df_encoded.loc[mask].copy()
    y = y.loc[mask].copy()

    # Encoding the target
    y_ser = pd.Series(y).astype(str).str.strip()

    # store target as plain numpy values under both keys for simplicity
    le = LabelEncoder()
    y_enc = le.fit_transform(y_ser)
    st.session_state["y_series"] = y_enc
    st.session_state["y_df"] = y
    st.session_state["X_encoded_df"] = X

    # Save a simple CSV in the data folder for quick reference (no models involved)
    classes_ = le.classes_.tolist()
    try:

        df_map = pd.DataFrame({"label": classes_, "id": list(range(len(classes_)))})
        df_map.to_csv("data/target_encoding.csv", index=False)
    except Exception as e:
        st.warning(f"Failed to save target encoding CSV: {e}")


def feature_selection():
    """Assess feature importance and persist selected feature subsets.

    Args:
        None: Relies on encoded data stored in Streamlit session state.

    Returns:
        None: Feature rankings and selections are displayed and saved.
    """
    # Feature importance analysis (only if target is available)
    y = st.session_state.get("y_df", None)
    X = st.session_state.get("X_encoded_df", None)
    st.subheader("Feature Importance Analysis")

    st.write(
        "The methods choesen for feature importance analysis are appropriate for data with underlying nonlinear relationships, as indicated by the multivariate analysis. The following methods will be used:\n"
    )
    st.write(
        "- Tree-based feature importances using weighted Random Forests\n"
        "- Mutual information"
    )

    try:
        model = RandomForestClassifier(
            n_estimators=200, random_state=0, n_jobs=-1, class_weight="balanced"
        )
        model.fit(X.values, y.values)

        # Tree-based feature importances
        feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(
            ascending=False
        )
        top_feat_imp = feat_imp.reset_index()
        top_feat_imp.columns = ["feature", "importance"]
        fig_imp = px.bar(
            top_feat_imp,
            x="feature",
            y="importance",
            title="Model Feature Importances (Tree-based)",
        )
        apply_standard_layout(fig_imp)
        st.plotly_chart(fig_imp, width="stretch")

        # Mutual information
        try:
            mi = mutual_info_classif(
                X.values, y.values, discrete_features="auto", random_state=0
            )
            mi_ser = pd.Series(mi, index=X.columns).sort_values(ascending=False)
            top_mi = mi_ser.reset_index()
            top_mi.columns = ["feature", "mutual_info"]
            fig_mi = px.bar(
                top_mi,
                x="feature",
                y="mutual_info",
                title="Mutual Information (Top Features)",
            )
            apply_standard_layout(fig_mi)
            st.plotly_chart(fig_mi, width="stretch")
        except Exception:
            st.info("Mutual information could not be computed in this environment.")
    except Exception as e:
        st.warning(f"Supervised importance computation failed: {e}")

    st.write(
        "Both methods indicate similar important features. The top 7 features will be selected for modeling."
    )

    st.session_state["selected_scaled_X"] = X[feat_imp.index].copy()
    st.session_state["selected_features"] = mi_ser.index[:7].tolist()

    st.write(
        f"**Selected Features for Modeling:** {', '.join(st.session_state['selected_features'])}"
    )
    # PCA visualization
    st.subheader("PCA Visualization")

    st.info(
        "Based on the Multivariate Analysis results, the relationships between features appear to be nonlinear. However, to get a first idea of how the data is structured in lower dimensions, PCA will be performed and visualized."
    )
    try:
        pca_fragment(X, y, n_components=X.shape[1])
        st.info(
            "It can be observed that the classes are not very well separated in PCA space, nor is there a clear elbow point at which to cut off components. "
            "This might be a consequence of the underlying nonlinearity. For investigating how PCA data behaves during modelling, the data will therefore be used further by starting simple, "
            "with just 2 principal components, and build up complexity from there."
        )
    except Exception as e:
        st.warning(f"PCA visualization failed: {e}")

    # csv = df_encoded.to_csv(index=False)
    # st.download_button(
    #     "Download encoded dataframe (CSV)",
    #     data=csv,
    #     file_name="encoded_df.csv",
    #     mime="text/csv",
    # )
