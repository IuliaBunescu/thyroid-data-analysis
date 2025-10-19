import pandas as pd
import plotly.express as px
import streamlit as st
from source.config import (
    AXIS_TICK_FONT_SIZE,
    AXIS_TITLE_FONT_SIZE,
    COLUMN_DESCRIPTIONS,
    TITLE_FONT_SIZE,
    pink_red_palette,
)


def general_ida_structure(
    df: pd.DataFrame, lab_references: pd.DataFrame, condition_codes: pd.DataFrame
):
    """'
    General structure for the IDA tab
    Inputs:
    - df: pd.DataFrame
        The main thyroid dataset
    - lab_references: pd.DataFrame
        The laboratory reference intervals dataset
    - condition_codes: pd.DataFrame
        The condition codes dataset
    """
    st.header("Initial Data Analysis (IDA)")
    general_metrics(df)

    st.markdown("---")

    st.header("Feature Exploration")
    feature_explanation_frag(
        df, lab_references=lab_references, condition_codes=condition_codes
    )

    st.markdown("---")
    col1, col2 = st.columns(2)
    col1.header("Abnormal Values")
    col1.write(
        "The most obvious abnormal values were whithin the *age* feature, where "
        "several entries had ages over 100 years old, which is unlikely. "
        "They were very few (4) and were removed during data cleaning."
        "In terms of thyroid measurements, there are no obvious aberrant values (i.e. negative values), however, some values are outside the normal lab reference intervals."
        "Most notably *TSH* has several extremely high values (over 100 mIU/L), which will be investigated further, but for now will remain part of the data."
    )
    col2.header("Outlier Analysis")
    col2.write(
        "Outliers can be observed in all numerical features of the dataset. However, given the clinical nature of the dataset, these outliers may represent true clinical extremes rather than data errors."
        " Therefore, no outlier removal has been performed at this stage and robust scaling methods will be considered during modelling."
    )

    st.markdown("---")
    st.header("Missing Data Analysis")

    missing_data_analysis_frag(df)


def general_metrics(df: pd.DataFrame):
    """
    Show general metrics about the dataset
    """
    if df is None or df.empty:
        st.info("No data available")
        return

    total_patients = len(df)
    total_features = len(df.columns)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Patients", f"{total_patients:,}", border=True)

    with col2:
        st.metric("Total Features", total_features, border=True)

    with col3:
        have_condition = len(df) - df[df["condition_primary"] == "-"].shape[0]
        st.metric(
            "Patients with a Thyroid Condition", f"{have_condition:,}", border=True
        )

    # Percentages of entries on medications
    st.subheader("Medication Usage Percentages")
    med_cols = [
        "on_thyroxine",
        "on_antithyroid_medication",
        "I131_treatment",
        "lithium",
    ]
    cols = st.columns(len(med_cols))
    for col_widget, col in zip(cols, med_cols):
        percent = df[col].mean() * 100
        with col_widget:
            st.metric(
                col.replace("_", " ").title(),
                f"{percent:.1f}%",
                border=True,
            )


@st.fragment
def feature_explanation_frag(
    df: pd.DataFrame,
    lab_references: pd.DataFrame = None,
    condition_codes: pd.DataFrame = None,
):
    """
    Provide explanation for a selected feature.
    """
    if df is None or df.empty:
        st.info("No data available")
        return
    lc, _ = st.columns(2)

    feature = lc.selectbox("Select feature", df.columns.tolist())

    if feature is None:
        return

    left_col, right_col = st.columns(2)

    with left_col:

        description = COLUMN_DESCRIPTIONS.get(feature, {})
        short_desc = description.get("short", "No description available.")
        long_desc = description.get("long", "")
        link = description.get("link", "")

        st.info(f"**{feature}** — {short_desc}")

        if long_desc:
            st.markdown(
                f"<div style='background-color:rgba(255, 192, 203, 0.25); "
                f"padding:1rem; border-radius:0.5rem; color:#5A1E1E;'>"
                f"{long_desc}</div>",
                unsafe_allow_html=True,
            )

        if link:
            st.markdown(f"[Learn more (Wikipedia)]({link})")

        if lab_references is not None and feature in lab_references["test_name"].values:
            ref_row = lab_references[lab_references["test_name"] == feature].iloc[0]
            normal_low = ref_row["normal_low"]
            normal_high = ref_row["normal_high"]
            units = ref_row["units"]
            # show key reference values as Streamlit metrics
            col_low, col_high, col_units = st.columns(3)
            with col_low:
                st.metric("Normal Low", normal_low, border=True)
            with col_high:
                st.metric("Normal High", normal_high, border=True)
            with col_units:
                st.metric("Units", units, border=True, help=units)

            # Show any additional descriptive/reference columns (e.g., description, notes)
            extra_cols = [
                c
                for c in ref_row.index
                if c not in {"test_name", "normal_low", "normal_high", "units"}
            ]
            if extra_cols:
                st.subheader("Additional reference information:")
                for c in extra_cols:
                    val = ref_row[c]
                    if pd.isna(val) or str(val).strip() == "":
                        continue
                    label = c.replace("_", " ").title()
                    st.write(f"**{label}:** {val}")

        if feature in ["condition_primary", "condition_secondary"]:
            st.subheader("Condition Code Details")
            st.dataframe(condition_codes)
    with right_col:
        series = df[feature].dropna()

        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
            series
        ):
            fig = px.histogram(
                x=series,
                nbins=75,
                marginal="box",
                title=f"Distribution of {feature}",
                labels={"x": feature},
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
        else:
            counts = series.astype(str).value_counts().reset_index()
            counts.columns = [feature, "count"]
            fig = px.bar(counts, x=feature, y="count", title=f"Counts of {feature}")
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

        st.plotly_chart(fig, use_container_width=True)


@st.fragment
def missing_data_analysis_frag(df: pd.DataFrame):
    """
    Analyze and visualize missing data pattern across rows and give option to sort by a feature.
    """
    if df is None or df.empty:
        st.info("No data available")
        return

    # summary table
    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100
    missing_df = pd.DataFrame(
        {"missing_count": missing_counts, "missing_percent": missing_percent}
    )
    missing_df = missing_df[missing_df["missing_count"] > 0].sort_values(
        by="missing_percent", ascending=False
    )

    if missing_df.empty:
        st.success("No missing data in the dataset!")
        return

    # Bar plot of missing percentages (counts shown on hover)
    plot_df = missing_df.reset_index().rename(columns={"index": "feature"})
    # human-friendly percent label to show on bars
    plot_df["pct_label"] = plot_df["missing_percent"].apply(lambda x: f"{x:.1f}%")

    fig = px.bar(
        plot_df,
        x="feature",
        y="missing_percent",
        # orientation="h",
        text="pct_label",
        hover_data=["missing_count", "missing_percent"],
        title="Missing data by feature",
        labels={"missing_percent": "Missing (%)", "feature": "Feature"},
        category_orders={"feature": plot_df["feature"].tolist()},
    )

    fig.update_traces(marker_color=pink_red_palette[1], textposition="outside")

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
            automargin=True,
        ),
        title=dict(font=dict(size=TITLE_FONT_SIZE)),
    )

    st.plotly_chart(fig, use_container_width=True)

    cols_with_missing = missing_df.index.tolist()
    st.subheader("Missingness Pattern")

    # Sample / size controls (datasets can be large)
    max_rows = len(df)
    col1, _ = st.columns(2)
    with col1:
        sample_size = st.slider(
            "Rows to display",
            min_value=1,
            max_value=max_rows,
            value=max_rows,
            step=1,
        )
        sample_method = st.selectbox(
            "Sampling method", ["Head (first rows)", "Random sample"]
        )

        # Choose feature to sort by (helps observe pattern relative to a specific feature)
        sort_feature = st.selectbox(
            "Sort rows by feature (to observe missingness pattern)",
            ["None"] + cols_with_missing,
        )
        sort_missing_first = st.checkbox(
            "Put missing values first when sorting", value=True
        )

    # Prepare the sampled dataframe
    if sample_method.startswith("Random"):
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df.head(sample_size).copy()

    # Optionally sort rows by selected feature's missingness (and value)
    if sort_feature != "None":
        sf = sort_feature
        # Create helper sorting keys
        sample_df["_isna_sort"] = sample_df[sf].isna().astype(int)
        # Use string representation for stable sorting of values (non-missing)
        sample_df["_val_sort"] = sample_df[sf].astype(str)
        # If we want missing first, sort _isna_sort descending, else ascending
        asc_isna = not sort_missing_first
        sample_df = sample_df.sort_values(
            by=["_isna_sort", "_val_sort"],
            ascending=[asc_isna, True],
            na_position="last",
        )
        sample_df = sample_df.drop(columns=["_isna_sort", "_val_sort"])

    # Build binary missing matrix (1 = missing, 0 = present)
    missing_matrix = sample_df[cols_with_missing].isna().astype(int).T

    # Plot heatmap of missingness
    fig = px.imshow(
        missing_matrix,
        labels={
            "x": "Row (sample index)",
            "y": "Feature",
            "color": "Missing (1=missing)",
        },
        x=missing_matrix.columns.astype(str),
        y=missing_matrix.index,
        aspect="auto",
        color_continuous_scale=pink_red_palette[2:4],
        origin="lower",
    )

    # layout adjustments
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
        title=dict(
            font=dict(size=TITLE_FONT_SIZE),
            text=f"Missingness pattern ({sample_size} rows)",
        ),
        coloraxis_showscale=False,
    )

    # avoid overcrowding x tick labels for large samples
    if sample_size > 80:
        fig.update_xaxes(showticklabels=False)

    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Blush cells = missing values; Magenta cells = present. Use sorting to observe how missingness aligns with a specific feature."
    )

    st.info(
        "**Conclusions**: \n"
        "- *sex*: MCAR (missing completely at random), no pattern observed; \n"
        "- *secondary_condition*: MCAR (missing completely at random), no pattern observed; it is also not mandatory for an observation to have a secondary condition; \n"
        "- *TT4*, *FTI*, *T4U*, *TSH*, *T3*: likely a combination of MNAR (not missing at random), as missingness seems to be synchronized between them, MAR (missing at random) as some entries appear to miss at random; \n"
        "- *TBG*: very likely MAR (missing at random), as missingness seems to be related to the value itself (only few non-missing values when TSH, T4U, T3, TT4, FTI are missing)."
    )

    st.subheader("TBG Presence Investigation")

    st.write(
        "*Note*: TBG is not a standard screening test for thyroid function; it is usually measured in specific clinical scenarios."
    )

    # Ensure TBG exists
    if "TBG" not in df.columns:
        st.info("Column 'TBG' not found in dataset.")
        return

    # Features of interest to check against TBG presence
    related_tests = ["TT4", "FTI", "T4U", "TSH", "T3"]
    related_tests = [c for c in related_tests if c in df.columns]

    # Missingness rates for related tests by TBG presence
    if related_tests:
        miss_rates = (
            df.assign(**{f"{c}_isna": df[c].isna() for c in related_tests})
            .groupby("TBG_measured")[[f"{c}_isna" for c in related_tests]]
            .mean()
            .reset_index()
            .melt(id_vars="TBG_measured", var_name="test", value_name="missing_rate")
        )
        # clean test name
        miss_rates["test"] = miss_rates["test"].str.replace("_isna$", "", regex=True)

        miss_rates["missing_pct"] = miss_rates["missing_rate"] * 100
        miss_rates["TBG_measured_str"] = miss_rates["TBG_measured"].map(
            {True: "TBG present", False: "TBG missing"}
        )

        fig_miss = px.bar(
            miss_rates,
            x="test",
            y="missing_pct",
            color="TBG_measured_str",
            barmode="group",
            text=miss_rates["missing_pct"].apply(lambda x: f"{x:.1f}%"),
            title="Missing % of related tests when TBG present vs missing",
            color_discrete_sequence=[pink_red_palette[3], pink_red_palette[2]],
            labels={
                "missing_pct": "Missing (%)",
                "test": "Test",
                "TBG_measured_str": "",
            },
        )
        fig_miss.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            hoverlabel=dict(font=dict(size=AXIS_TICK_FONT_SIZE)),
            xaxis=dict(tickfont=dict(size=AXIS_TICK_FONT_SIZE)),
            yaxis=dict(title_font=dict(size=AXIS_TITLE_FONT_SIZE)),
            title=dict(font=dict(size=TITLE_FONT_SIZE)),
        )
        st.plotly_chart(fig_miss, use_container_width=True)

    st.markdown("### Explore distributions when TBG is measured")

    # Determine TBG subsets (prefer explicit TBG_measured flag, fall back to TBG presence)
    if "TBG_measured" in df.columns:
        mask_true = df["TBG_measured"].astype(bool)
    else:
        mask_true = (
            df["TBG"].notna()
            if "TBG" in df.columns
            else pd.Series([False] * len(df), index=df.index)
        )

    tb_df = df[mask_true].copy()
    no_tb_df = df[~mask_true].copy()

    # Candidate features (exclude TBG-related technical cols)
    candidate_features = [c for c in df.columns if c not in {"TBG", "TBG_measured"}]
    pick = st.multiselect(
        "Select features to examine (only percentage-based comparisons are shown).",
        candidate_features,
        default=["sex", "condition_primary"],
    )

    if not pick:
        st.info("No features selected.")
    else:
        for feat in pick:
            if feat not in df.columns:
                st.write(f"Feature {feat} not found.")
                continue

            # Prepare datasets (add indicator for plotting)
            df_tb = (
                tb_df[[feat]].assign(_tbg="TBG measured")
                if not tb_df.empty
                else pd.DataFrame(columns=[feat, "_tbg"])
            )
            df_no = (
                no_tb_df[[feat]].assign(_tbg="TBG missing")
                if not no_tb_df.empty
                else pd.DataFrame(columns=[feat, "_tbg"])
            )

            if df_tb.empty and df_no.empty:
                st.write(f"No data for feature {feat} in either subset.")
                continue

            # Numeric: overlay percent-normalized histograms
            if pd.api.types.is_numeric_dtype(df[feat]):
                comb = pd.concat([df_tb, df_no], ignore_index=True)
                if comb[feat].dropna().empty:
                    st.write(f"No non-missing numeric values for {feat}.")
                    continue

                fig = px.histogram(
                    comb,
                    x=feat,
                    color="_tbg",
                    barmode="overlay",
                    histnorm="percent",
                    opacity=0.6,
                    nbins=50,
                    marginal="box",
                    title=f"Percent distribution of {feat} — measured vs missing TBG",
                    labels={feat: feat, "_tbg": ""},
                    color_discrete_sequence=pink_red_palette[0:2],
                )
                fig.update_traces(marker=dict(line=dict(width=0)))
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    hoverlabel=dict(font=dict(size=AXIS_TICK_FONT_SIZE)),
                    xaxis=dict(
                        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
                        tickfont=dict(size=AXIS_TICK_FONT_SIZE),
                    ),
                    yaxis=dict(
                        title="Percent (%)",
                        title_font=dict(size=AXIS_TITLE_FONT_SIZE),
                        tickfont=dict(size=AXIS_TICK_FONT_SIZE),
                    ),
                    title=dict(font=dict(size=TITLE_FONT_SIZE)),
                )
                st.plotly_chart(fig, use_container_width=True)

            # Categorical: compute percent distributions for each subset and plot together
            else:
                s_tb = (
                    df_tb[feat].dropna().astype(str)
                    if not df_tb.empty
                    else pd.Series(dtype=str)
                )
                s_no = (
                    df_no[feat].dropna().astype(str)
                    if not df_no.empty
                    else pd.Series(dtype=str)
                )

                if s_tb.empty and s_no.empty:
                    st.write(f"No non-missing categorical values for {feat}.")
                    continue

                pct_tb = (
                    s_tb.value_counts(normalize=True)
                    .mul(100)
                    .rename("percent")
                    .reset_index()
                    .rename(columns={"index": feat})
                )
                pct_tb["_tbg"] = "TBG measured"
                pct_no = (
                    s_no.value_counts(normalize=True)
                    .mul(100)
                    .rename("percent")
                    .reset_index()
                    .rename(columns={"index": feat})
                )
                pct_no["_tbg"] = "TBG missing"

                pct_comb = pd.concat([pct_tb, pct_no], ignore_index=True)
                # Ensure categories present in one subset but not the other are shown (missing will be absent => 0%)
                # Plot grouped bars of percent
                fig = px.bar(
                    pct_comb,
                    x=feat,
                    y="percent",
                    color="_tbg",
                    barmode="group",
                    title=f"Percent distribution of {feat} — measured vs missing TBG",
                    labels={feat: feat, "percent": "Percent (%)", "_tbg": ""},
                    color_discrete_sequence=pink_red_palette[2:4],
                )
                fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
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
                st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Conclusions**: \n"
        "- TBG is most likely being measured when the other thyroid function tests are being inconclusive. \n"
        "- There is a difference in the distribution of gender, where TBG is being measured for females more often than males."
        "- Most likely TBG is affected by data outside the dataset which might be related to patient history, most notably inherited conditions."
        "- MNAR (not missing at random) mechanism is likely at play here."
    )
