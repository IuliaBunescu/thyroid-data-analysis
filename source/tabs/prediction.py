from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from source.utils import apply_standard_layout


@st.cache_resource(show_spinner=False)
def _load_prediction_model(model_path: str):
    """Load the trained model once per deployment and neutralize parallel workers."""

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at {path}")

    saved = joblib.load(path)
    model = saved.get("model")
    if model is None:
        raise ValueError("Saved object missing 'model'.")

    feature_count = int(saved.get("feature_count", 8))

    # Ensure predictions run single-threaded to avoid joblib worker issues on shared hosts
    try:
        model.set_params(n_jobs=1)
    except (TypeError, ValueError, AttributeError):
        pass

    return model, feature_count


@st.fragment
def general_prediction_section(
    lab_references: pd.DataFrame = None, enc_df: pd.DataFrame = None
):
    """Render the prediction workflow and display model outputs.

    Args:
        lab_references (pandas.DataFrame | None): Reference ranges and labels used to annotate inputs.

    Returns:
        None: The Streamlit form and prediction results are rendered to the page.
    """
    st.subheader("Predict Thyroid Condition")

    # Load trained final model
    model_path = (
        Path(__file__).resolve().parents[2] / "models" / "best_rf_selected8.joblib"
    )
    try:
        rf, feature_count = _load_prediction_model(str(model_path))
    except FileNotFoundError:
        st.warning(
            "Final model not found. Please run the Modelling tab to train and save the model."
        )
        return
    except Exception as e:
        st.error(f"Failed to load saved model: {e}")
        return

    # Determine the required feature names and order (first 8 selected features)
    X_df = st.session_state.get("selected_scaled_X")
    if X_df is None or not isinstance(X_df, pd.DataFrame):
        st.warning(
            "Selected feature DataFrame not available. Please complete EDA/Feature Selection."
        )
        return
    feature_names = X_df.columns[:feature_count].tolist()

    if "on_thyroxine" in X_df.columns and "on_thyroxine" not in feature_names:
        feature_names = ["on_thyroxine"] + feature_names
        feature_names = feature_names[:feature_count]

    st.info(
        "Enter all required features used by the final model. All fields are mandatory."
    )
    st.markdown("#### Input Features")

    # Inputs layout in two columns: encode booleans, keep others raw
    # Prepare a lookup from provided lab reference dataframe (required)
    ref_map = {}
    if lab_references is None or not isinstance(lab_references, pd.DataFrame):
        st.error("Lab reference DataFrame is required and must be a DataFrame.")
        return
    try:
        # Build both raw and normalized maps for robust key matching
        def norm_key(s: str) -> str:
            s = (s or "").strip().lower()
            # remove spaces and non-alphanumeric, keep letters/numbers only
            return "".join(ch for ch in s if ch.isalnum())

        ref_map_norm = {}
        for _, r in lab_references.iterrows():
            key = str(r.get("test_name", "")).strip()
            if key:
                item = {
                    "units": str(r.get("units", "")).strip() or None,
                    "complete_name": str(r.get("complete_name", "")).strip() or None,
                    "normal_low": r.get("normal_low", None),
                    "normal_high": r.get("normal_high", None),
                }
                ref_map[key] = item
                ref_map_norm[norm_key(key)] = item
    except Exception as e:
        st.error(f"Failed to parse lab reference DataFrame: {e}")
        return

    # Helper available to both columns for feature lookup
    def get_ref(name: str):
        exact = ref_map.get(name)
        if exact:
            return exact
        nk = (name or "").strip().lower()
        nk = "".join(ch for ch in nk if ch.isalnum())
        return ref_map_norm.get(nk, {})

    def is_boolean_feature(name: str) -> bool:
        n = name.lower()
        return (
            n.startswith("on_")
            or n.endswith("_measured")
            or n.startswith("query_")
            or "thyroxine" in n
        )

    left, right = st.columns(2)

    # Split features for columnar layout
    mid = (len(feature_names) + 1) // 2
    left_feats = feature_names[:mid]
    right_feats = feature_names[mid:]

    user_vals = {}

    with left:
        for i, feat in enumerate(left_feats):
            if is_boolean_feature(feat):
                sel = st.selectbox(
                    f"{feat} (Yes/No)", ["No", "Yes"], index=0, key=f"pred_bool_{feat}"
                )
                user_vals[feat] = float(1 if sel == "Yes" else 0)
            else:
                val = (
                    float(X_df[feat].median())
                    if (
                        feat in X_df.columns
                        and pd.api.types.is_numeric_dtype(X_df[feat])
                    )
                    else 0.0
                )
                # Add units and normal range info from provided lab_references
                # Lookup by exact and normalized names
                ref = get_ref(feat)
                units = ref.get("units")
                cname = ref.get("complete_name")
                nlow = ref.get("normal_low")
                nhigh = ref.get("normal_high")

                label = f"{feat}{f' ({units})' if units else ''}"
                rng = None
                if nlow is not None and nhigh is not None:
                    rng = f"Normal range: {nlow}–{nhigh}"
                help_txt = (
                    " | ".join([p for p in [cname, rng] if p])
                    if (cname or rng)
                    else None
                )

                num = st.number_input(
                    label,
                    value=val,
                    format="%f",
                    key=f"pred_input_left_{i}_{feat}",
                    help=help_txt,
                )
                user_vals[feat] = float(num)

    with right:
        for j, feat in enumerate(right_feats):
            if is_boolean_feature(feat):
                sel = st.selectbox(
                    f"{feat} (Yes/No)", ["No", "Yes"], index=0, key=f"pred_bool_{feat}"
                )
                user_vals[feat] = float(1 if sel == "Yes" else 0)
            else:
                val = (
                    float(X_df[feat].median())
                    if (
                        feat in X_df.columns
                        and pd.api.types.is_numeric_dtype(X_df[feat])
                    )
                    else 0.0
                )
                ref = get_ref(feat)
                units = ref.get("units")
                cname = ref.get("complete_name")
                nlow = ref.get("normal_low")
                nhigh = ref.get("normal_high")

                label = f"{feat}{f' ({units})' if units else ''}"
                rng = None
                if nlow is not None and nhigh is not None:
                    rng = f"Normal range: {nlow}–{nhigh}"
                help_txt = (
                    " | ".join([p for p in [cname, rng] if p])
                    if (cname or rng)
                    else None
                )

                num = st.number_input(
                    label,
                    value=val,
                    format="%f",
                    key=f"pred_input_right_{j}_{feat}",
                    help=help_txt,
                )
                user_vals[feat] = float(num)

    # Submit and run prediction
    if st.button("Predict", type="primary"):
        try:
            # Assemble in the exact order expected by the model
            x_row = np.array(
                [user_vals[f] for f in feature_names], dtype=float
            ).reshape(1, -1)

            # Note: Selected features were standardized earlier; avoid double-scaling here.
            # If a preprocessing pipeline is introduced, load and apply it instead.
            y_pred = rf.predict(x_row)
            y_proba = None
            try:
                y_proba = rf.predict_proba(x_row)
            except Exception:
                y_proba = None

            # Map class id -> human-readable label using target_encoding.csv
            label = str(y_pred[0])
            try:

                id_to_label = {
                    int(r["id"]): str(r["label"]) for _, r in enc_df.iterrows()
                }
                label = id_to_label.get(int(y_pred[0]), label)
            except Exception:
                pass

            st.info(f"Predicted condition: **{label}**")

            if y_proba is not None:
                # Build probability display ordered by id_to_label mapping if available
                try:

                    id_to_label = {
                        int(r["id"]): str(r["label"]) for _, r in enc_df.iterrows()
                    }
                    labels = [
                        id_to_label.get(i, str(i)) for i in range(y_proba.shape[1])
                    ]
                except Exception:
                    labels = [str(i) for i in range(y_proba.shape[1])]

                prob_df = pd.DataFrame({"class": labels, "probability": y_proba[0]})
                prob_df = prob_df.sort_values("probability", ascending=False)

                fig = px.bar(
                    prob_df,
                    x="class",
                    y="probability",
                    title="Predicted Class Probabilities",
                    labels={"class": "Class", "probability": "Probability"},
                )
                apply_standard_layout(
                    fig, extra_layout={"margin": dict(l=0, r=0, t=40, b=0)}
                )
                st.plotly_chart(fig, width="content")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
