import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

from ..utils import _metric_plot


def general_modelling_structure():
    """Render the modelling tab structure and accompanying explanations.

    Args:
        None: This function does not accept any parameters.

    Returns:
        None: Streamlit components for modelling are rendered to the page.
    """
    st.subheader("Training and Evaluation")

    st.write(
        "- We use Stratified K-Fold CV to preserve class proportions across folds. \n"
        "- For imbalance, a simple class_weight='balanced' option was used, which reweights classes at training time for models that accept class_weight.\n"
        "- Results are evaluated using the following metrics, all suitable for multiclass classification:\n"
        "    - Balanced Accuracy: accounts for class imbalance by averaging recall obtained on each class.\n"
        "    - Macro F1: harmonic mean of precision and recall, averaged equally across classes.\n"
        "    - ROC AUC (OvR, Macro): area under the ROC curve using One-vs-Rest approach, averaged equally across classes.\n"
        "- Hyperparameters are kept mostly at default values, with some adjustments for training speed and convergence.\n"
        "- Experiments are run live in the app, which may be time and resource intensive, therefore results are also saved to disk for later visualization.\n"
    )
    st.markdown("---")

    modelling_section(
        "PCA Data Modelling Experiments",
        X_name="pca_scaled_X",
        y_name="y_series",
        feature_set="pca",
    )
    st.write(
        "As expected given the underlying non-linearities in the data, PCA features do not help in reducing dimensionality while preserving model performance."
        " Therefore, we proceed to use manually selected features based on EDA results."
    )

    st.markdown("---")
    modelling_section(
        "Manually Selected Features Modelling Experiments",
        X_name="selected_scaled_X",
        y_name="y_series",
        feature_set="selected",
    )

    st.subheader("Model Selection")
    st.write(
        "Based on the modelling experiments above, using the manually selected features based on the EDA results is preferred. "
        "The best performing model overall is **Random Forest**, achieving balanced accuracy of approximately 0.75 and macro F1 of approximately 0.73 with around 8 features."
        " This model will be selected for further tuning and deployment in the prediction section."
    )

    st.markdown("---")
    st.subheader("Testing the Chosen Model")
    best_model_testing()
    st.info(
        "The model does a good job for classifying *Hypothyroid* and *General Health*, however it still struggles for the other classes. Mostly, the model confuses the other classes with the majority class **Normal**."
    )

    st.markdown("---")
    st.subheader("Modelling Conclusions")
    st.info(
        "- Dimensionality reduction via PCA did not yield better performance; manually selected features based on EDA were more effective.\n"
        "- Random Forest emerged as the best performing model among those tested, likely due to its ability to handle non-linear relationships and feature interactions.\n"
        "- Further hyperparameter tuning and ensemble methods could be explored to enhance model performance.\n"
        "- The selected model will be used for predictions in the next section."
    )


def run_model_experiments(X, y, chosen, comp_values, skf, scoring, feature_set: str):
    """Execute cross-validated experiments for the selected classifiers.

    Args:
        X (numpy.ndarray): Feature matrix aligned with the target vector.
        y (numpy.ndarray): Target labels corresponding to `X`.
        chosen (list[tuple[str, object]]): Sequence of (name, estimator) pairs to evaluate.
        comp_values (list[int]): Component counts or feature counts to iterate over.
        skf (sklearn.model_selection.StratifiedKFold): Cross-validation splitter.
        scoring (dict[str, str]): Mapping of metric names to sklearn scoring identifiers.
        feature_set (str): Identifier for the feature set currently under evaluation.

    Returns:
        pandas.DataFrame: Aggregated cross-validation results for each classifier and component count.
    """
    results = []
    total_tasks = len(chosen) * len(comp_values)
    completed = 0
    progress = st.progress(0)

    for name, clf in chosen:
        for n_comp in comp_values:
            X_sub = X[:, :n_comp]
            try:
                cv_res = cross_validate(
                    clf, X_sub, y, cv=skf, scoring=scoring, n_jobs=-1
                )
                results.append(
                    {
                        "classifier": name,
                        "feature_set": feature_set,
                        "n_components": n_comp,
                        "balanced_accuracy": float(
                            np.mean(cv_res.get("test_balanced_accuracy", [np.nan]))
                        ),
                        "f1_macro": float(
                            np.mean(cv_res.get("test_f1_macro", [np.nan]))
                        ),
                        "roc_auc_ovr": float(
                            np.mean(cv_res.get("test_roc_auc_ovr", [np.nan]))
                        ),
                    }
                )
            except Exception:
                results.append(
                    {
                        "classifier": name,
                        "feature_set": feature_set,
                        "n_components": n_comp,
                        "balanced_accuracy": np.nan,
                        "f1_macro": np.nan,
                        "roc_auc_ovr": np.nan,
                    }
                )
            completed += 1
            progress.progress(int((completed / total_tasks) * 100))

    progress.progress(100)
    return pd.DataFrame(results)


def save_results_df(results_df: pd.DataFrame, save_dir: str | None = None):
    """Persist modelling results to disk without overwriting other feature sets.

    Args:
        results_df (pandas.DataFrame): Cross-validation results to be written.
        save_dir (str | None): Optional directory override for the results file.

    Returns:
        str: Filesystem path to the saved CSV file.
    """
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "data", "modelling_results")
    os.makedirs(save_dir, exist_ok=True)

    fullpath = os.path.join(save_dir, "model_results.csv")

    try:
        if os.path.isfile(fullpath):
            existing = pd.read_csv(fullpath)
            key_cols = ["classifier", "feature_set", "n_components"]
            # Drop any duplicates in existing
            if not existing.empty:
                existing = existing.drop_duplicates(subset=key_cols, keep="last")
            # Drop duplicates in new
            results_df = results_df.drop_duplicates(subset=key_cols, keep="last")
            # Merge: prefer new rows for same key
            merged = pd.concat([existing, results_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=key_cols, keep="last")
            merged.to_csv(fullpath, index=False)
        else:
            results_df.to_csv(fullpath, index=False)
    except Exception:
        # Fallback: write new results (never delete existing file contents)
        results_df.to_csv(fullpath, index=False)

    return fullpath


def modelling_section(subheader_title: str, X_name: str, y_name: str, feature_set: str):
    """Render a modelling section with live experiments and stored results.

    Args:
        subheader_title (str): Title displayed above the section contents.
        X_name (str): Session-state key storing the feature matrix.
        y_name (str): Session-state key storing the encoded targets.
        feature_set (str): Identifier describing the feature subset in use.

    Returns:
        None: Streamlit elements for modelling are rendered to the interface.
    """

    st.subheader(subheader_title)
    with st.expander(
        "Live Modelling Experiments (time consuming and resource intensive)",
        expanded=False,
    ):
        live_modelling_fragment(
            start_components=2,
            X_name=X_name,
            y_name=y_name,
            feature_set=feature_set,
        )

    st.markdown("#### Modelling Results Visualization")
    visualize_previous_results(feature_set_filter=feature_set)


@st.fragment
def live_modelling_fragment(
    X_name: str,
    y_name: str,
    feature_set: str,
    start_components: int = 2,
    X=None,
    y=None,
):
    """Allow users to run live modelling experiments from the Streamlit app.

    Args:
        X_name (str): Session-state key holding the feature matrix if `X` is None.
        y_name (str): Session-state key holding the target vector if `y` is None.
        feature_set (str): Identifier describing the feature subset in use.
        start_components (int, optional): Minimum number of features/components to evaluate.
        X (numpy.ndarray | None, optional): Feature matrix provided directly, bypassing session state.
        y (numpy.ndarray | None, optional): Target vector provided directly, bypassing session state.

    Returns:
        None: Results, controls, and visualizations are rendered in Streamlit.
    """
    if X is None:
        X = st.session_state.get(X_name)
    if y is None:
        y = st.session_state.get(y_name)

    if X is None or y is None:
        available = list(st.session_state.keys())
        st.warning(
            f"PCA-scaled features (`{X_name}`) or target (`{y_name}`/`target_series`) not found. Available keys: {available}"
        )
        return

    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        st.warning(
            f"Number of samples mismatch between X ({X.shape[0]}) and y ({y.shape[0]}). Keep both as plain arrays of equal length."
        )
        return

    n_features = X.shape[1]
    max_components = n_features

    start_components = max(1, min(start_components, max_components))

    # Use stable keys per feature_set to avoid duplicates across fragments

    # Base classifier constructors (limited to RF, XGB, SVC, LR)
    # Multiclass-only: determine number of classes
    classes = np.unique(y)
    n_classes = len(classes)

    base_classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=0, n_jobs=-1
        ),
        "SVC": SVC(probability=True, kernel="rbf", random_state=0),
    }
    # Configure XGBoost for multiclass
    base_classifiers["XGBoost"] = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=n_classes,
        n_jobs=-1,
        random_state=0,
    )
    # Build actual classifiers applying class_weight where available
    classifiers = {}
    for name, clf in base_classifiers.items():
        if name in ("LogisticRegression", "RandomForest", "SVC"):
            try:
                if name == "SVC":
                    classifiers[name] = SVC(
                        probability=True,
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=0,
                    )
                else:
                    # re-create with class_weight param
                    classifiers[name] = clf.__class__(
                        **{**clf.get_params(), "class_weight": "balanced"}
                    )
            except Exception:
                classifiers[name] = clf
        elif name == "XGBoost":
            # XGB uses scale_pos_weight already; keep as is
            classifiers[name] = clf
        else:
            classifiers[name] = clf
    st.write("Select classifiers to include in the modelling.")

    # classifier selection UI
    chosen = []
    cols = st.columns(len(classifiers))
    for i, (name, clf) in enumerate(classifiers.items()):
        if cols[i].checkbox(
            name,
            value=True,
            key=f"clf_select_{feature_set}_{i}_{name}",
        ):
            chosen.append((name, clf))

    if not chosen:
        st.info("Please select at least one classifier.")
        return

    # Controls for components
    max_c = st.slider(
        "Max features to evaluate",
        min_value=start_components,
        max_value=max_components,
        value=max_components,
        step=1,
        key=f"max_components_slider_{feature_set}",
    )
    step = st.number_input(
        "Step size for components",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        key=f"components_step_input_{feature_set}",
    )

    run_button = st.button(
        "Run experiments", key=f"run_experiments_button_{feature_set}"
    )
    if not run_button:
        st.write(
            f"Ready to run experiments using components {start_components}..{max_c} (step {step})"
        )
        return

    # Prepare evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # Metrics: multiclass only, include macro ROC AUC (OvR)
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "roc_auc_ovr": "roc_auc_ovr",
    }
    comp_values = list(range(start_components, max_c + 1, step))

    # run experiments via reusable helper and update progress bar per completed task
    results_df = run_model_experiments(
        X, y, chosen, comp_values, skf, scoring, feature_set=feature_set
    )

    # Always save results to the single CSV file
    save_results_df(results_df)

    metrics_section(results_df)

    st.success("Experiments complete.")


def metrics_section(results_df: pd.DataFrame):
    """Visualize modelling metrics across feature counts.

    Args:
        results_df (pandas.DataFrame): Cross-validation results containing metric columns.

    Returns:
        None: Plotly charts are rendered in Streamlit.
    """

    bacc_fig = _metric_plot(
        results_df, "balanced_accuracy", "Balanced Accuracy vs no. Features"
    )
    f1m_fig = _metric_plot(results_df, "f1_macro", "Macro F1 vs no. Features")

    col1, col2 = st.columns(2)
    with col1:
        if bacc_fig is not None:
            st.plotly_chart(bacc_fig, use_container_width=True)
    with col2:
        if f1m_fig is not None:
            st.plotly_chart(f1m_fig, use_container_width=True)
    roc_fig = _metric_plot(
        results_df, "roc_auc_ovr", "ROC AUC (OvR, Macro) vs no. Features"
    )
    if roc_fig is not None:
        st.plotly_chart(roc_fig, use_container_width=True)


def visualize_previous_results(feature_set_filter: str = None):
    """Display previously saved modelling results from disk.

    Args:
        feature_set_filter (str | None): Filter condition to restrict results to a feature subset.

    Returns:
        None: Existing results are loaded and visualized in Streamlit.
    """
    load_dir = os.path.join(os.getcwd(), "data", "modelling_results")
    fullpath = os.path.join(load_dir, "model_results.csv")

    if os.path.isfile(fullpath):
        try:
            loaded = pd.read_csv(fullpath)
            if feature_set_filter:
                loaded = loaded[loaded["feature_set"] == feature_set_filter]
                if loaded.empty:
                    st.warning(
                        f"No results found for feature_set '{feature_set_filter}'."
                    )
                    return
            metrics_section(loaded)
        except Exception as e:
            st.error(f"Failed to load results: {e}")
    else:
        st.info("No saved results file found in data/modelling_results/")


def best_model_testing():
    """Train, evaluate, and persist the selected Random Forest model.

    Args:
        None: This function reads data from Streamlit session state only.

    Returns:
        None: Evaluation metrics, plots, and saved artefacts are produced.
    """
    # Use Random Forest on first 8 features from selected_scaled_X
    X_df = st.session_state.get("selected_scaled_X")
    y = st.session_state.get("y_series")
    # Capture feature names for the first 8 selected features (for importances)
    feature_names_8 = None
    if isinstance(X_df, pd.DataFrame):
        feature_names_8 = X_df.columns[:8].tolist()
        X = X_df.values
    else:
        X = np.asarray(X_df)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        st.warning(
            f"Sample size mismatch: X has {X.shape[0]} rows, y has {y.shape[0]} rows."
        )
        return

    X8 = X[:, :8]
    st.write("Training Random Forest with the first 8 selected features.")

    rf = RandomForestClassifier(
        n_estimators=300, random_state=0, n_jobs=-1, class_weight="balanced"
    )

    # Split into train/test for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X8, y, test_size=0.2, stratify=y, random_state=0
    )

    # Cross-validated evaluation (Stratified K-Fold) on training split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "roc_auc_ovr": "roc_auc_ovr",
    }
    try:
        cv_res = cross_validate(
            rf, X_train, y_train, cv=skf, scoring=scoring, n_jobs=-1
        )
        mean_bacc = float(np.mean(cv_res.get("test_balanced_accuracy", [np.nan])))
        mean_f1m = float(np.mean(cv_res.get("test_f1_macro", [np.nan])))
        mean_roc = float(np.mean(cv_res.get("test_roc_auc_ovr", [np.nan])))
        st.success(
            f"Train CV (k=5) — Balanced Acc: {mean_bacc:.3f}, Macro F1: {mean_f1m:.3f}, ROC AUC (OvR): {mean_roc:.3f}"
        )
    except Exception as e:
        st.warning(f"Cross-validated evaluation on train split failed: {e}")

    # Fit on training data
    rf.fit(X_train, y_train)

    # Save the final trained model automatically under models/
    try:
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "best_rf_selected8.joblib")
        joblib.dump({"model": rf, "feature_count": 8}, model_path)
    except Exception as e:
        st.warning(f"Failed to save model: {e}")

    # Evaluate on test split
    y_pred = rf.predict(X_test)
    # Raw counts and normalized recall
    raw_cm = confusion_matrix(y_test, y_pred)
    cm = raw_cm.astype(float) / raw_cm.sum(axis=1, keepdims=True)
    cm = np.nan_to_num(cm)
    # Per-true-class totals (denominator for recall)
    row_totals = raw_cm.sum(axis=1)
    present_ids = list(np.unique(y_test))

    # Map ids -> labels from CSV, fallback to ids
    try:
        map_df = pd.read_csv("data/target_encoding.csv")
        id_to_label = {
            int(row["id"]): str(row["label"]) for _, row in map_df.iterrows()
        }
        tick_labels = [id_to_label.get(int(i), str(i)) for i in present_ids]
    except Exception:
        tick_labels = [str(i) for i in present_ids]

    # Custom data: [count, total] per cell to display in hover
    customdata = np.dstack(
        [
            raw_cm,
            np.repeat(row_totals[:, None], raw_cm.shape[1], axis=1),
        ]
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=tick_labels,
            y=tick_labels,
            colorscale="Reds",
            colorbar=dict(title="Recall"),
            customdata=customdata,
            hovertemplate="True %{y}<br>Pred %{x}<br>Recall %{z:.2f}<br>Count %{customdata[0]} / %{customdata[1]}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Confusion Matrix (Per-Class Recall, Test Split)",
        xaxis_title="Predicted",
        yaxis_title="True",
    )
    # Add text annotations (percent recall) centered in each cell
    for i, ylab in enumerate(tick_labels):
        for j, xlab in enumerate(tick_labels):
            fig.add_annotation(
                x=xlab,
                y=ylab,
                text=f"{(cm[i, j]*100):.1f}%",
                showarrow=False,
                font=dict(color="black" if cm[i, j] < 0.6 else "white"),
            )
    fig.update_xaxes(side="top")
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance plot for the final model
    try:
        importances = getattr(rf, "feature_importances_", None)
        if importances is not None and feature_names_8 is not None:
            imp_df = pd.DataFrame(
                {
                    "feature": feature_names_8,
                    "importance": importances[: len(feature_names_8)],
                }
            ).sort_values("importance", ascending=False)
            fig_imp = px.bar(
                imp_df,
                x="feature",
                y="importance",
                title="Final Model Feature Importances (Random Forest)",
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info(
                "Feature importances are unavailable for this model or feature names are missing."
            )
    except Exception as e:
        st.warning(f"Failed to render feature importances: {e}")

    # Show classification report
    # Show concise test metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    macro_f1 = report.get("macro avg", {}).get("f1-score", None)
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", None)
    st.write(
        f"Test Macro F1: {macro_f1:.3f} | Test Weighted F1: {weighted_f1:.3f}"
        if macro_f1 is not None and weighted_f1 is not None
        else "Test metrics could not be computed."
    )

    # Store the trained model and feature slice count in session state
    st.session_state["best_model_rf"] = rf
    st.session_state["best_model_features"] = 8
