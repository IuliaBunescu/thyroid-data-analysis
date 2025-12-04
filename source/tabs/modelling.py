import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def general_modelling_structure():
    """
    General structure for the Modelling tab
    """
    st.header("Modelling")

    st.write(
        "Classification experiments using PCA-reduced features (from session state)."
    )
    modelling_fragment()


@st.fragment
def modelling_fragment(
    X: pd.DataFrame = None,
    y: pd.Series = None,
    start_components: int = 2,
    max_components: int = None,
):
    """Run simple classification experiments using the first N PCA components.

    - If `X`/`y` are not provided, the fragment will try to read `st.session_state['pca_scaled_X']`
      and `st.session_state['y_series']`.
    - Runs a small set of classifiers with Stratified K-Fold CV and reports accuracy / f1 / precision / recall.
    - By default starts with `start_components` (3) and increments up to `max_components` (or n_features).
    """
    # Simple resolution: load plain arrays stored in session_state

    if X is None:
        X = st.session_state.get("pca_scaled_X")
    if y is None:
        # prefer y_series then target_series
        y = st.session_state.get("y_series")

    if X is None or y is None:
        available = list(st.session_state.keys())
        st.warning(
            f"PCA-scaled features (`pca_scaled_X`) or target (`y_series`/`target_series`) not found. Available keys: {available}"
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
    if max_components is None:
        max_components = n_features

    start_components = max(1, min(start_components, max_components))

    # Base classifier constructors
    base_classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, solver="lbfgs", multi_class="auto"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=0, n_jobs=-1
        ),
        "SVC": SVC(probability=True, kernel="rbf", random_state=0),
        "GradientBoosting": GradientBoostingClassifier(random_state=0),
        "KNeighbors": KNeighborsClassifier(),
    }

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
        else:
            classifiers[name] = clf

    st.info(
        """ **Training and Evaluation**

        -We use Stratified K-Fold CV to preserve class proportions across folds.
        -For imbalance handling the app supports a simple class_weight='balanced' option which reweights classes at training time for models that accept class_weight.
        -Results are summarized using weighted metrics (F1/precision/recall) so performance reflects class distribution.)
    """
    )
    st.write(
        "Select classifiers to include in the comparison and run CV across increasing PCA components."
    )

    # classifier selection UI
    chosen = []
    cols = st.columns(len(classifiers))
    for i, (name, clf) in enumerate(classifiers.items()):
        if cols[i].checkbox(name, value=True):
            chosen.append((name, clf))

    if not chosen:
        st.info("Please select at least one classifier.")
        return

    # Controls for components
    max_c = st.slider(
        "Max PCA components to evaluate",
        min_value=start_components,
        max_value=max_components,
        value=max_components,
        step=1,
    )
    step = st.number_input(
        "Step size for components", min_value=1, max_value=5, value=1, step=1
    )

    run_button = st.button("Run experiments")
    if not run_button:
        st.write(
            f"Ready to run experiments using components {start_components}..{max_c} (step {step})"
        )
        return

    # Prepare evaluation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # Use metrics that are less sensitive to imbalance by using weighted averages
    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": "f1_weighted",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
    }

    results = []
    comp_values = list(range(start_components, max_c + 1, step))

    progress = st.progress(0)
    total_tasks = len(chosen) * len(comp_values)
    completed = 0

    for name, clf in chosen:
        for n_comp in comp_values:
            X_sub = X[:, :n_comp]
            try:
                cv_res = cross_validate(
                    clf, X_sub, y, cv=skf, scoring=scoring, n_jobs=-1
                )
            except Exception as e:
                st.warning(
                    f"Evaluation failed for {name} with {n_comp} components: {e}"
                )
                # record NaNs
                results.append(
                    {
                        "classifier": name,
                        "n_components": n_comp,
                        "accuracy": np.nan,
                        "f1_weighted": np.nan,
                        "precision_weighted": np.nan,
                        "recall_weighted": np.nan,
                    }
                )
                completed += 1
                progress.progress(int(completed / total_tasks * 100))
                continue

            results.append(
                {
                    "classifier": name,
                    "n_components": n_comp,
                    "accuracy": float(np.mean(cv_res["test_accuracy"])),
                    "f1_weighted": float(np.mean(cv_res["test_f1_weighted"])),
                    "precision_weighted": float(
                        np.mean(cv_res["test_precision_weighted"])
                    ),
                    "recall_weighted": float(np.mean(cv_res["test_recall_weighted"])),
                }
            )

            completed += 1
            progress.progress(int(completed / total_tasks * 100))

    results_df = pd.DataFrame(results)

    if results_df.empty:
        st.warning("No results to display.")
        return

    # Layout: two columns, show two metric plots per column
    col1, col2 = st.columns(2)

    # Helper to build a line plot for a metric
    def _metric_plot(df, metric, title):
        try:
            fig = px.line(
                df,
                x="n_components",
                y=metric,
                color="classifier",
                markers=True,
                title=title,
                labels={"n_components": "# PCA components", metric: title},
            )
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            return fig
        except Exception:
            return None

    figs = {
        "accuracy": _metric_plot(results_df, "accuracy", "Accuracy vs PCA components"),
        "f1_weighted": _metric_plot(
            results_df, "f1_weighted", "F1 (weighted) vs PCA components"
        ),
        "precision_weighted": _metric_plot(
            results_df, "precision_weighted", "Precision (weighted) vs PCA components"
        ),
        "recall_weighted": _metric_plot(
            results_df, "recall_weighted", "Recall (weighted) vs PCA components"
        ),
    }

    with col1:
        if figs["accuracy"] is not None:
            st.plotly_chart(figs["accuracy"], use_container_width=True)
        else:
            st.info("Accuracy plot not available.")

        if figs["f1_weighted"] is not None:
            st.plotly_chart(figs["f1_weighted"], use_container_width=True)
        else:
            st.info("F1 (weighted) plot not available.")

    with col2:
        if figs["precision_weighted"] is not None:
            st.plotly_chart(figs["precision_weighted"], use_container_width=True)
        else:
            st.info("Precision (weighted) plot not available.")

        if figs["recall_weighted"] is not None:
            st.plotly_chart(figs["recall_weighted"], use_container_width=True)
        else:
            st.info("Recall (weighted) plot not available.")

    st.success("Experiments complete.")
