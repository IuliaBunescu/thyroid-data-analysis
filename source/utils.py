import os
from pathlib import Path

import joblib
import plotly.express as px
import streamlit as st

from source.config import AXIS_TICK_FONT_SIZE, AXIS_TITLE_FONT_SIZE, TITLE_FONT_SIZE


# Helper to load custom CSS
def load_css(file_path):
    """Inject custom CSS from disk into the Streamlit app.

    Args:
        file_path (str | pathlib.Path): Location of the CSS file to load.

    Returns:
        None: The function writes the stylesheet to the Streamlit page.
    """
    path = Path(file_path)
    if not path.exists():
        st.warning(f"CSS file not found: {path}")
        return

    with path.open("r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def apply_standard_layout(
    fig,
    *,
    show_legend: bool | None = True,
    extra_layout: dict | None = None,
):
    """Apply consistent styling to Plotly figures across the application.

    Args:
        fig (plotly.graph_objects.Figure): Figure to style.
        show_legend (bool | None): Whether to show legend (None keeps current setting).
        extra_layout (dict | None): Additional layout parameters applied after defaults.

    Returns:
        plotly.graph_objects.Figure: The updated figure (for chaining).
    """

    base_layout = {
        "margin": dict(l=0, r=0, t=30, b=20),
        "hoverlabel": dict(font=dict(size=AXIS_TICK_FONT_SIZE)),
        "xaxis": dict(
            title_font=dict(size=AXIS_TITLE_FONT_SIZE),
            tickfont=dict(size=AXIS_TICK_FONT_SIZE),
        ),
        "yaxis": dict(
            title_font=dict(size=AXIS_TITLE_FONT_SIZE),
            tickfont=dict(size=AXIS_TICK_FONT_SIZE),
            automargin=True,
        ),
        "title": dict(font=dict(size=TITLE_FONT_SIZE)),
    }

    if show_legend is True:
        base_layout["legend"] = dict(font=dict(size=AXIS_TICK_FONT_SIZE))
    elif show_legend is False:
        base_layout["showlegend"] = False

    fig.update_layout(**base_layout)

    if extra_layout:
        fig.update_layout(**extra_layout)

    return fig


# Helper to build a line plot for a metric
def _metric_plot(df, metric, title):
    """Create a metric line plot summarizing model scores by feature count.

    Args:
        df (pandas.DataFrame): DataFrame containing modelling results.
        metric (str): Column name holding the metric to plot on the y-axis.
        title (str): Title to display on the Plotly figure.

    Returns:
        plotly.graph_objects.Figure | None: The configured line chart, or None if plotting fails.
    """
    try:
        fig = px.line(
            df,
            x="n_components",
            y=metric,
            color="classifier",
            markers=True,
            title=title,
            labels={"n_components": "# Features", metric: title},
        )
        return apply_standard_layout(
            fig, extra_layout={"margin": dict(l=20, r=20, t=40, b=20)}
        )
    except Exception:
        return None


def resolve_parallel_jobs(default: int = 1) -> int:
    """Resolve desired parallel worker count from Streamlit secrets or environment."""

    candidates = []
    try:
        candidates.append(st.secrets.get("PARALLEL_JOBS"))
    except Exception:
        candidates.append(None)
    candidates.append(os.environ.get("PARALLEL_JOBS"))

    for value in candidates:
        if value is None:
            continue
        try:
            jobs = int(value)
            if jobs != 0:
                return max(1, jobs)
        except (TypeError, ValueError):
            continue

    return max(1, default)


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str, *, force_single_thread: bool = True):
    """Load a persisted model artifact and optionally cap its parallelism.

    Args:
        model_path (str): Path to the persisted model file to load.
        force_single_thread (bool, optional): If True, set the model's parallelism to a single thread. Defaults to True.

    Returns:
        tuple: (model, feature_count)
            model: The loaded model object.
            feature_count (int): The number of features the model was trained with.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found at {path}")

    saved = joblib.load(path)
    model = saved.get("model")
    if model is None:
        raise ValueError("Saved object missing 'model'.")

    feature_count = int(saved.get("feature_count", 8))

    if force_single_thread:
        try:
            model.set_params(n_jobs=1)
        # Some models do not support the 'n_jobs' parameter or do not implement set_params;
        # safely ignore these exceptions as single-threading is a best-effort operation.
        except (TypeError, ValueError, AttributeError):
            pass

    return model, feature_count
