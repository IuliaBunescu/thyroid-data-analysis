import plotly.express as px
import streamlit as st


# Helper to load custom CSS
def load_css(file_path):
    """Inject custom CSS from disk into the Streamlit app.

    Args:
        file_path (str | pathlib.Path): Location of the CSS file to load.

    Returns:
        None: The function writes the stylesheet to the Streamlit page.
    """
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig
    except Exception:
        return None
