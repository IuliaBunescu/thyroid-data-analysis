import plotly.express as px
import streamlit as st


# Helper to load custom CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


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
            labels={"n_components": "# Features", metric: title},
        )
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig
    except Exception:
        return None
