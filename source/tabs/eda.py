import pandas as pd
import plotly.express as px
import streamlit as st


def general_eda_structure(df: pd.DataFrame):
    """
    General structure for the EDA tab
    Inputs:
    - df: DataFrame containing thyroid data
    """
    st.header("EDA")
