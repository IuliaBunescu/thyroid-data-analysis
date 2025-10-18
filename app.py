"""
Thyroid Data Analysis Dashboard
A Streamlit dashboard with sidebar inputs and three main tabs
"""

import pathlib

import pandas as pd
import source.sidebar as sb
import source.tabs.eda as eda
import source.tabs.ida as ida
import source.tabs.readme as readme
import streamlit as st
from source.utils import load_css

# Page configuration
st.set_page_config(
    page_title="Thyroid Data Analysis",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded",
)

css_path = pathlib.Path("assets/style.css")
load_css(css_path)

# Title
st.title("Thyroid Data Analysis Dashboard")

# Medical disclaimer at the top
st.info(
    "**This is a prototype application and not for clinical use.**\n\n"
    "The data and analyses presented here are for educational purposes only. "
    "They should not be used for medical diagnosis or treatment without consulting a qualified healthcare professional."
)

sb.sidebar_setup()

data = pd.read_csv("data/thyroid_data.csv")

# Create tabs
tab1, tab2, tab3 = st.tabs(["IDA", "EDA", "ReadMe"])

with tab1:
    ida.general_ida_structure(data)
with tab2:
    eda.general_eda_structure(data)
with tab3:
    readme.general_readme_structure(
        st.session_state.age,
        st.session_state.gender,
        st.session_state.tsh,
        st.session_state.t3,
        st.session_state.t4,
        st.session_state.ft4,
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("**Thyroid Data Analysis Dashboard**\n\nVersion 1.0")
