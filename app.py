import pathlib

import pandas as pd
import source.sidebar as sb
import source.tabs.eda as eda
import source.tabs.ida as ida
import source.tabs.info as info
import source.tabs.modelling as modelling
import source.tabs.prediction as prediction
import streamlit as st
from source.utils import load_css

# Page configuration
st.set_page_config(
    page_title="Thyroid Data Analysis",
    page_icon="💗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = pathlib.Path("assets/style.css")
load_css(css_path)

# Title
st.title("Thyroid Data Analysis Dashboard")
st.info(
    "Please refer to the sidebar for scope and navigation instructions and to the Info tab for background and citation information."
)

sb.sidebar_setup()

data = pd.read_csv("data/thyroid_data.csv", index_col="patient_id")
lab_references = pd.read_csv("data/lab_reference_intervals.csv")
condition_codes = pd.read_csv("data/condition_codes.csv")
target_data = pd.read_csv("data/thyroid_data_target.csv", index_col="patient_id")


# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["IDA", "EDA", "Modelling", "Prediction", "Info"]
)

with tab1:
    ida.general_ida_structure(data, lab_references, condition_codes)
with tab2:
    eda.general_eda_structure(data, target_data, condition_codes)
with tab3:
    modelling.general_modelling_structure()
with tab4:
    prediction.general_prediction_section(lab_references)
with tab5:
    info.general_readme_structure()

# Footer
st.sidebar.markdown("---")
st.sidebar.info("**Thyroid Data Analysis Dashboard**\n\nVersion 1.0")
