import streamlit as st


def sidebar_setup():
    # Sidebar - User Inputs
    st.sidebar.title("How to use this app")
    st.sidebar.markdown(
        """
    This Streamlit app allows you to explore and analyze thyroid disease data.  
    Use the top navigation bar to navigate between different sections:
    - **Overview**: Get an introduction to thyroid diseases and the dataset used.
    - **Exploratory Data Analysis (EDA)**: Visualize and explore the dataset with interactive plots.
    - **Modelling**: Experiment with machine learning models to predict thyroid disease.
    - **Prediction**: Input patient data to get thyroid disease predictions (coming soon).
    You can adjust parameters and settings in each section to see how they affect the analysis and results.
    """
    )
