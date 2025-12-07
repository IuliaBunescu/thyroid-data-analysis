import streamlit as st


def general_prediction_section():
    st.info(
        "Work in progress. Will be added once the ML modelling is ready and users can make predictions based on their inputs."
    )

    # # Sample input fields for thyroid data analysis
    st.subheader("Patient Information")
    st.session_state.age = st.slider("Age", 1, 100, 35, disabled=True)
    st.session_state.gender = st.selectbox("Gender", ["Male", "Female"], disabled=True)

    st.subheader("Thyroid Measurements")
    st.session_state.tsh = st.number_input(
        "TSH Level (mIU/L)", 0.0, 10.0, 2.5, 0.1, disabled=True
    )
    st.session_state.t3 = st.number_input(
        "T3 Level (ng/dL)", 0.0, 300.0, 120.0, 1.0, disabled=True
    )
    st.session_state.t4 = st.number_input(
        "T4 Level (μg/dL)", 0.0, 20.0, 8.0, 0.1, disabled=True
    )
    st.session_state.ft4 = st.number_input(
        "Free T4 (ng/dL)", 0.0, 3.0, 1.2, 0.1, disabled=True
    )
