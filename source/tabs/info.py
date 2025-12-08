import streamlit as st


def general_readme_structure():
    """Render the informational tab with background notes and references.

    Args:
        None: This function accepts no parameters.

    Returns:
        None: Content is written directly to the Streamlit info tab.
    """

    st.header("About This Dashboard")
    st.markdown(
        """
        This Streamlit project bundles a complete workflow around the UCI **Thyroid Disease** dataset.
        It combines exploratory analysis, careful preprocessing, modelling, and a point-of-care style
        prediction form so that clinicians or analysts can experiment with thyroid lab measurements
        and understand how the selected model behaves.
        """
    )

    st.header("How the App Is Organised")
    st.markdown(
        """
        1. **Initial Data Analysis (IDA)** – examine data quality, missingness patterns, and reference
           intervals with guided commentary.
        2. **Exploratory Data Analysis (EDA)** – inspect target balances, multivariate relationships, and
           feature importance side by side with medical context.
        3. **Modelling** – run and review cross-validated experiments, compare classifiers, and evaluate the
           final Random Forest model with labelled confusion matrices and feature importances.
        4. **Prediction** – enter patient measurements using lab reference hints to obtain class probabilities
           mapped to human-readable thyroid conditions.
        """
    )

    st.header("Using the Prediction Tab")
    st.markdown(
        """
        - Provide values for every feature listed; boolean inputs are entered as **Yes/No** and mapped to 1/0.
        - Numerical inputs display units and normal laboratory ranges derived from the loaded reference intervals.
        - Predictions return the most likely thyroid condition plus a probability distribution across all classes.
        - Result interpretation is exploratory only and should be corroborated by a clinician.
        """
    )

    st.header("Dataset & Citation")
    st.info("If you use the dataset in publications, please cite the original data as:")
    citation = (
        "Quinlan, Ross (1986). Thyroid Disease. UCI Machine Learning Repository. "
        "DOI: https://doi.org/10.24432/C5D010"
    )
    st.write(citation)

    bibtex = """@misc{thyroid_disease_102,
  author       = {Quinlan, Ross},
  title        = {{Thyroid Disease}},
  year         = {1986},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5D010}
}"""
    st.code(bibtex, language="bibtex")

    st.header("Further Reading & Resources")
    st.markdown(
        """
        - [Project README](https://github.com/IuliaBunescu/thyroid-data-analysis#readme) – documentation, setup instructions, and screenshots.
        - [Lab Reference Intervals](https://en.wikipedia.org/wiki/Thyroid_function_tests) – external overview of thyroid function testing.
        - [Streamlit Documentation](https://docs.streamlit.io/) – customise or deploy additional dashboards.
        """
    )
