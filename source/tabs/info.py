import pandas as pd
import streamlit as st


def general_readme_structure():
    """
    General structure for the ReadMe tab
    """
    st.header("Dataset")
    st.info("If you use the dataset in publications, please cite the original data as:")
    # human-readable citation
    citation = (
        "Quinlan, Ross (1986). Thyroid Disease. UCI Machine Learning Repository. "
        "DOI: https://doi.org/10.24432/C5D010"
    )
    st.write(citation)

    # BibTeX formatted citation
    bibtex = """@misc{thyroid_disease_102,
  author       = {Quinlan, Ross},
  title        = {{Thyroid Disease}},
  year         = {1986},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5D010}
}"""
    st.code(bibtex, language="bibtex")
