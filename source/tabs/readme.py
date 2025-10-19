import pandas as pd
import streamlit as st


def general_readme_structure(
    age: int, gender: str, tsh: float, t3: float, t4: float, ft4: float
):
    """
    General structure for the ReadMe tab
    Inputs:
    - age: Patient age
    - gender: Patient gender
    - tsh: TSH level
    - t3: T3 level
    - t4: T4 level
    - ft4: Free T4 level
    """
    st.header("ReadMe")
