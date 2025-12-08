import streamlit as st


def sidebar_setup():
    """Render sidebar content with overview, navigation, and disclaimers.

    Args:
        None: This function does not accept any parameters.

    Returns:
        None: The Streamlit sidebar is populated with guidance and context.
    """
    # Sidebar - User Inputs
    st.sidebar.title("Welcome")

    st.sidebar.header("About This App")
    st.sidebar.markdown(
        """
    This app helps you understand and model thyroid disease data.
    You can explore the dataset, learn what each lab test means,
    build simple machine learning models, and try predictions.
    Each step explains the reasoning and the choices made.
    """
    )
    st.sidebar.markdown("---")
    st.sidebar.header("How To Navigate")
    st.sidebar.markdown(
        """
    Use the tabs at the top to switch sections:
    - **IDA** — Initial Data Analysis: Background on thyroid conditions and the dataset.
    - **EDA** — Exploratory Data Analysis: Visualize distributions, relationships, and quality checks.
    - **Modelling**: Split the data, cross‑validate, and train classifiers.
    - **Prediction**: Enter patient values and get predicted condition probabilities.
    - **Info**: Data sources, references, and additional notes.
    """
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Quick Start")
    st.sidebar.markdown(
        """
        1) Start with **IDA** to understand the dataset and targets.
        2) Explore patterns in **EDA**.
        3) Understand the models trained in **Modelling** (or train your own - time consuming).
        4) Use **Prediction** to input values and see results.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.subheader("Lab Test Abbreviations")
    st.sidebar.markdown(
        """
        - **TSH** — Thyroid Stimulating Hormone (mIU/L). High often suggests hypothyroidism; low suggests hyperthyroidism.
        - **T3** — Triiodothyronine (pg/mL). High in hyperthyroidism; low in hypothyroidism or illness.
        - **TT4** — Total Thyroxine (µg/dL). Overall thyroid hormone production.
        - **T4U** — Thyroxine Binding Ratio (unitless). Reflects binding protein activity (TBG).
        - **FTI** — Free Thyroxine Index (index units). Proxy for free thyroid hormone activity.
        - **TBG** — Thyroxine Binding Globulin (µg/mL). Binding protein affecting total hormone levels.
        """
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Disclaimer")
    st.sidebar.info(
        "**Prototype — not for clinical use.**\n\n"
        "Content is for learning only and may be incomplete or simplified. "
        "Do not use for diagnosis or treatment. Always consult a qualified healthcare professional."
    )
