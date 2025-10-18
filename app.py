"""
Thyroid Data Analysis Dashboard
A Streamlit dashboard with sidebar inputs and three main tabs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Thyroid Data Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🏥 Thyroid Data Analysis Dashboard")

# Medical disclaimer at the top
st.info("⚕️ **Medical Disclaimer:** This is a demonstration dashboard for educational purposes only. Please consult with a healthcare professional for actual medical advice and diagnosis.")

# Sidebar - User Inputs
st.sidebar.header("User Inputs")

# Sample input fields for thyroid data analysis
st.sidebar.subheader("Patient Information")
age = st.sidebar.slider("Age", 1, 100, 35)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])

st.sidebar.subheader("Thyroid Measurements")
tsh = st.sidebar.number_input("TSH Level (mIU/L)", 0.0, 10.0, 2.5, 0.1)
t3 = st.sidebar.number_input("T3 Level (ng/dL)", 0.0, 300.0, 120.0, 1.0)
t4 = st.sidebar.number_input("T4 Level (μg/dL)", 0.0, 20.0, 8.0, 0.1)
ft4 = st.sidebar.number_input("Free T4 (ng/dL)", 0.0, 3.0, 1.2, 0.1)

st.sidebar.subheader("Analysis Options")
show_reference_range = st.sidebar.checkbox("Show Reference Range", value=True)
data_points = st.sidebar.slider("Number of Sample Data Points", 10, 100, 50)

# Generate sample data based on inputs
# Using seed 42 for reproducibility of demo data
np.random.seed(42)
sample_data = pd.DataFrame({
    'Patient_ID': range(1, data_points + 1),
    'Age': np.random.randint(18, 80, data_points),
    'Gender': np.random.choice(['Male', 'Female'], data_points),
    'TSH': np.random.uniform(0.5, 8.0, data_points),
    'T3': np.random.uniform(80, 200, data_points),
    'T4': np.random.uniform(4.5, 12.0, data_points),
    'FT4': np.random.uniform(0.8, 1.8, data_points)
})

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔍 Analysis"])

# Tab 1: Data Overview
with tab1:
    st.header("Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patients", len(sample_data))
    with col2:
        st.metric("Average Age", f"{sample_data['Age'].mean():.1f}")
    with col3:
        st.metric("Male/Female Ratio", f"{(sample_data['Gender'] == 'Male').sum()}/{(sample_data['Gender'] == 'Female').sum()}")
    with col4:
        st.metric("Avg TSH Level", f"{sample_data['TSH'].mean():.2f}")
    
    st.subheader("Sample Dataset")
    st.dataframe(sample_data, use_container_width=True, height=400)
    
    st.subheader("Statistical Summary")
    st.dataframe(sample_data.describe(), use_container_width=True)

# Tab 2: Visualizations
with tab2:
    st.header("Data Visualizations")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("TSH Distribution")
        fig1 = px.histogram(
            sample_data, 
            x='TSH', 
            nbins=20,
            title='Distribution of TSH Levels',
            labels={'TSH': 'TSH Level (mIU/L)', 'count': 'Frequency'}
        )
        if show_reference_range:
            fig1.add_vline(x=0.5, line_dash="dash", line_color="green", annotation_text="Min Normal")
            fig1.add_vline(x=4.5, line_dash="dash", line_color="red", annotation_text="Max Normal")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("T3 vs T4 Scatter Plot")
        fig3 = px.scatter(
            sample_data,
            x='T3',
            y='T4',
            color='Gender',
            title='T3 vs T4 Levels by Gender',
            labels={'T3': 'T3 Level (ng/dL)', 'T4': 'T4 Level (μg/dL)'}
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution by Gender")
        fig2 = px.box(
            sample_data,
            x='Gender',
            y='Age',
            title='Age Distribution by Gender',
            labels={'Age': 'Age (years)'}
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Thyroid Hormone Levels")
        # Scale T3 and FT4 for better visualization comparison across different units
        # T3 is divided by 10 (ng/dL) and FT4 is multiplied by 5 (ng/dL) to bring values to comparable scale
        avg_levels = pd.DataFrame({
            'Hormone': ['TSH', 'T3', 'T4', 'FT4'],
            'Average': [
                sample_data['TSH'].mean(),
                sample_data['T3'].mean() / 10,  # Scaled: ng/dL ÷ 10
                sample_data['T4'].mean(),
                sample_data['FT4'].mean() * 5   # Scaled: ng/dL × 5
            ]
        })
        fig4 = px.bar(
            avg_levels,
            x='Hormone',
            y='Average',
            title='Average Hormone Levels (Scaled for Comparison)',
            labels={'Average': 'Average Level (Scaled)'}
        )
        fig4.add_annotation(
            text="Note: T3 and FT4 values are scaled for visualization purposes",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            showarrow=False,
            font=dict(size=10)
        )
        st.plotly_chart(fig4, use_container_width=True)

# Tab 3: Analysis
with tab3:
    st.header("Patient Analysis")
    
    st.subheader("Current Patient Input")
    
    # Display current patient info
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Patient Demographics:**")
        st.write(f"- Age: {age} years")
        st.write(f"- Gender: {gender}")
    
    with col2:
        st.write("**Thyroid Measurements:**")
        st.write(f"- TSH: {tsh:.2f} mIU/L")
        st.write(f"- T3: {t3:.1f} ng/dL")
        st.write(f"- T4: {t4:.1f} μg/dL")
        st.write(f"- Free T4: {ft4:.2f} ng/dL")
    
    st.subheader("Reference Ranges")
    reference_data = pd.DataFrame({
        'Measurement': ['TSH', 'T3', 'T4', 'Free T4'],
        'Normal Range': ['0.5 - 4.5 mIU/L', '80 - 200 ng/dL', '4.5 - 12.0 μg/dL', '0.8 - 1.8 ng/dL'],
        'Your Value': [f"{tsh:.2f}", f"{t3:.1f}", f"{t4:.1f}", f"{ft4:.2f}"],
        'Status': [
            'Normal' if 0.5 <= tsh <= 4.5 else 'Outside Range',
            'Normal' if 80 <= t3 <= 200 else 'Outside Range',
            'Normal' if 4.5 <= t4 <= 12.0 else 'Outside Range',
            'Normal' if 0.8 <= ft4 <= 1.8 else 'Outside Range'
        ]
    })
    st.dataframe(reference_data, use_container_width=True, hide_index=True)
    
    # Interpretation
    st.subheader("Interpretation")
    
    issues = []
    if tsh < 0.5:
        issues.append("⚠️ TSH level is below normal range - may indicate hyperthyroidism")
    elif tsh > 4.5:
        issues.append("⚠️ TSH level is above normal range - may indicate hypothyroidism")
    
    if t3 < 80:
        issues.append("⚠️ T3 level is below normal range")
    elif t3 > 200:
        issues.append("⚠️ T3 level is above normal range")
    
    if t4 < 4.5:
        issues.append("⚠️ T4 level is below normal range")
    elif t4 > 12.0:
        issues.append("⚠️ T4 level is above normal range")
    
    if ft4 < 0.8:
        issues.append("⚠️ Free T4 level is below normal range")
    elif ft4 > 1.8:
        issues.append("⚠️ Free T4 level is above normal range")
    
    if issues:
        for issue in issues:
            st.warning(issue)
    else:
        st.success("✅ All thyroid measurements are within normal ranges")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("**Thyroid Data Analysis Dashboard**\n\nVersion 1.0")
