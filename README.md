# Thyroid Data Analysis Dashboard

A Streamlit-based interactive dashboard for thyroid data analysis with comprehensive visualizations and patient analysis tools.

## Features

- **Side Panel with User Inputs**: Interactive sidebar for entering patient information, thyroid measurements, and analysis options
- **Three Main Tabs**:
  - 📊 **Data Overview**: Summary statistics, sample dataset, and statistical summaries
  - 📈 **Visualizations**: Interactive charts including TSH distribution, age distribution, T3 vs T4 scatter plots, and hormone level comparisons
  - 🔍 **Analysis**: Patient-specific analysis with reference ranges and interpretation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IuliaBunescu/thyroid-data-analysis.git
cd thyroid-data-analysis
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit dashboard:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Dashboard Components

### Sidebar Inputs
- **Patient Information**: Age and Gender
- **Thyroid Measurements**: TSH, T3, T4, and Free T4 levels
- **Analysis Options**: Reference range display and sample data size

### Main Dashboard
- **Data Overview Tab**: View dataset statistics and sample data
- **Visualizations Tab**: Interactive charts with Plotly for data exploration
- **Analysis Tab**: Patient-specific interpretation with reference ranges

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Plotly**: Interactive visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.