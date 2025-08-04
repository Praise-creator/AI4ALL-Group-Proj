# Optiver Volatility Prediction Analysis Dashboard

This Streamlit application provides an interactive dashboard to explore and visualize the results of the Optiver Realized Volatility Prediction analysis.

## Features

📋 **Dataset Overview**
- Comprehensive dataset statistics
- Target variable distribution
- Data sample exploration

🔧 **Feature Analysis**
- Feature-target correlations
- Feature importance ranking
- Interactive correlation visualization

🎯 **Model Performance**
- Linear regression model results
- Cross-validation metrics
- Feature importance analysis
- Model comparison

📈 **Visualizations**
- Correlation heatmaps
- Time series analysis
- Volatility distribution plots

🔍 **Data Explorer**
- Interactive data filtering
- Custom analysis capabilities
- Data download functionality

## Installation & Setup

### **Method 1: Standard Installation**
```bash
pip install -r requirements.txt
```

### **Method 2: If you encounter Python 3.13 compatibility issues**
```bash
# Install packages individually with latest versions
pip install streamlit pandas numpy plotly seaborn matplotlib scikit-learn pyarrow
```

### **Method 3: Using conda (recommended for Python 3.13)**
```bash
conda install -c conda-forge streamlit pandas numpy plotly seaborn matplotlib scikit-learn pyarrow
```

### **Run the Application**
```bash
streamlit run streamlit_app.py
```

### **Open your browser** and navigate to `http://localhost:8501`

## Troubleshooting

**If you get `streamlit not recognized` error:**
1. Install streamlit: `pip install streamlit`
2. Try: `python -m streamlit run streamlit_app.py`
3. Or: `py -m streamlit run streamlit_app.py`

**If you get `distutils` errors with Python 3.13:**
1. Try Method 2 or 3 above
2. Or downgrade to Python 3.11/3.12: `conda install python=3.11`

**If packages are missing:**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**If permission errors:**
```bash
pip install --user streamlit pandas numpy plotly seaborn matplotlib scikit-learn pyarrow
```

## Data Requirements

The application expects the following files in the `notebooks/` directory:
- `training_table.parquet` (preferred) or `training_table.csv`

These files should contain the engineered features from your volatility prediction analysis.

## Usage

1. **Navigate** through different sections using the sidebar
2. **Explore** dataset statistics and feature correlations
3. **Analyze** model performance and feature importance
4. **Visualize** data patterns and relationships
5. **Filter** and download data using the Data Explorer

## Key Insights

The dashboard reveals that:
- **Price spread features** are the strongest volatility predictors
- **Log-transformed volatility measures** show high correlation with target
- The **linear regression model** explains ~77% of volatility variance
- **RMSE of 0.001397** indicates good prediction accuracy

## Technical Details

- **Model**: Linear Regression with StandardScaler
- **Validation**: Time Series Cross-Validation (5 folds)
- **Features**: Top 10 most correlated features selected
- **Target**: Realized volatility (10-minute intervals)

## Project Structure

```
AI4ALL-Group-Proj/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── notebooks/
│   ├── featureengineering.ipynb    # Feature engineering notebook
│   ├── MLModel_Optiver.ipynb       # ML model development
│   ├── training_table.parquet      # Processed training data
│   └── training_table.csv          # Backup CSV format
└── data/                     # Raw data files
```

---


