# Stock Volatility Prediction - AI4ALL Group Project

## Setup Instructions
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Kaggle API credentials
4. Download dataset to `./data/` folder

## Running the Streamlit App
1. Navigate to the project directory
2. Run: `streamlit run streamlit_app.py`
3. Open your browser to the local URL shown in the terminal (usually http://localhost:8501)

Alternatively, you can run the batch file on Windows: `run_streamlit.bat`

## Data Analysis
Run the Jupyter notebooks in the `notebooks/` folder for detailed analysis and model training.




#  REALIZED VOLATILITY PREDICTION - AI4ALL PORTFOLIO PROJECT
## Benjamin Silva, Juan Buitrago, and Praise Olatide
An ML model used to predict realized volatility over 10-minute intervals.

## Problem Statement
This project was developed out of interest in working with real-world competition data from the OPTIVER REALIZED VOLATILITY PREDICTION KAGGLE CHALLENGE.] 


## Key Results
Linear Regression (10 top correlated features, RMSE):

  -Cross-Validation RMSE: 0.001397 (not directly comparable to leaderboard RMSPE).

  -R² Score: ~0.7705

LightGBM Model (full feature set, RMSPE — comparable to competition metric):
  -Training RMSPE: 0.286029

  -Validation RMSPE: 0.298207

  -Validation R² Score: 0.782160

Feature analysis revealed spread-related features (price_spread_avg_mean, price_spread_1_mean) as the strongest predictors.


## Methodologies
Data Preparation
  Merged and processed Kaggle-provided order book and trade data.
  
  Engineered 46 statistical and volatility-based features per stock-time combination.
  
  Selected top features using correlation with target value.

Validation Strategy
  Sequential split: train on earlier periods, validate on later periods to mimic real-world forecasting.
  
  Avoided random splits to preserve time-series structure.

Models Used
  Linear Regression: simple baseline using top-10 correlated features, scaled with StandardScaler.
  
  LightGBM: gradient boosting model using the full feature set for improved accuracy.

Evaluation
  RMSE used for internal Linear Regression evaluation.
  
  RMSPE (competition metric) used for LightGBM to match leaderboard scoring.

## Data Sources
Kaggle Dataset: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/overview 


## Technologies Used
  Python
  
  pandas, NumPy (data manipulation)
  
  scikit-learn (Linear Regression, scaling, cross-validation)
  
  LightGBM (gradient boosting model)
  
  matplotlib, seaborn (visualization)

  

## Authors
This project was completed in collaboration with:

Benjamin Silva 
Juan Buitrago
Praise Olatide
