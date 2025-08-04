@echo off
echo Installing required packages...
echo.
echo Trying standard installation...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo Standard installation failed. Trying alternative method...
    echo Installing packages individually...
    pip install streamlit pandas numpy plotly seaborn matplotlib scikit-learn pyarrow
)

echo.
echo Starting Streamlit application...
echo Open your browser and go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run streamlit_app.py
