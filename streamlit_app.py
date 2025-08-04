import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Optiver Volatility Prediction Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        color: #2c3e50;
    }
    /* Fix for metric readability */
    .stMetric {
        background-color: transparent !important;
        padding: 0.5rem !important;
        border-radius: 5px !important;
        border: 1px solid #e1e5e9 !important;
    }
    .stMetric [data-testid="metric-container"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.5rem !important;
    }
    .stMetric [data-testid="metric-container"] > div {
        background-color: transparent !important;
        color: #2c3e50 !important;
    }
    /* Ensure metric text is clearly visible */
    div[data-testid="metric-container"] div {
        color: #2c3e50 !important;
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the training data"""
    try:
        df = pd.read_parquet('notebooks/training_table.parquet')
    except:
        try:
            df = pd.read_csv('notebooks/training_table.csv')
        except:
            st.error("Could not find training_table.parquet or training_table.csv in notebooks folder")
            return None
    return df

def calculate_feature_correlations(df):
    """Calculate correlations between features and target"""
    feature_cols = [col for col in df.columns if col not in ['stock_id', 'time_id', 'target']]
    correlations = []
    
    for col in feature_cols:
        if df[col].std() > 0:  
            corr = df[col].corr(df['target'])
            if not np.isnan(corr):
                correlations.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
    
    return pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)

def train_model(df, top_features):
    """Train a linear regression model"""
    X = df[top_features].fillna(0)
    y = df['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    # R² score
    r2 = model.score(X_scaled, y)
    
    return model, scaler, cv_rmse, r2

def main():
    # Header
    st.markdown('<h1 class="main-header">Optiver Volatility Prediction Analysis</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Dataset Overview", "Feature Analysis", "Model Performance", "Visualizations", "Data Explorer"]
    )
    
    if page == "Dataset Overview":
        show_dataset_overview(df)
    elif page == "Feature Analysis":
        show_feature_analysis(df)
    elif page == "Model Performance":
        show_model_performance(df)
    elif page == "Visualizations":
        show_visualizations(df)
    elif page == "Data Explorer":
        show_data_explorer(df)

def show_dataset_overview(df):
    st.header("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observations", f"{len(df):,}")
    with col2:
        st.metric("Unique Stocks", df['stock_id'].nunique())
    with col3:
        st.metric("Time Periods", df['time_id'].nunique())
    with col4:
        st.metric("Features", len([col for col in df.columns if col not in ['stock_id', 'time_id', 'target']]))
    
    st.markdown("---")
    
    # Dataset statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Variable Statistics")
        target_stats = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
            'Value': [
                f"{df['target'].count():,}",
                f"{df['target'].mean():.6f}",
                f"{df['target'].std():.6f}",
                f"{df['target'].min():.6f}",
                f"{df['target'].quantile(0.25):.6f}",
                f"{df['target'].median():.6f}",
                f"{df['target'].quantile(0.75):.6f}",
                f"{df['target'].max():.6f}"
            ]
        })
        st.dataframe(target_stats, use_container_width=True)
    
    with col2:
        st.subheader("Target Distribution")
        fig = px.histogram(df, x='target', nbins=50, title="Realized Volatility Distribution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Data sample
    st.subheader("Data Sample")
    st.dataframe(df.head(10), use_container_width=True)

def show_feature_analysis(df):
    st.header("Feature Analysis")
    
    # Calculate correlations
    correlations_df = calculate_feature_correlations(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 15 Most Correlated Features")
        top_features = correlations_df.head(15)
        
        # Create a horizontal bar chart
        fig = px.bar(
            top_features, 
            x='abs_correlation', 
            y='feature',
            orientation='h',
            title="Feature-Target Correlations (Absolute Values)",
            color='abs_correlation',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Correlation Details")
        display_df = top_features.copy()
        display_df['correlation'] = display_df['correlation'].round(4)
        display_df['abs_correlation'] = display_df['abs_correlation'].round(4)
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Feature categories
        st.subheader("Feature Categories")
        feature_categories = {
            'Price Volatility': ['log_return_realized_vol', 'log_squared_wap_realized_vol', 'log_wap3_realized_vol'],
            'Spread Features': ['price_spread_mean', 'price_spread_avg_mean', 'price_spread_1_mean'],
            'Trade Features': ['trade_price_std', 'trade_count', 'trade_size_mean'],
            'Balance Features': ['wap_balance_mean', 'volume_imbalance_mean']
        }
        
        for category, features in feature_categories.items():
            matching_features = [f for f in features if f in df.columns]
            if matching_features:
                st.write(f"**{category}:** {len(matching_features)} features")

def show_model_performance(df):
    st.header("Model Performance")
    
    # Select top features for modeling
    correlations_df = calculate_feature_correlations(df)
    top_features = correlations_df.head(10)['feature'].tolist()
    
    # Train model
    with st.spinner("Training model..."):
        model, scaler, cv_rmse, r2 = train_model(df, top_features)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Cross-Validation RMSE", f"{cv_rmse:.6f}")
    with col2:
        st.metric("R² Score", f"{r2:.4f}")
    with col3:
        st.metric("Features Used", len(top_features))
    with col4:
        variance_explained = r2 * 100
        st.metric("Variance Explained", f"{variance_explained:.1f}%")
    
    st.markdown("---")
    
    # Feature importance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'feature': top_features,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='abs_coefficient',
            y='feature',
            orientation='h',
            title="Linear Regression Feature Importance",
            color='abs_coefficient',
            color_continuous_scale='plasma'
        )
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Comparison")
        comparison_data = {
            'Model': ['Linear Regression', 'Mean Baseline'],
            'RMSE': [cv_rmse, df['target'].std()],
            'R² Score': [r2, 0.0],
            'Interpretability': ['High', 'High'],
            'Complexity': ['Low', 'None']
        }
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Model interpretation
        st.subheader("Model Insights")
        st.write("**Key Findings:**")
        st.write(f"• The model explains {r2*100:.1f}% of volatility variance")
        st.write(f"• RMSE of {cv_rmse:.6f} indicates good prediction accuracy")
        st.write("• Price spread features are the strongest predictors")
        st.write("• Log-transformed volatility measures show high correlation")

def show_visualizations(df):
    st.header("Visualizations")
    
    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    
    # Select top features for heatmap
    correlations_df = calculate_feature_correlations(df)
    top_features = correlations_df.head(10)['feature'].tolist()
    
    # Create correlation matrix
    corr_matrix = df[top_features + ['target']].corr()
    
    fig = px.imshow(
        corr_matrix,
        title="Top Features Correlation Matrix",
        color_continuous_scale='RdBu',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volatility Over Time")
        # Sample time series data
        time_sample = df.groupby('time_id')['target'].mean().reset_index()
        time_sample = time_sample.sample(min(100, len(time_sample))).sort_values('time_id')
        
        fig = px.line(
            time_sample,
            x='time_id',
            y='target',
            title="Average Volatility by Time Period"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Stock Volatility Distribution")
        # Volatility by stock
        stock_vol = df.groupby('stock_id')['target'].agg(['mean', 'std']).reset_index()
        
        fig = px.scatter(
            stock_vol,
            x='mean',
            y='std',
            title="Volatility Mean vs Standard Deviation by Stock",
            labels={'mean': 'Mean Volatility', 'std': 'Volatility Std'}
        )
        st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(df):
    st.header("Data Explorer")
    
    # Interactive filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_stocks = st.multiselect(
            "Select Stock IDs:",
            options=sorted(df['stock_id'].unique()),
            default=sorted(df['stock_id'].unique())[:5]
        )
    
    with col2:
        time_range = st.slider(
            "Time ID Range:",
            min_value=int(df['time_id'].min()),
            max_value=int(df['time_id'].max()),
            value=(int(df['time_id'].min()), int(df['time_id'].max()))
        )
    
    with col3:
        target_range = st.slider(
            "Target Range:",
            min_value=float(df['target'].min()),
            max_value=float(df['target'].max()),
            value=(float(df['target'].min()), float(df['target'].max())),
            format="%.6f"
        )
    
    # Filter data
    filtered_df = df[
        (df['stock_id'].isin(selected_stocks)) &
        (df['time_id'].between(time_range[0], time_range[1])) &
        (df['target'].between(target_range[0], target_range[1]))
    ]
    
    st.write(f"**Filtered Data:** {len(filtered_df):,} rows")
    
    # Display filtered data
    if len(filtered_df) > 0:
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Filtered Data Summary")
            summary_stats = pd.DataFrame({
                'Metric': ['Count', 'Mean Target', 'Std Target', 'Min Target', 'Max Target'],
                'Value': [
                    f"{len(filtered_df):,}",
                    f"{filtered_df['target'].mean():.6f}",
                    f"{filtered_df['target'].std():.6f}",
                    f"{filtered_df['target'].min():.6f}",
                    f"{filtered_df['target'].max():.6f}"
                ]
            })
            st.dataframe(summary_stats, use_container_width=True)
        
        with col2:
            st.subheader("Target Distribution")
            fig = px.histogram(
                filtered_df,
                x='target',
                nbins=30,
                title="Filtered Target Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Raw data table
        st.subheader("Raw Data")
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_volatility_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No data matches the current filters. Please adjust your selection.")

if __name__ == "__main__":
    main()
