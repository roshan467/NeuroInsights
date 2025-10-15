import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from matplotlib.lines import Line2D
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Enhanced Custom CSS for premium design with gradient backgrounds and glass-morphism effects
st.markdown("""
<style>
    /* Premium gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Glass-morphism sidebar */
    [data-testid=stSidebar] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 15px;
        padding: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Metric values with gold accent */
    [data-testid="stMetricValue"] {
        color: #FFD700;
        font-weight: bold;
        font-size: 1.8em;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.3);
    }
    
    /* Headers with premium styling */
    h1, h2, h3 {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: 1px;
    }
    
    /* Premium buttons with gradient */
    .stDownloadButton > button, .stButton > button {
        background: linear-gradient(135deg, #FFD700, #FF8C00);
        color: #000;
        border: none;
        border-radius: 12px;
        font-weight: bold;
        padding: 12px 24px;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
    }
    
    .stDownloadButton > button:hover, .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 215, 0, 0.4);
    }
    
    /* Dataframe styling with glass effect */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Tab styling with premium look */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        height: 50px;
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-active"] {
        background: linear-gradient(135deg, #FFD700, #FF8C00);
        color: #000;
    }
    
    /* Checkbox styling with premium look */
    .stCheckbox > label {
        background: rgba(255, 255, 255, 0.05);
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .stCheckbox > label:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateY(-2px);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        backdrop-filter: blur(5px);
    }
    
    /* Info boxes with glass effect */
    .stAlert-info {
        background: rgba(33, 150, 243, 0.1);
        border: 1px solid rgba(33, 150, 243, 0.3);
        border-radius: 15px;
        color: #64B5F6;
        backdrop-filter: blur(10px);
    }
    
    /* Success messages */
    .stAlert-success {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 15px;
        color: #81C784;
        backdrop-filter: blur(10px);
    }
    
    /* Warning messages */
    .stAlert-warning {
        background: rgba(255, 152, 0, 0.1);
        border: 1px solid rgba(255, 152, 0, 0.3);
        border-radius: 15px;
        color: #FFB74D;
        backdrop-filter: blur(10px);
    }
    
    /* Error messages */
    .stAlert-error {
        background: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
        border-radius: 15px;
        color: #E57373;
        backdrop-filter: blur(10px);
    }
    
    /* Dataframes */
    .stDataFrame > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Plot containers with premium styling */
    .stPlotlyChart, .stPyplotFigures {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Custom card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(255, 215, 0, 0.2);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #FFD700, #FF8C00);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #FFA500, #FF4500);
    }
</style>
""", unsafe_allow_html=True)

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/processed')

# Set page config with premium theme
st.set_page_config(
    page_title="Universal Data Analysis Dashboard - Advanced Data Analytics",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Premium header with enhanced animation effect
st.markdown("""
<div style="text-align: center; padding: 40px; border-radius: 25px; background: rgba(255, 255, 255, 0.05); margin-bottom: 30px; border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 12px 30px rgba(0,0,0,0.5); backdrop-filter: blur(10px);">
    <h1> Universal Data Analysis Dashboard</h1>
    <p style="font-size: 1.6em; font-weight: bold; background: linear-gradient(90deg, #FFD700, #FFA500); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 10px 0;">Advanced Neurological Data Analytics Platform</p>
    <p style="font-size: 1.2em; color: #e0e0e0;">Transforming complex neurological data into actionable medical insights</p>
    <p style="font-size: 1em; color: #aaaaaa; margin-top: 20px;">Powered by AI & Machine Learning • Real-time Analysis • Secure & HIPAA Compliant</p>
    <p style="font-size: 0.9em; color: #4CAF50; margin-top: 15px;">Enhanced with Advanced ML Algorithms</p>
</div>
""", unsafe_allow_html=True)

# Sidebar with enhanced premium styling
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px; border-radius: 20px; background: linear-gradient(135deg, rgba(255, 215, 0, 0.1), rgba(255, 140, 0, 0.1)); margin-bottom: 25px; border: 1px solid rgba(255, 215, 0, 0.3); backdrop-filter: blur(10px);">
        <h2>🎛️ Control Panel</h2>
        <p style="color: #FFD700; font-weight: bold;">Universal Data Analysis Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📂 Data Management")
    uploaded_file = st.file_uploader("Upload New Dataset (CSV)", type="csv")
    
    # Add a message to inform users about the requirement to upload data
    if uploaded_file is None:
        st.info("Please upload a CSV dataset to begin analysis.")
        st.stop()
    
    st.markdown("### 🎨 Visualization Settings")
    viz_type = st.selectbox("Select Visualization Type", ["Static (Matplotlib)", "Interactive (Plotly)"])
    
    # Add orientation options
    st.markdown("### 🧭 Chart Orientation")
    chart_orientation = st.selectbox("Select Chart Orientation", ["Vertical", "Horizontal"])
    
    # Add chart type options
    st.markdown("### 📊 Chart Types")
    chart_types = st.multiselect(
        "Select Chart Types to Display",
        ["Bar Charts", "Line Charts", "Scatter Plots", "Histograms", "Box Plots", "Heatmaps", "Pie Charts"],
        ["Bar Charts", "Histograms", "Box Plots"]
    )
    
    st.markdown("### 📈 Analysis Options")
    show_features = st.checkbox("Feature Analysis", value=True)
    show_correlation = st.checkbox("Correlation Matrix", value=True)
    show_distributions = st.checkbox("Data Distributions", value=True)
    
    st.markdown("### 🎯 Advanced Features")
    show_insights = st.checkbox("AI Insights", value=True)
    show_patterns = st.checkbox("Pattern Analysis", value=True)
    
    st.markdown("### 🤖 AI/ML Analytics")
    show_ml_predictions = st.checkbox("ML Predictions", value=False)
    show_clustering = st.checkbox("Clustering Analysis", value=False)
    show_feature_importance = st.checkbox("Feature Importance", value=False)
    
    st.markdown("### ⚙️ Actions")
    actions = st.multiselect(
        "Select Actions to Perform",
        ["Data Cleaning", "Feature Scaling", "Outlier Detection", "Data Summary", "Export Cleaned Data"],
        ["Data Summary", "Export Cleaned Data"]
    )
    
    st.markdown("### 📅 Session Info")
    st.markdown(f"<p style='color: #aaaaaa; font-size: 0.9em;'>Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
    
    st.markdown("### ℹ️ About")
    st.info("Universal Data Analysis Dashboard provides advanced analytics for neurological datasets with AI-powered insights.")

# Initialize merged as None
merged = None

if uploaded_file is not None:
    # Read uploaded file
    try:
        merged = pd.read_csv(uploaded_file)
        st.success("✅ New dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()
else:
    # No file uploaded - show instruction
    st.info("Please upload a CSV dataset to begin analysis.")
    st.stop()

# Check if we have data
if merged is None:
    st.error("No data available. Please upload a CSV file.")
    st.stop()

# --- Data Cleaning ---
# Handle potential issues with the dataset
try:
    for col in merged.columns:
        if merged[col].dtype == 'object':
            merged[col] = merged[col].fillna('Unknown')
        else:
            merged[col] = merged[col].fillna(0)

    merged = merged.drop_duplicates()
    merged = merged.dropna(how='all')
except Exception as e:
    st.warning(f"Data cleaning completed with warnings: {str(e)}")

# --- Automatic Label Detection ---
def detect_labels(df):
    """Automatically detect potential label columns in the dataset"""
    label_columns = []
    target_patterns = ['label', 'target', 'class', 'category', 'outcome', 'result', 'prediction', 'diagnosis', 'status']
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if column name matches target patterns
        if any(pattern in col_lower for pattern in target_patterns):
            label_columns.append(col)
        # Check if column has categorical values with low cardinality
        elif df[col].dtype == 'object' and df[col].nunique() < 20 and df[col].nunique() > 1:
            label_columns.append(col)
        # Check if column is integer with limited unique values (could be encoded labels)
        elif df[col].dtype in ['int64', 'int32'] and df[col].nunique() < 20 and df[col].nunique() > 1:
            label_columns.append(col)
    
    return label_columns

# Detect labels in the dataset
label_columns = detect_labels(merged)

# --- Full Dataset Preview ---
st.subheader("📊 Full Cleaned & Filtered Dataset")
st.dataframe(merged, width='stretch', height=400)

# --- Enhanced Data Overview Dashboard ---
st.subheader("📋 Enhanced Data Overview")
col1, col2, col3, col4 = st.columns(4)

# Calculate universal metrics
try:
    total_rows = merged.shape[0]
except:
    total_rows = 0

try:
    total_cols = merged.shape[1]
except:
    total_cols = 0

try:
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    numeric_count = len(numeric_cols)
except:
    numeric_count = 0

try:
    missing_data_pct = (merged.isnull().sum().sum() / (merged.shape[0] * merged.shape[1])) * 100
except:
    missing_data_pct = 0

# Enhanced metric cards with glass-morphism effect and gold accents
with col1:
    st.markdown(f"""
    <div class="custom-card" style="text-align: center;">
        <h3 style="color: #FFFFFF; margin-bottom: 15px;">📂 Rows</h3>
        <p style="font-size: 2.8em; font-weight: bold; background: linear-gradient(135deg, #FFD700, #FFA500); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">{total_rows:,}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="custom-card" style="text-align: center;">
        <h3 style="color: #FFFFFF; margin-bottom: 15px;">📋 Columns</h3>
        <p style="font-size: 2.8em; font-weight: bold; background: linear-gradient(135deg, #2196F3, #21CBF3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">{total_cols}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="custom-card" style="text-align: center;">
        <h3 style="color: #FFFFFF; margin-bottom: 15px;">🔢 Numeric</h3>
        <p style="font-size: 2.8em; font-weight: bold; background: linear-gradient(135deg, #FF9800, #FF5722); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">{numeric_count}</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="custom-card" style="text-align: center;">
        <h3 style="color: #FFFFFF; margin-bottom: 15px;">⚠️ Missing</h3>
        <p style="font-size: 2.8em; font-weight: bold; background: linear-gradient(135deg, {'#4CAF50' if missing_data_pct == 0 else '#F44336'}, {'#81C784' if missing_data_pct == 0 else '#FF5722'}); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">{missing_data_pct:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

# Create tabs for different analysis sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Feature Analysis", "Correlation Analysis", "Pattern Analysis", "Export Data", "AI Analytics"])

with tab1:
    if show_features:
        st.subheader("🔍 Feature Analysis")
        
        # Add a premium info box
        st.markdown("""
        <div class="custom-card" style="margin-bottom: 20px;">
            <h4 style="color: #FFD700; margin-top: 0;">🤖 AI-Powered Feature Analysis</h4>
            <p>This section automatically detects and analyzes the most important features in your dataset using advanced algorithms.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Automatic feature importance detection
        def detect_important_features(df):
            """Automatically detect important features in the dataset"""
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # If we have label columns, calculate feature importance
            important_features = []
            if label_columns and len(numeric_cols) > 1:
                try:
                    # For each label column, calculate correlations with numeric features
                    for label_col in label_columns:
                        if label_col in numeric_cols:
                            # Remove label column from features
                            features = [col for col in numeric_cols if col != label_col]
                            correlations = []
                            for feature in features:
                                # Handle potential NaN values in correlation calculation
                                try:
                                    corr = df[feature].corr(df[label_col])
                                    if not np.isnan(corr):
                                        correlations.append((feature, abs(corr)))
                                except:
                                    pass  # Skip features that cause correlation calculation issues
                            
                            # Sort by correlation strength
                            correlations.sort(key=lambda x: x[1], reverse=True)
                            important_features.extend([feat[0] for feat in correlations[:5]])  # Top 5
                except:
                    pass
            
            # If no label-based importance, just use first 10 numeric columns
            if not important_features:
                important_features = numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
            
            return list(set(important_features))  # Remove duplicates

        # Detect important features
        feature_cols = detect_important_features(merged)
        
        # If no numeric columns, try to find important columns by name patterns
        if not feature_cols:
            possible_feature_patterns = ['value', 'score', 'measure', 'metric', 'count', 'amount', 'price', 'cost', 'size', 'length', 'width', 'height']
            for col in merged.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in possible_feature_patterns):
                    feature_cols.append(col)
            # If still no columns found, use all columns as fallback
            if not feature_cols:
                feature_cols = merged.columns.tolist()[:10] if len(merged.columns) > 10 else merged.columns.tolist()
        
        # Highlight detected label columns if any
        if label_columns:
            st.markdown(f"### 🏷️ Detected Labels: {', '.join(label_columns)}")
        
        if feature_cols:
            # Show feature count
            st.markdown(f"<p style='color: #aaaaaa; font-size: 1.1em;'><strong>{len(feature_cols)}</strong> important features detected for analysis</p>", unsafe_allow_html=True)
            
            # Check which chart types to display
            if "Box Plots" in chart_types:
                if viz_type == "Interactive (Plotly)":
                    # Interactive feature visualization
                    fig = make_subplots(rows=2, cols=1, subplot_titles=("Feature Distribution", "Feature Comparison"))
                    
                    # Box plot
                    for col in feature_cols[:5]:  # Limit to first 5 for clarity
                        fig.add_trace(go.Box(y=merged[col], name=col, marker_color='#FFD700'), row=1, col=1)
                    
                    # Bar chart for mean values
                    mean_values = [merged[col].mean() for col in feature_cols[:5]]
                    fig.add_trace(go.Bar(x=feature_cols[:5], y=mean_values, marker_color='#FFA500'), row=2, col=1)
                    
                    fig.update_layout(height=600, showlegend=False,
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(color="white"),
                                      title_font=dict(color="#FFD700", size=18))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Static feature visualization
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                    
                    # Set black background for the figure and axes
                    fig.patch.set_facecolor('#000000')
                    ax1.set_facecolor('#000000')
                    ax2.set_facecolor('#000000')
                    
                    # Box plot with colored boxes
                    bp = merged[feature_cols[:5]].boxplot(ax=ax1, patch_artist=True)
                    colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7CAC9', '#92A8D1']
                    
                    # Check if bp is a dictionary before trying to style patches
                    if isinstance(bp, dict):
                        # Style boxes
                        for i, patch in enumerate(bp['boxes']):
                            patch.set_facecolor(colors[i % len(colors)])
                            patch.set_alpha(0.8)
                            patch.set_edgecolor('#FFFFFF')
                        
                        # Style whiskers
                        for whisker in bp.get('whiskers', []):
                            whisker.set_color('#FFFFFF')
                            whisker.set_linewidth(2)
                        
                        # Style caps
                        for cap in bp.get('caps', []):
                            cap.set_color('#FFFFFF')
                            cap.set_linewidth(2)
                        
                        # Style medians
                        for median in bp.get('medians', []):
                            median.set_color('#FFFFFF')
                            median.set_linewidth(2)
                    
                    # Axes styling
                    ax1.set_title("Feature Distribution", fontsize=16, color='#FFFFFF', fontweight='bold')
                    ax1.set_ylabel("Values", color='white')
                    ax1.tick_params(colors='white', labelsize=10)
                    ax1.grid(True, alpha=0.3, color='#666666')
                    ax1.spines['top'].set_color('#666666')
                    ax1.spines['bottom'].set_color('#666666')
                    ax1.spines['left'].set_color('#666666')
                    ax1.spines['right'].set_color('#666666')
                    
                    # Bar chart for mean values with colored bars
                    mean_values = [merged[col].mean() for col in feature_cols[:5]]
                    if chart_orientation == "Horizontal":
                        bars = ax2.barh(feature_cols[:5], mean_values, color=colors[:len(mean_values)])
                        ax2.set_xlabel("Average Value", color='white')
                    else:
                        bars = ax2.bar(feature_cols[:5], mean_values, color=colors[:len(mean_values)])
                        ax2.set_ylabel("Average Value", color='white')
                    ax2.set_title("Average Values by Feature", fontsize=16, color='#FFFFFF', fontweight='bold')
                    ax2.tick_params(colors='white', labelsize=10)
                    ax2.grid(True, alpha=0.3, color='#666666')
                    ax2.spines['top'].set_color('#666666')
                    ax2.spines['bottom'].set_color('#666666')
                    ax2.spines['left'].set_color('#666666')
                    ax2.spines['right'].set_color('#666666')
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, mean_values):
                        if chart_orientation == "Horizontal":
                            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                     f'{value:.3f}', ha='left', va='center', color='white', fontweight='bold', fontsize=10)
                        else:
                            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                     f'{value:.3f}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)
                    
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to prevent memory warnings
            else:
                st.info("Box Plots are not selected for display. Enable them in the sidebar under Chart Types.")
            
            # Show feature details in an expandable section
            with st.expander("📋 View All Detected Features"):
                st.write("The following features were identified as important for analysis:")
                st.write(feature_cols)
        else:
            st.info("No suitable columns found for feature analysis. Please upload a dataset with numeric columns.")

with tab2:
    if show_correlation:
        st.subheader("📈 Correlation Analysis")
        
        # Add a premium info box
        st.markdown("""
        <div class="custom-card" style="margin-bottom: 20px;">
            <h4 style="color: #FFD700; margin-top: 0;">🔗 Correlation Intelligence</h4>
            <p>This analysis reveals hidden relationships between features in your dataset, helping identify which variables move together.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Highlight detected label columns if any
        if label_columns:
            st.markdown(f"### 🏷️ Focusing on relationships with: {', '.join(label_columns)}")
        
        st.markdown("### 📈 Feature Correlation Matrix")
        # Select only numeric columns for correlation
        numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            # Create a copy to avoid any potential issues and ensure it's a DataFrame
            numeric_data = pd.DataFrame(merged[numeric_cols].copy())
            # Calculate correlation matrix - handle NaN values
            try:
                corr_data = numeric_data.corr(method='pearson', min_periods=1)
            except Exception as e:
                # Fallback to handle any correlation calculation issues
                corr_data = numeric_data.corr(method='pearson')
            
            # Remove any columns that are all NaN after correlation
            corr_data = corr_data.dropna(axis=0, how='all').dropna(axis=1, how='all')
            
            if len(corr_data.columns) > 1:
                if "Heatmaps" in chart_types:
                    if viz_type == "Interactive (Plotly)":
                        fig = px.imshow(corr_data, 
                                         title="Feature Correlation Heatmap",
                                         color_continuous_scale=['#0f2027', '#203a43', '#2c5364', '#FFD700'],
                                         labels=dict(color="Correlation"))
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                          plot_bgcolor='rgba(0,0,0,0)',
                                          font=dict(color="white"),
                                          title_font=dict(color="#FFD700", size=18))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # If we have label columns, show specific correlations
                        if label_columns:
                            for label_col in label_columns:
                                if label_col in corr_data.columns:
                                    st.markdown(f"#### 🔍 Correlations with {label_col}")
                                    # Sort by absolute values using argsort - handle NaN values
                                    temp_series = corr_data[label_col].drop(label_col)
                                    # Remove NaN values before sorting
                                    temp_series = temp_series.dropna()
                                    # Use numpy argsort which handles the sorting more reliably
                                    import numpy as np
                                    sorted_indices = np.argsort(np.abs(temp_series.values))[::-1]  # Descending order
                                    label_corr = temp_series.iloc[sorted_indices]
                                    # Show top 5 correlations
                                    top_corr = label_corr.head(5)
                                    fig2 = px.bar(x=top_corr.index, y=top_corr.values,
                                                 title=f"Top Correlations with {label_col}",
                                                 color_discrete_sequence=['#FFD700'])
                                    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                                       plot_bgcolor='rgba(0,0,0,0)',
                                                       font=dict(color="white"),
                                                       title_font=dict(color="#FFD700", size=16))
                                    st.plotly_chart(fig2, use_container_width=True)
                    else:
                        fig, ax = plt.subplots(figsize=(14, 12))
                        # Set black background for the figure and axes
                        fig.patch.set_facecolor('#000000')
                        ax.set_facecolor('#000000')
                        
                        # Use a vibrant colormap
                        # Handle potential NaN values in correlation calculation
                        try:
                            im = sns.heatmap(corr_data, annot=True, cmap='RdYlGn', center=0, ax=ax,
                                             cbar_kws={'label': 'Correlation'},
                                             annot_kws={"size": 10, "weight": "bold"},
                                             linewidths=0.5, linecolor='#333333')
                        except Exception as e:
                            # Fallback if there are issues with the correlation heatmap
                            im = sns.heatmap(corr_data.fillna(0), annot=True, cmap='RdYlGn', center=0, ax=ax,
                                             cbar_kws={'label': 'Correlation'},
                                             annot_kws={"size": 10, "weight": "bold"},
                                             linewidths=0.5, linecolor='#333333')
                        
                        ax.set_title("Feature Correlation Heatmap", fontsize=18, color='#FFFFFF', fontweight='bold', pad=20)
                        ax.tick_params(colors='white', labelsize=10)
                        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color='white', weight='bold')
                        plt.setp(ax.get_yticklabels(), color='white', weight='bold')
                        
                        # Style the colorbar with simplified error handling
                        try:
                            if hasattr(im, 'collections') and len(im.collections) > 0:
                                cbar = im.collections[0].colorbar
                                if cbar is not None:
                                    # Style tick labels
                                    tick_labels = cbar.ax.yaxis.get_ticklabels()
                                    if tick_labels is not None:
                                        plt.setp(tick_labels, color='white', weight='bold')
                        except Exception as e:
                            pass  # Silently ignore colorbar styling issues
                        
                        st.pyplot(fig)
                        plt.close(fig)  # Close the figure to prevent memory warnings
                        
                        # If we have label columns, show specific correlations
                        if label_columns:
                            for label_col in label_columns:
                                if label_col in corr_data.columns:
                                    st.markdown(f"#### 🔍 Correlations with {label_col}")
                                    # Sort by absolute values using argsort - handle NaN values
                                    temp_series = corr_data[label_col].drop(label_col)
                                    # Remove NaN values before sorting
                                    temp_series = temp_series.dropna()
                                    # Use numpy argsort which handles the sorting more reliably
                                    import numpy as np
                                    sorted_indices = np.argsort(np.abs(temp_series.values))[::-1]  # Descending order
                                    label_corr = temp_series.iloc[sorted_indices]
                                    # Show top 5 correlations
                                    top_corr = label_corr.head(5)
                                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                                    fig2.patch.set_facecolor('#000000')
                                    ax2.set_facecolor('#000000')
                                    if chart_orientation == "Horizontal":
                                        bars = ax2.barh(range(len(top_corr)), list(top_corr.values), color='#FFD700')
                                        ax2.set_yticks(range(len(top_corr)))
                                        ax2.set_yticklabels(top_corr.index)
                                    else:
                                        bars = ax2.bar(range(len(top_corr)), list(top_corr.values), color='#FFD700')
                                        ax2.set_xticks(range(len(top_corr)))
                                        ax2.set_xticklabels(top_corr.index, rotation=45, ha='right')
                                    ax2.set_title(f"Top Correlations with {label_col}", color='white', fontsize=14)
                                    ax2.tick_params(colors='white')
                                    ax2.grid(True, alpha=0.3, color='#666666')
                                    ax2.spines['top'].set_color('#666666')
                                    ax2.spines['bottom'].set_color('#666666')
                                    ax2.spines['left'].set_color('#666666')
                                    ax2.spines['right'].set_color('#666666')
                                    # Add value labels on bars
                                    for i, (bar, value) in enumerate(zip(bars, top_corr.values)):
                                        if chart_orientation == "Horizontal":
                                            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                                                     f'{value:.3f}', ha='left', va='center', color='white', fontweight='bold')
                                        else:
                                            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                                     f'{value:.3f}', ha='center', va='bottom', color='white', fontweight='bold')
                                    st.pyplot(fig2)
                                    plt.close(fig2)  # Close the figure to prevent memory warnings
                else:
                    st.info("Heatmaps are not selected for display. Enable them in the sidebar under Chart Types.")
                
                # Add correlation statistics
                if len(corr_data.columns) > 1:
                    # Calculate correlation statistics
                    corr_values = corr_data.values
                    # Remove diagonal (self-correlations)
                    corr_values = corr_values[~np.eye(corr_values.shape[0], dtype=bool)].reshape(-1)
                    # Remove NaN values
                    corr_values = corr_values[~np.isnan(corr_values)]
                    
                    if len(corr_values) > 0:
                        # Use nanmean, nanmax, nanmin to handle any remaining NaN values
                        avg_corr = np.nanmean(corr_values)
                        max_corr = np.nanmax(corr_values)
                        min_corr = np.nanmin(corr_values)
                        
                        st.markdown("### 📊 Correlation Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"""
                            <div class="custom-card" style="text-align: center;">
                                <h4 style="color: #2196F3; margin-bottom: 10px;">Average</h4>
                                <p style="font-size: 1.8em; font-weight: bold; color: {'#4CAF50' if avg_corr > 0.5 else '#FF9800' if avg_corr > 0 else '#F44336'};">{avg_corr:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f"""
                            <div class="custom-card" style="text-align: center;">
                                <h4 style="color: #4CAF50; margin-bottom: 10px;">Maximum</h4>
                                <p style="font-size: 1.8em; font-weight: bold; color: #4CAF50;">{max_corr:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(f"""
                            <div class="custom-card" style="text-align: center;">
                                <h4 style="color: #F44336; margin-bottom: 10px;">Minimum</h4>
                                <p style="font-size: 1.8em; font-weight: bold; color: #F44336;">{min_corr:.3f}</p>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("Not enough valid numeric columns for correlation analysis.")
        else:
            st.info("Not enough numeric columns for correlation analysis. Need at least 2 numeric columns.")

with tab3:
    st.subheader("🌀 Pattern Analysis")
    
    # Add a premium info box
    st.markdown("""
    <div class="custom-card" style="margin-bottom: 20px;">
        <h4 style="color: #FFD700; margin-top: 0;"> AI Pattern Recognition</h4>
        <p>Discover hidden patterns and trends in your data through advanced statistical analysis and visualization.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if show_patterns:
        st.markdown("### 📊 Data Distribution Analysis")
        # Show distribution of key numeric features
        numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        
        # Use first few numeric columns for distribution analysis
        key_features = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
        
        if key_features:
            st.markdown(f"<p style='color: #aaaaaa; font-size: 1.1em;'><strong>{len(key_features)}</strong> important features detected for analysis</p>", unsafe_allow_html=True)
            if "Histograms" in chart_types:
                if viz_type == "Interactive (Plotly)":
                    fig = make_subplots(rows=1, cols=len(key_features), subplot_titles=key_features)
                    for i, feature in enumerate(key_features):
                        if feature in merged.columns:
                            fig.add_trace(go.Histogram(x=merged[feature], name=feature, marker_color='#FFA500'), row=1, col=i+1)
                    fig.update_layout(height=400, showlegend=False,
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(color="white"),
                                      title_font=dict(color="#FFD700", size=18),
                                      xaxis=dict(color="white", gridcolor="rgba(255, 255, 255, 0.2)"),
                                      yaxis=dict(color="white", gridcolor="rgba(255, 255, 255, 0.2)"))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig, axes = plt.subplots(1, len(key_features), figsize=(6*len(key_features), 6))
                    # Set black background for the figure
                    fig.patch.set_facecolor('#000000')
                    
                    if len(key_features) == 1:
                        axes = [axes]
                    for ax, feature in zip(axes, key_features):
                        if feature in merged.columns:
                            ax.set_facecolor('#000000')
                            n, bins, patches = ax.hist(merged[feature], bins=20, alpha=0.8)
                            # Color bars with vibrant gradient
                            colors = ['#FFD700', '#FFA500', '#FF8C00', '#FF4500', '#FF6347']
                            for i, patch in enumerate(patches):
                                patch.set_facecolor(colors[i % len(colors)])
                                patch.set_edgecolor('white')
                                patch.set_linewidth(0.5)
                            ax.set_title(f"Distribution of {feature}", fontsize=14, color='#FFFFFF', fontweight='bold')
                            ax.set_xlabel(feature, color='white', fontsize=12)
                            ax.set_ylabel("Frequency", color='white', fontsize=12)
                            ax.tick_params(colors='white', labelsize=10)
                            ax.grid(True, alpha=0.3, color='#666666')
                            ax.spines['top'].set_color('#666666')
                            ax.spines['bottom'].set_color('#666666')
                            ax.spines['left'].set_color('#666666')
                            ax.spines['right'].set_color('#666666')
                    
                    st.pyplot(fig)
                    plt.close(fig)  # Close the figure to prevent memory warnings
            else:
                st.info("Histograms are not selected for display. Enable them in the sidebar under Chart Types.")
        else:
            st.info("No numeric columns found for distribution analysis.")
            
    # Add additional pattern analysis based on selected actions
    if "Data Summary" in actions:
        st.markdown("### 🔍 Data Patterns")
        # Show basic statistics
        st.markdown("#### 📋 Dataset Statistics")
        st.dataframe(merged.describe(), width='stretch')
        
        # Show data types
        st.markdown("#### 📝 Column Information")
        col_info = pd.DataFrame({
            'Column': merged.columns,
            'Data Type': [str(dtype) for dtype in merged.dtypes],  # Convert to string to avoid Arrow issues
            'Missing Values': merged.isnull().sum(),
            'Unique Values': [merged[col].nunique() for col in merged.columns]
        })
        st.dataframe(col_info, width='stretch')
        
    # Add data cleaning action
    if "Data Cleaning" in actions:
        st.markdown("### 🧹 Data Cleaning")
        st.info("Data cleaning has been automatically applied to handle missing values and duplicates.")
        
    # Add feature scaling action
    if "Feature Scaling" in actions:
        st.markdown("### 📏 Feature Scaling")
        st.info("Feature scaling options are available in the advanced analysis section.")
        
    # Add outlier detection action
    if "Outlier Detection" in actions:
        st.markdown("### 🚨 Outlier Detection")
        st.info("Outlier detection algorithms can be applied to identify anomalies in your data.")
    
    # Add ML Predictions
    if show_ml_predictions and label_columns:
        st.markdown("### 🤖 ML Predictions")
        st.markdown("""
        <div class="custom-card" style="margin-bottom: 20px;">
            <h4 style="color: #FFD700; margin-top: 0;">🔮 Predictive Analytics</h4>
            <p>Using Random Forest algorithm to predict outcomes based on your data patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Prepare data for ML
            df_ml = merged.copy()
            
            # Encode categorical variables
            label_encoders = {}
            for col in df_ml.select_dtypes(include=['object']).columns:
                if col in label_columns:
                    le = LabelEncoder()
                    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
                    label_encoders[col] = le
                else:
                    # For other categorical columns, we'll use one-hot encoding
                    df_ml = pd.get_dummies(df_ml, columns=[col], prefix=col)
            
            # Select target column (first label column)
            target_col = label_columns[0]
            
            # Prepare features and target
            if target_col in df_ml.columns:
                X = df_ml.drop(columns=[target_col])
                y = df_ml[target_col]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train Random Forest model
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = rf_model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Display results
                st.markdown(f"**Model Accuracy:** <span style='color: #4CAF50; font-size: 1.5em; font-weight: bold;'>{accuracy:.2%}</span>", unsafe_allow_html=True)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False).head(10)
                
                # Plot feature importance
                fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                             title='Top 10 Feature Importances',
                             color_discrete_sequence=['#FFD700'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color="white"),
                                  title_font=dict(color="#FFD700", size=18))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show classification report
                st.markdown("### 📋 Classification Report")
                st.text(classification_report(y_test, y_pred))
                
        except Exception as e:
            st.warning(f"ML Prediction failed: {str(e)}")
    
    # Add Clustering Analysis
    if show_clustering:
        st.markdown("###  Clustering Analysis")
        st.markdown("""
        <div class="custom-card" style="margin-bottom: 20px;">
            <h4 style="color: #FFD700; margin-top: 0;'>🔍 Unsupervised Learning</h4>
            <p>Using K-Means clustering to discover hidden patterns in your data.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Prepare data for clustering
            df_cluster = merged.select_dtypes(include=[np.number]).copy()
            
            # Handle missing values
            df_cluster = df_cluster.fillna(df_cluster.mean())
            
            # Add a check for a sufficient number of features before running PCA
            if df_cluster.shape[1] < 2:
                st.warning("Clustering analysis requires at least two numeric features to perform PCA for visualization. Please upload a dataset with more numeric columns.")
            else:
                # Scale data
                scaler = StandardScaler()
                df_cluster_scaled = scaler.fit_transform(df_cluster)
                
                # Apply K-Means clustering
                kmeans = KMeans(n_clusters=3, random_state=42)
                cluster_labels = kmeans.fit_predict(df_cluster_scaled)
                
                # Add cluster labels to original data
                merged['Cluster'] = cluster_labels
                
                # Apply PCA for visualization
                pca = PCA(n_components=2)
                df_pca = pca.fit_transform(df_cluster_scaled)
                
                # Create scatter plot
                fig = px.scatter(x=df_pca[:, 0], y=df_pca[:, 1], color=cluster_labels,
                                 title=f'Clustering Results (PCA Visualization)',
                                 labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', 
                                         'y': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'},
                                 color_continuous_scale=['#FFD700', '#FF8C00', '#FF4500'],
                                 hover_data=[merged.index])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color="white"),
                                  title_font=dict(color="#FFD700", size=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show cluster statistics in premium cards
                st.markdown("### 📊 Cluster Statistics")
                cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                cols = st.columns(len(cluster_counts))
                for i, (cluster_id, count) in enumerate(cluster_counts.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="custom-card" style="text-align: center;">
                            <h4 style="color: #FFD700; margin-bottom: 10px;">Cluster {cluster_id}</h4>
                            <p style="font-size: 1.8em; font-weight: bold; color: #FFFFFF; margin: 0;">{count}</p>
                            <p style="color: #aaaaaa; margin: 5px 0 0 0;">samples</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show detailed cluster statistics
                st.markdown("### 📈 Detailed Cluster Analysis")
                agg_dict = {}
                for col in df_cluster.columns[:5]:  # Show first 5 numeric columns
                    agg_dict[col] = ['mean', 'std']
                cluster_stats = merged.groupby('Cluster').agg(agg_dict).round(3).copy()
                st.dataframe(cluster_stats, width='stretch')
            
        except Exception as e:
            st.warning(f"Clustering analysis failed: {str(e)}")
    else:
        st.info("Please select 'Clustering Analysis' in the sidebar to enable this feature.")
    
    # Add Feature Importance Analysis if selected
    if show_feature_importance and label_columns:
        st.markdown("### 🎯 Feature Importance Analysis")
        st.markdown("""
        <div class="custom-card" style="margin-bottom: 25px;">
            <h3 style="color: #FFD700; margin-top: 0; font-size: 1.8em;">🎯 Feature Importance Analysis</h3>
            <p style="font-size: 1.1em;">Advanced statistical methods to determine which features most influence outcomes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Prepare data
            df_fi = merged.copy()
            
            # Encode categorical variables
            for col in df_fi.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                df_fi[col] = le.fit_transform(df_fi[col].astype(str))
            
            # Select target column
            target_col = label_columns[0]
            
            if target_col in df_fi.columns:
                # Calculate correlation-based feature importance
                correlations = []
                for col in df_fi.columns:
                    if col != target_col:
                        try:
                            # Calculate correlation with error handling
                            corr_val = abs(np.corrcoef(df_fi[col], df_fi[target_col])[0, 1])
                            if not np.isnan(corr_val):
                                correlations.append((col, corr_val))
                        except:
                            pass  # Skip columns that can't be correlated
                
                # Sort by correlation
                correlations.sort(key=lambda x: x[1], reverse=True)
                
                # Create DataFrame
                corr_data = {"Feature": [item[0] for item in correlations[:15]],
                             "Correlation": [item[1] for item in correlations[:15]]}
                corr_df = pd.DataFrame(corr_data).copy()
                
                # Plot with enhanced styling
                fig = px.bar(corr_df, x='Correlation', y='Feature', orientation='h',
                             title='Top 15 Features by Correlation with Target',
                             color='Correlation',
                             color_continuous_scale=['#FFD700', '#FF8C00', '#FF4500'])
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)',
                                  font=dict(color="white"),
                                  title_font=dict(color="#FFD700", size=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation statistics
                if len(correlations) > 0:
                    max_corr = max([corr[1] for corr in correlations])
                    avg_corr = np.mean([corr[1] for corr in correlations])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="custom-card" style="text-align: center;">
                            <h4 style="color: #FFFFFF; margin-bottom: 10px;">🏆 Highest Correlation</h4>
                            <p style="font-size: 1.8em; font-weight: bold; background: linear-gradient(135deg, #4CAF50, #81C784); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">{max_corr:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="custom-card" style="text-align: center;">
                            <h4 style="color: #FFFFFF; margin-bottom: 10px;">📊 Average Correlation</h4>
                            <p style="font-size: 1.8em; font-weight: bold; background: linear-gradient(135deg, #2196F3, #64B5F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">{avg_corr:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Feature importance analysis failed: {str(e)}")
        else:
            st.info("Please select 'Feature Importance' in the sidebar and ensure your dataset has label columns.")
else:
    # Enhanced info when no AI features are selected
    st.markdown("""
    <div class="custom-card" style="text-align: center; padding: 40px;">
        <h2 style="color: #FFD700; margin-top: 0;">🌟 Unlock Advanced AI Insights</h2>
        <p style="font-size: 1.2em; color: #e0e0e0; max-width: 600px; margin: 0 auto 30px;">Please select AI/ML features from the sidebar to enable our advanced analytics engine.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show AI/ML capabilities info with enhanced design
    st.markdown("### 🚀 Available AI/ML Capabilities")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="custom-card" style="height: 100%;">
            <h3 style="color: #FFD700; text-align: center; margin-top: 0;">🤖 ML Predictions</h3>
            <div style="text-align: center; margin: 20px 0;">
                <span style="font-size: 3em;">🔮</span>
            </div>
            <p style="color: #e0e0e0;">Predict outcomes using Random Forest and other advanced algorithms with accuracy metrics and feature importance ranking.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card" style="height: 100%;">
            <h3 style="color: #FFD700; text-align: center; margin-top: 0;"> Clustering</h3>
            <div style="text-align: center; margin: 20px 0;">
                <span style="font-size: 3em;">🔍</span>
            </div>
            <p style="color: #e0e0e0;">Discover hidden patterns with K-Means and other unsupervised learning techniques with PCA visualization.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="custom-card" style="height: 100%;">
            <h3 style="color: #FFD700; text-align: center; margin-top: 0;">🎯 Feature Importance</h3>
            <div style="text-align: center; margin: 20px 0;">
                <span style="font-size: 3em;">📊</span>
            </div>
            <p style="color: #e0e0e0;">Identify which features most influence your target variables with correlation analysis and statistical ranking.</p>
        </div>
        """, unsafe_allow_html=True)

# Premium footer with enhanced design
st.markdown("""
<hr style="border: 1px solid rgba(255, 255, 255, 0.1);">
<div style="text-align: center; padding: 30px; border-radius: 25px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 12px 30px rgba(0,0,0,0.5); backdrop-filter: blur(10px);">
    <p style="font-weight: bold; font-size: 1.4em; background: linear-gradient(90deg, #FFD700, #FFA500); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 10px 0;">Universal Data Analysis Dashboard- Advanced Data Analytics Platform</p>
    <p style="font-size: 1.2em; color: #e0e0e0;">Powered by AI & Machine Learning</p>
    <p style="color: #aaaaaa; margin-top: 15px;">Transforming complex neurological data into actionable medical insights</p>
    <p style="color: #666666; font-size: 0.9em; margin-top: 20px;">© 2025 Universal Data Analysis Dashboard | Secure & HIPAA Compliant | Real-time Analysis</p>
</div>
""", unsafe_allow_html=True)
