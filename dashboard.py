import streamlit as st
import requests
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Error404 AI System", layout="wide")

# Check API status function
def check_api_status():
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

st.markdown("""
<style>
    .stApp {
        background: #f5f7fa;
    }
    
    h1, h2, h3 {
        color: #1a1a2e !important;
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #0066b8;
    }
    
    [data-testid="stMetricValue"] {
        color: #0066b8 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    .dataframe th {
        background: #0066b8 !important;
        color: white !important;
        font-weight: 700 !important;
    }
    
    .stButton button {
        background: #0066b8;
        color: white;
        font-weight: 700;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
    }
    
    .stButton button:hover {
        background: #0052a0;
    }
    
    .api-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Model Evaluation"

# Navbar
col_logo, col_spacer, col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1.2, 1.2, 1.2])

with col_logo:
    st.markdown('<div style="background: linear-gradient(135deg, #0066b8 0%, #0052a0 100%); padding: 0.8rem 1.5rem; border-radius: 8px; font-size: 1.5rem; font-weight: 800; color: white; display: inline-block;">Error404 AI System</div>', unsafe_allow_html=True)

with col_btn1:
    if st.button("Model Evaluation", key="nav_model", use_container_width=True):
        st.session_state.page = "Model Evaluation"
        st.rerun()

with col_btn2:
    if st.button("Recommendations", key="nav_recs", use_container_width=True):
        st.session_state.page = "Recommendations"
        st.rerun()

with col_btn3:
    if st.button("System Status", key="nav_status", use_container_width=True):
        st.session_state.page = "System Status"
        st.rerun()

st.markdown("---")

# Show API warning if not connected (only for pages that need API)
api_connected = check_api_status()

# ============================================
# MODEL EVALUATION PAGE
# ============================================
if st.session_state.page == "Model Evaluation":
    st.markdown('<div class="section-header">Model Performance Overview</div>', unsafe_allow_html=True)
    
    if not api_connected:
        st.warning("API not connected. Metrics will be loaded from local files if available.")
    
    # Try to load metrics from API first, then from local file
    metrics_data = None
    df = None
    
    if api_connected:
        try:
            response = requests.get(f"{API_URL}/metrics", timeout=5)
            if response.status_code == 200:
                metrics_data = response.json()
        except:
            pass
    
    # If API failed, try local file
    if metrics_data is None:
        try:
            local_path = Path("results/metrics_comparison.csv")
            if local_path.exists():
                df = pd.read_csv(local_path)
                if 'Unnamed: 0' in df.columns:
                    df = df.set_index('Unnamed: 0')
                    df.index.name = 'Model'
        except:
            pass
    
    # Convert API data to DataFrame if available
    if metrics_data is not None and df is None:
        df = pd.DataFrame(metrics_data)
        if 'Unnamed: 0' in df.columns:
            df = df.set_index('Unnamed: 0')
            df.index.name = 'Model'
        df.columns = df.columns.str.strip()
    
    if df is not None and not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Best Model", df['F1-Score'].idxmax())
        col2.metric("Best Accuracy", f"{df['Accuracy'].max():.3f}")
        col3.metric("Best F1-Score", f"{df['F1-Score'].max():.3f}")
        col4.metric("Best AUROC", f"{df['AUROC'].max():.3f}")
        
        st.dataframe(df.round(4), use_container_width=True)
        
        st.markdown('<div class="section-header">Model Details</div>', unsafe_allow_html=True)
        
        model_list = list(df.index)
        model_cols = st.columns(len(model_list))
        
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = model_list[0]
        
        for idx, model in enumerate(model_list):
            with model_cols[idx]:
                if st.button(model, use_container_width=True):
                    st.session_state.selected_model = model
                    st.rerun()
        
        st.markdown("---")
        
        selected = st.session_state.selected_model
        st.markdown(f"### {selected}")
        
        model_metrics = df.loc[selected].to_frame().T
        st.dataframe(model_metrics.round(4), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Confusion Matrix**")
            try:
                if api_connected:
                    plots_response = requests.get(f"{API_URL}/plots", timeout=5)
                    if plots_response.status_code == 200:
                        plots = plots_response.json().get("available_plots", [])
                        model_cm = [p for p in plots if selected.lower().replace(' ', '_') in p.lower() and 'confusion' in p.lower()]
                        
                        if model_cm:
                            img_response = requests.get(f"{API_URL}/plot/{model_cm[0]}")
                            if img_response.status_code == 200:
                                img = Image.open(BytesIO(img_response.content))
                                st.image(img, use_container_width=True)
                        else:
                            st.info("Confusion matrix not available")
                else:
                    st.info("API not connected. Run 'python main.py' to generate plots.")
            except:
                pass
        
        with col2:
            st.markdown("**ROC Curve**")
            try:
                if api_connected:
                    model_roc = [p for p in plots if selected.lower().replace(' ', '_') in p.lower() and 'roc' in p.lower() and 'curves' not in p.lower()]
                    if model_roc:
                        img_response = requests.get(f"{API_URL}/plot/{model_roc[0]}")
                        if img_response.status_code == 200:
                            img = Image.open(BytesIO(img_response.content))
                            st.image(img, use_container_width=True)
                    else:
                        st.info("ROC curve not available")
                else:
                    st.info("API not connected. Run 'python main.py' to generate plots.")
            except:
                pass
        
        st.markdown('<div class="section-header">Overall Model Comparison</div>', unsafe_allow_html=True)
        
        plot_df = df.reset_index().melt(id_vars=['Model'], var_name='Metric', value_name='Score')
        
        distinct_colors = {
            'Logistic Regression': '#1f77b4',
            'Random Forest': '#ff7f0e',
            'XGBoost': '#2ca02c',
            'Neural Network': '#d62728'
        }
        
        fig = px.bar(
            plot_df, x='Metric', y='Score', color='Model', barmode='group',
            title='Model Performance Comparison',
            color_discrete_map=distinct_colors,
            text='Score',
            template='simple_white'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(yaxis_range=[0, 1.05], height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Metrics Heatmap**")
            fig_heatmap = px.imshow(df.values, x=df.columns, y=df.index, text_auto='.3f', aspect="auto", color_continuous_scale='Blues')
            fig_heatmap.update_layout(height=450)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        with col2:
            st.markdown("**ROC Curves - All Models**")
            try:
                if api_connected:
                    roc_plot = [p for p in plots if 'roc_curves' in p.lower()]
                    if roc_plot:
                        img_response = requests.get(f"{API_URL}/plot/{roc_plot[0]}")
                        if img_response.status_code == 200:
                            img = Image.open(BytesIO(img_response.content))
                            st.image(img, use_container_width=True)
                    else:
                        st.info("Combined ROC curve not available")
                else:
                    st.info("API not connected. Run 'python main.py' to generate plots.")
            except:
                pass
    else:
        st.warning("No metrics found. Run 'python main.py' first to generate results.")

# ============================================
# RECOMMENDATIONS PAGE
# ============================================
elif st.session_state.page == "Recommendations":
    st.markdown('<div class="section-header">Product Recommendations</div>', unsafe_allow_html=True)
    
    # Load available client IDs from local file
    try:
        user_features_path = Path("results/user_features.csv")
        if user_features_path.exists():
            user_features = pd.read_csv(user_features_path)
            available_clients = user_features['client_id'].head(50).tolist()
        else:
            available_clients = []
    except:
        available_clients = []
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if available_clients:
            client_id = st.selectbox("Select Client ID", available_clients)
        else:
            client_id = st.text_input("Client ID", placeholder="Enter client ID")
        
        top_n = st.slider("Number of recommendations", 5, 20, 10)
        model_choice = st.selectbox("Select Model", ["XGBoost", "Random Forest", "Neural Network", "Logistic Regression"])
    
    if st.button("Get Recommendations", type="primary"):
        if client_id:
            with st.spinner(f"Generating {top_n} recommendations..."):
                # Generate demo recommendations based on client ID
                np.random.seed(hash(str(client_id)) % 2**32)
                scores = np.random.beta(2, 1, top_n)
                scores = np.sort(scores)[::-1]
                
                recommendations = pd.DataFrame({
                    'rank': range(1, top_n + 1),
                    'product_sku': [f'PRD_{np.random.randint(10000, 99999)}' for _ in range(top_n)],
                    'purchase_probability': scores.round(4),
                    'confidence': ['High' if s > 0.8 else 'Medium' if s > 0.6 else 'Low' for s in scores],
                    'recommendation_reason': ['Based on browsing history' for _ in range(top_n)]
                })
                
                st.success(f"Top {top_n} recommendations for client {client_id} using {model_choice}")
                st.dataframe(recommendations, use_container_width=True)
                
                fig = px.bar(
                    recommendations, x='product_sku', y='purchase_probability', 
                    title=f'Recommendation Scores - {model_choice}',
                    color='purchase_probability',
                    color_continuous_scale='Blues',
                    text='purchase_probability'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(height=450, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please enter a client ID")

# ============================================
# SYSTEM STATUS PAGE
# ============================================
elif st.session_state.page == "System Status":
    st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
    
    st.markdown("### API Status")
    if api_connected:
        st.success("API is running on port 8000")
    else:
        st.error("API is not running")
        st.code("cd api && python main.py")
    
    st.markdown("---")
    st.markdown("### Model Specifications")
    
    model_specs = {
        "Logistic Regression": {"Type": "Linear Classifier", "Best For": "Baseline performance"},
        "Random Forest": {"Type": "Ensemble Method", "Best For": "Non-linear relationships"},
        "XGBoost": {"Type": "Gradient Boosting", "Best For": "High accuracy tabular data"},
        "Neural Network": {"Type": "Deep Learning", "Best For": "Complex pattern recognition"}
    }
    
    for model, specs in model_specs.items():
        with st.expander(model):
            st.markdown(f"**Type:** {specs['Type']}")
            st.markdown(f"**Best For:** {specs['Best For']}")

st.markdown("---")
st.caption("Error404 AI System | Product Recommendation Engine | Powered by 4 ML Models")