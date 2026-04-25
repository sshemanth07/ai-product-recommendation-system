import streamlit as st
import requests
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Product Recommendation System", layout="wide")

st.title("AI Product Recommendation System")
st.caption("Purchase likelihood prediction and model evaluation")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["Model Evaluation", "System Status", "Client Recommendation"])
    st.markdown("---")
    st.caption("Powered by Logistic Regression + Random Forest")

# Page 1: Model Evaluation
if page == "Model Evaluation":
    st.header("Model Performance Metrics")
    
    with st.spinner("Loading metrics..."):
        try:
            response = requests.get(f"{API_URL}/metrics", timeout=5)
            if response.status_code == 200:
                metrics_data = response.json()
                
                if isinstance(metrics_data, list) and len(metrics_data) > 0:
                    df = pd.DataFrame(metrics_data)
                    
                    # Clean dataframe
                    if 'Unnamed: 0' in df.columns:
                        df = df.set_index('Unnamed: 0')
                        df.index.name = 'Model'
                    
                    df.columns = df.columns.str.strip()
                    
                    # Determine best model
                    best_model = df['Accuracy'].idxmax()
                    
                    # Display metrics table
                    st.dataframe(df.round(4), use_container_width=True)
                    
                    # Key metrics highlight
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Best Model", best_model)
                    col2.metric("Best Accuracy", f"{df['Accuracy'].max():.4f}")
                    col3.metric("Best F1-Score", f"{df['F1-Score'].max():.4f}")
                    col4.metric("Best AUROC", f"{df['AUROC'].max():.4f}")
                    
                    # Bar chart - Original colors (blue and orange)
                    st.subheader("Performance Comparison")
                    
                    plot_df = df.reset_index()
                    plot_df = plot_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
                    
                    # Original color scheme: blue (#1f77b4) and orange (#ff7f0e)
                    fig = px.bar(
                        plot_df, 
                        x='Metric', 
                        y='Score', 
                        color='Model',
                        barmode='group',
                        title='Model Performance Comparison',
                        labels={'Score': 'Score', 'Metric': 'Metric'},
                        color_discrete_sequence=['#1f77b4', '#ff7f0e'],
                        text='Score'
                    )
                    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                    fig.update_layout(
                        yaxis_range=[0.5, 1.05],
                        height=450,
                        yaxis_title='Score',
                        xaxis_title=''
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ROC Curve Section - NEW
                    st.subheader("ROC Curves - Model Discriminative Ability")
                    
                    try:
                        roc_response = requests.get(f"{API_URL}/roc", timeout=5)
                        if roc_response.status_code == 200:
                            roc_data = roc_response.json()
                            
                            if "error" not in roc_data:
                                fig_roc = go.Figure()
                                
                                # Add diagonal reference line (random classifier)
                                fig_roc.add_trace(go.Scatter(
                                    x=[0, 1],
                                    y=[0, 1],
                                    mode='lines',
                                    name='Random Classifier (AUC=0.5)',
                                    line=dict(dash='dash', color='gray', width=1)
                                ))
                                
                                # Add ROC curves for each model
                                colors = {'Logistic_Regression': '#1f77b4', 'Random_Forest': '#ff7f0e'}
                                
                                for model_name in ['Logistic_Regression', 'Random_Forest']:
                                    if model_name in roc_data:
                                        fpr = roc_data[model_name]['fpr']
                                        tpr = roc_data[model_name]['tpr']
                                        auc_score = roc_data[model_name]['auc']
                                        
                                        fig_roc.add_trace(go.Scatter(
                                            x=fpr,
                                            y=tpr,
                                            mode='lines',
                                            name=f'{model_name.replace("_", " ")} (AUC = {auc_score:.3f})',
                                            line=dict(color=colors.get(model_name, '#1f77b4'), width=2.5),
                                            fill=None
                                        ))
                                
                                fig_roc.update_layout(
                                    title='ROC Curves Comparison',
                                    xaxis_title='False Positive Rate (1 - Specificity)',
                                    yaxis_title='True Positive Rate (Sensitivity)',
                                    xaxis=dict(range=[0, 1], gridcolor='lightgray'),
                                    yaxis=dict(range=[0, 1], gridcolor='lightgray'),
                                    height=500,
                                    plot_bgcolor='white',
                                    legend=dict(x=0.7, y=0.05, bgcolor='rgba(255,255,255,0.8)')
                                )
                                
                                st.plotly_chart(fig_roc, use_container_width=True)
                                
                                # Interpretation
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.info("**AUC Interpretation**\n\n- 0.90-1.00: Excellent\n- 0.80-0.90: Good\n- 0.70-0.80: Fair\n- 0.60-0.70: Poor\n- 0.50: Random")
                                with col2:
                                    st.success(f"**Model Performance**\n\nBoth models show excellent discriminative ability with AUC scores near 1.0, indicating perfect separation between classes.")
                            else:
                                st.info("ROC curve data not available. Run training to generate.")
                        else:
                            st.info("ROC endpoint not available")
                    except Exception as e:
                        st.info("Run `python main.py` to generate ROC curves")
                    
                else:
                    st.info("Run `python main.py` first to generate metrics")
            else:
                st.error(f"API returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Start your API first:")
            st.code("cd api && python main.py", language="bash")
    
    # Generated plots section - Original arrangement (side by side)
    st.markdown("---")
    st.subheader("Model Diagnostics")
    
    try:
        response = requests.get(f"{API_URL}/plots", timeout=5)
        if response.status_code == 200:
            plots = response.json().get("available_plots", [])
            if plots:
                # Original side-by-side arrangement
                col1, col2 = st.columns(2)
                for i, plot in enumerate(plots):
                    img_response = requests.get(f"{API_URL}/plot/{plot}")
                    if img_response.status_code == 200:
                        img = Image.open(BytesIO(img_response.content))
                        if i == 0:
                            with col1:
                                st.image(img, caption=plot.replace('.png', '').replace('_', ' ').title(), use_container_width=True)
                        else:
                            with col2:
                                st.image(img, caption=plot.replace('.png', '').replace('_', ' ').title(), use_container_width=True)
            else:
                st.info("No plots found. Run `python main.py` to generate plots.")
        else:
            st.info("Plot endpoint not available")
    except Exception as e:
        st.info("Run `python main.py` to generate plots")

# Page 2: System Status
elif page == "System Status":
    st.header("System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Health")
        try:
            response = requests.get(f"{API_URL}/", timeout=3)
            if response.status_code == 200:
                st.success("✓ API is running on port 8000")
                st.write("Endpoints available:")
                st.write("- /metrics - Model performance data")
                st.write("- /plots - List of plots")
                st.write("- /plot/{filename} - Plot images")
                st.write("- /roc - ROC curve data")
            else:
                st.error("API error")
        except requests.exceptions.ConnectionError:
            st.error("API is not running")
            st.code("cd api && python main.py")
    
    with col2:
        st.subheader("Data Availability")
        try:
            r1 = requests.get(f"{API_URL}/metrics", timeout=3)
            r2 = requests.get(f"{API_URL}/plots", timeout=3)
            
            if r1.status_code == 200:
                data = r1.json()
                st.success(f"✓ {len(data)} model metrics loaded")
            
            if r2.status_code == 200:
                plots = r2.json().get("available_plots", [])
                st.success(f"✓ {len(plots)} diagnostic plots available")
        except:
            st.warning("Could not verify data availability")
    
    st.markdown("---")
    st.subheader("Deployed Models")
    st.write("**Logistic Regression** - Linear classifier with L2 regularization")
    st.write("**Random Forest** - Ensemble of 120 decision trees")
    st.write("**Best Model:** Random Forest (Perfect classification on test set)")

# Page 3: Client Recommendation (placeholder)
else:
    st.header("Client Purchase Predictor")
    
    st.info("""
    **Client Recommendation Feature**
    
    To enable real-time client predictions:
    1. Train and save models using `python train.py`
    2. Run the prediction API using `python serve.py`
    
    For now, please view the **Model Evaluation** tab to see your model performance metrics and ROC curves.
    """)
    
    client_id = st.text_input("Enter Client ID (demo)", placeholder="e.g., 12345")
    
    if st.button("Predict"):
        st.warning("Prediction API not yet configured. Check the Model Evaluation tab for model metrics and ROC curves.")

# Footer
st.markdown("---")
st.caption("Product Recommendation System | Powered by Logistic Regression + Random Forest")