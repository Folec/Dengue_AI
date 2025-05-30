import streamlit as st
from shapanalyzer import ShapAnalyzer
from prediction_model import linear_regression
from gemini import GeminiInterface
import pandas as pd
import google.generativeai as genai
from dashboard import Dashboard 

# Streamlit page configuration
st.set_page_config(
    page_title="Dengue Prediction Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and cache the data to avoid recomputation on every interaction"""
    X_final, feature_names, model, df = linear_regression.create_X_final()
    return X_final, feature_names, model, df

@st.cache_resource
def initialize_components():
    """Initialize and cache the components"""
    # Initialize GeminiInterface
    API_KEY = "AIzaSyBl0at8sQs2lgBngITENuswZG-xUSkimoc" # Replace with your actual API key
    genai.configure(api_key=API_KEY)
    gemini = GeminiInterface()
    return gemini

def main():
    # Load data
    X_final, feature_names, model, df = load_data()
    gemini = initialize_components()
    
    # Display basic info in sidebar
    st.sidebar.write(f"Data shape: {X_final.shape}")
    st.sidebar.write(f"Features: {len(feature_names)}")
    
    # Initialize SHAP analyzer
    with st.spinner("Computing SHAP values..."):
        analyzer = ShapAnalyzer(model, X_final, feature_names=feature_names)
        analyzer.compute_shap_values()
    
    # Generate SHAP summary
    shap_summary = analyzer.generate_text_summary()
    
    # Define RAG query
    rag_query = "Explain the impact of weather features on dengue cases."
    
    # Initialize and run the dashboard
    dashboard = Dashboard(dataframe=df, analyzer=analyzer, gemini=gemini)
    dashboard.run(shap_summary, rag_query)

if __name__ == "__main__":
    main()