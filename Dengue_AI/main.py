import streamlit as st
from shapanalyzer import ShapAnalyzer
from prediction_model import DengueLSTM
from gemini import GeminiInterface
import pandas as pd
import google.generativeai as genai
from dashboard import Dashboard 
import torch
import torch.nn as nn
import prediction_model

# Streamlit page configuration
st.set_page_config(
    page_title="Dengue Prediction Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and cache the data
@st.cache_data(ttl=3600, max_entries=1) 
def load_data():
    """Load and cache the data to avoid recomputation on every interaction"""
    model = DengueLSTM()
    dfs = model.get_city_dataframes()  
    df_sj = dfs['San Juan']
    df_iq = dfs['Iquitos']
    return df_sj, df_iq

@st.cache_resource(ttl=3600, max_entries=1)
def initialize_components():
    """Initialize and cache the components"""
    # Initialize GeminiInterface
    API_KEY = "AIzaSyBl0at8sQs2lgBngITENuswZG-xUSkimoc" 
    genai.configure(api_key=API_KEY)
    gemini = GeminiInterface()
    return gemini

def main():
    # Load data both dataframes for San Juan and Iquitos
    df_sj, df_iq = load_data()

    # Start geimin if need
    gemini = initialize_components()
    
    # Define RAG query
    rag_query = "As an epidemiologist, you'll need to use all the information provided by: LSTM model predictions, SHAP method feedback and RAG system information to provide context and analysis of the situation and progress of the dengue epidemic, based on sources and statistics."
    
    # Initialize and run the dashboard
    dataframes = {'San Juan': df_sj, 'Iquitos': df_iq}

    dashboard = Dashboard(dataframe=dataframes, analyzer=None, gemini=gemini)
    dashboard.run(shap_summary=None, rag_query=None)

if __name__ == "__main__":
    main()