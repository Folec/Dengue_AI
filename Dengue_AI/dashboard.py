import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from shapanalyzer import ShapAnalyzer
from gemini import GeminiInterface
import io
import os
from model_loader import DengueLSTM


class Dashboard:
    def __init__(self, dataframe, analyzer, gemini):
        """
        Initialize the Dashboard class.

        :param dataframe: Dict with city names as keys and DataFrames as values.
        :param analyzer: An instance of the ShapAnalyzer class.
        :param gemini: An instance of the GeminiInterface class.
        """
        self.dataframe = dataframe
        self.analyzer = analyzer
        self.gemini = gemini

    def display_dataframe(self, df):
        """
        Display the DataFrame containing prediction results.
        """
        st.subheader("🔢 Prediction Results")
        
        # Add some basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            if "predicted_cases" in df.columns:
                st.metric("Avg Prediction", f"{df['predicted_cases'].mean():.2f}")
        with col3:
            st.metric("Features", len(df.columns))
        
        # Display the dataframe with search and filtering
        st.dataframe(df, use_container_width=True, height=400)
        
        # Option to download the data
        csv = df.to_csv(index=False)
        # Use a unique key for each download_button based on the DataFrame's id
        st.download_button(
            label="📥 Download data as CSV",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv',
            key=f"download_csv_{id(df)}"
        )

    def display_prediction_graph(self, df):
        st.subheader("📈 Prediction Visualization (by Year & Week)")
    
        # Remove leading/trailing spaces from column names
        df = df.rename(columns=lambda x: x.strip())
    
        required_cols = {"year", "weekofyear", "predicted_cases"}
        missing = required_cols - set(df.columns)
        if missing:
            st.warning(f"Missing columns for plotting: {missing}")
            st.dataframe(df)
            return
    
        # Group by year and week, sum predicted cases
        df_grouped = df.groupby(["year", "weekofyear"], as_index=False)["predicted_cases"].sum()
        df_grouped["year_week"] = df_grouped["year"].astype(str) + "-W" + df_grouped["weekofyear"].astype(str)
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_grouped["year_week"],
            y=df_grouped["predicted_cases"],
            mode='lines+markers',
            name='Predicted Cases',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        fig.update_layout(
            title="Predicted Dengue Cases per Week",
            xaxis_title="Year-Week",
            yaxis_title="Predicted Cases",
            hovermode='x unified',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)


    def display_shap_graphs(self, df):
        """
        Display SHAP graphs (mean SHAP plot and swarm plot).
        """
        st.subheader("🔍 SHAP Analysis")

        # Always use the correct city code for model loading
        city = 'sj'
        if 'city' in df.columns:
            city_val = str(df['city'].iloc[0]).strip().lower()
            if city_val in ['san juan', 'sj']:
                city = 'sj'
            elif city_val in ['iquitos', 'iq']:
                city = 'iq'

        try:
            model_obj = DengueLSTM(city=city)
            models_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "Models")
            model_path = os.path.join(models_dir, f'dengue_lstm_{city}.pth')
            scaler_path = os.path.join(models_dir, f'dengue_scalers_{city}.pkl')
            model_obj.load_model(model_path, scaler_path)
            feature_cols = model_obj.sj_top_features
            features_df = df[[col for col in feature_cols if col in df.columns]]
            analyzer = ShapAnalyzer(model_obj.model, features_df, feature_names=features_df.columns.tolist())
        except Exception as e:
            st.warning(f"SHAP model could not be loaded: {e}")
            return

        # Create tabs for different SHAP visualizations
        tab1, tab2 = st.tabs(["Feature Importance", "Feature Effects"])
        
        with tab1:
            st.write("**Mean SHAP Values (Feature Importance)**")
            try:
                plt.figure(figsize=(10, 8))
                analyzer.plot_mean_shap()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close()
            except Exception as e:
                st.error(f"Error generating mean SHAP plot: {str(e)}")
        
        with tab2:
            st.write("**SHAP Swarm Plot (Feature Effects)**")
            try:
                plt.figure(figsize=(10, 8))
                analyzer.plot_swarm()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                st.image(buf, use_container_width=True)
                plt.close()
            except Exception as e:
                st.error(f"Error generating swarm plot: {str(e)}")
        
        # Display SHAP summary as text
        with st.expander("📝 SHAP Analysis Summary"):
            try:
                summary = analyzer.generate_text_summary()
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating SHAP summary: {str(e)}")

    def display_prompt_result(self, df, shap_summary, rag_query):
        """
        Display the result of the prompt generated by the Gemini LLM.
        """
        st.subheader("🤖 AI-Generated Insights")
        
        # Display the query
        st.write("**Query:**")
        st.info(rag_query)
        
        # Generate and display insights
        with st.spinner("Generating insights with AI..."):
            try:
                insights = self.gemini.generate_insights(shap_summary, df, rag_query)
                
                st.write("**AI Analysis:**")
                st.write(insights)
                
                # Option to ask custom questions
                st.write("---")
                custom_query = st.text_area(
                    "Ask a custom question about the data:",
                    placeholder="e.g., What are the top 3 factors affecting dengue predictions?"
                )
                
                if st.button("Generate Custom Insights") and custom_query:
                    with st.spinner("Processing your question..."):
                        custom_insights = self.gemini.generate_insights(shap_summary, df, custom_query)
                        st.write("**Custom Analysis:**")
                        st.success(custom_insights)
                        
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")

    def display_sidebar_info(self, df):
        """
        Display additional information in the sidebar.
        """
        st.sidebar.header("📊 Dashboard Info")
        
        # Dataset info
        st.sidebar.subheader("Dataset Overview")
        st.sidebar.write(f"• **Rows:** {len(df)}")
        st.sidebar.write(f"• **Columns:** {len(df.columns)}")
        
        # Column names
        st.sidebar.subheader("Available Columns")
        for col in df.columns:
            st.sidebar.write(f"• {col}")
            
        # Basic statistics
        if "predicted_cases" in df.columns:
            st.sidebar.subheader("Prediction Statistics")
            pred_stats = df["predicted_cases"].describe()
            st.sidebar.write(f"• **Mean:** {pred_stats['mean']:.2f}")
            st.sidebar.write(f"• **Std:** {pred_stats['std']:.2f}")
            st.sidebar.write(f"• **Min:** {pred_stats['min']:.2f}")
            st.sidebar.write(f"• **Max:** {pred_stats['max']:.2f}")

    def run(self, shap_summary, rag_query):
        """
        Run the Streamlit dashboard.
        """
        st.cache_data.clear()
        st.cache_resource.clear()
        st.title("🦟 Dengue Prediction Dashboard")
        st.markdown("---")

        # Tabs for different cities
        city_tabs = st.tabs(list(self.dataframe.keys()))
        for i, city in enumerate(self.dataframe.keys()):
            with city_tabs[i]:
                st.header(f"🌆 {city} Data Overview")
                st.write("This section provides an overview of the dengue prediction data for the selected city.")
        
                city_df = self.dataframe[city]          

                self.display_sidebar_info(city_df)
                self.display_dataframe(city_df)
                self.display_prediction_graph(city_df)
                self.display_shap_graphs(city_df)
                self.display_prompt_result(city_df, shap_summary, rag_query)