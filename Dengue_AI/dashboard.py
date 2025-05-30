import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from shapanalyzer import ShapAnalyzer
from gemini import GeminiInterface
import io


class Dashboard:
    def __init__(self, dataframe, analyzer, gemini):
        """
        Initialize the Dashboard class.

        :param dataframe: The DataFrame containing prediction results.
        :param analyzer: An instance of the ShapAnalyzer class.
        :param gemini: An instance of the GeminiInterface class.
        """
        self.dataframe = dataframe
        self.analyzer = analyzer
        self.gemini = gemini

    def display_dataframe(self):
        """
        Display the DataFrame containing prediction results.
        """
        st.subheader("🔢 Prediction Results")
        
        # Add some basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(self.dataframe))
        with col2:
            if "prediction" in self.dataframe.columns:
                st.metric("Avg Prediction", f"{self.dataframe['prediction'].mean():.2f}")
        with col3:
            st.metric("Features", len(self.dataframe.columns))
        
        # Display the dataframe with search and filtering
        st.dataframe(self.dataframe, use_container_width=True, height=400)
        
        # Option to download the data
        csv = self.dataframe.to_csv(index=False)
        st.download_button(
            label="📥 Download data as CSV",
            data=csv,
            file_name='prediction_results.csv',
            mime='text/csv',
        )

    def display_prediction_graph(self):
        """
        Display a graph of the predictions using both matplotlib and plotly.
        """
        st.subheader("📈 Prediction Visualization")
        
        # Check if prediction column exists
        if "prediction" not in self.dataframe.columns:
            st.warning("No 'prediction' column found in the dataframe. Please check your data.")
            return
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Interactive Plot", "Static Plot"])
        
        with tab1:
            # Interactive plotly chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.dataframe.index,
                y=self.dataframe["prediction"],
                mode='lines+markers',
                name='Predictions',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title="Prediction Over Time/Index",
                xaxis_title="Index",
                yaxis_title="Prediction Value",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Static matplotlib chart
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.dataframe.index, self.dataframe["prediction"], 
                   label="Predictions", color="blue", linewidth=2)
            ax.set_xlabel("Index")
            ax.set_ylabel("Prediction")
            ax.set_title("Prediction Graph")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()

    def display_shap_graphs(self):
        """
        Display SHAP graphs (mean SHAP plot and swarm plot).
        """
        st.subheader("🔍 SHAP Analysis")
        
        # Create tabs for different SHAP visualizations
        tab1, tab2 = st.tabs(["Feature Importance", "Feature Effects"])
        
        with tab1:
            st.write("**Mean SHAP Values (Feature Importance)**")
            try:
                # Create a new figure for mean SHAP plot
                plt.figure(figsize=(10, 8))
                self.analyzer.plot_mean_shap()
                
                # Capture the current figure
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close()
                
            except Exception as e:
                st.error(f"Error generating mean SHAP plot: {str(e)}")
        
        with tab2:
            st.write("**SHAP Swarm Plot (Feature Effects)**")
            try:
                # Create a new figure for swarm plot
                plt.figure(figsize=(10, 8))
                self.analyzer.plot_swarm()
                
                # Capture the current figure
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close()
                
            except Exception as e:
                st.error(f"Error generating swarm plot: {str(e)}")
        
        # Display SHAP summary as text
        with st.expander("📝 SHAP Analysis Summary"):
            try:
                summary = self.analyzer.generate_text_summary()
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating SHAP summary: {str(e)}")

    def display_prompt_result(self, shap_summary, rag_query):
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
                insights = self.gemini.generate_insights(shap_summary, self.dataframe, rag_query)
                
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
                        custom_insights = self.gemini.generate_insights(shap_summary, self.dataframe, custom_query)
                        st.write("**Custom Analysis:**")
                        st.success(custom_insights)
                        
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")

    def display_sidebar_info(self):
        """
        Display additional information in the sidebar.
        """
        st.sidebar.header("📊 Dashboard Info")
        
        # Dataset info
        st.sidebar.subheader("Dataset Overview")
        st.sidebar.write(f"• **Rows:** {len(self.dataframe)}")
        st.sidebar.write(f"• **Columns:** {len(self.dataframe.columns)}")
        
        # Column names
        st.sidebar.subheader("Available Columns")
        for col in self.dataframe.columns:
            st.sidebar.write(f"• {col}")
            
        # Basic statistics
        if "prediction" in self.dataframe.columns:
            st.sidebar.subheader("Prediction Statistics")
            pred_stats = self.dataframe["prediction"].describe()
            st.sidebar.write(f"• **Mean:** {pred_stats['mean']:.2f}")
            st.sidebar.write(f"• **Std:** {pred_stats['std']:.2f}")
            st.sidebar.write(f"• **Min:** {pred_stats['min']:.2f}")
            st.sidebar.write(f"• **Max:** {pred_stats['max']:.2f}")

    def run(self, shap_summary, rag_query):
        """
        Run the Streamlit dashboard.
        """
        # Main title with emoji
        st.title("🦟 Dengue Prediction Dashboard")
        st.markdown("---")
        
        # Display sidebar information
        self.display_sidebar_info()
        
        # Main content
        try:
            # Create tabs for better organization
            tab1, tab2, tab3, tab4 = st.tabs([
                "📊 Data Overview", 
                "📈 Predictions", 
                "🔍 SHAP Analysis", 
                "🤖 AI Insights"
            ])
            
            with tab1:
                self.display_dataframe()
            
            with tab2:
                self.display_prediction_graph()
            
            with tab3:
                self.display_shap_graphs()
            
            with tab4:
                self.display_prompt_result(shap_summary, rag_query)
                
        except Exception as e:
            st.error(f"An error occurred while running the dashboard: {str(e)}")
            st.write("Please check your data and model configuration.")