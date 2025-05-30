import google.generativeai as genai
import pandas as pd


class GeminiInterface:
    def __init__(self, api_key = "AIzaSyBl0at8sQs2lgBngITENuswZG", model_name="gemini-2.0-flash"):
        """
        Initialize the GeminiInterface class.

        :param api_key: API key for accessing Google's Gemini LLM.
        :param model_name: The Gemini model to use (default: "gemini-2.0-flash").
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(self.model_name)

    def create_prompt(self, shap_summary, dataframe, rag_context):
        """
        Create a prompt by combining SHAP summary, DataFrame data, and RAG context.

        :param shap_summary: Text summary of SHAP analysis.
        :param dataframe: Relevant data from a DataFrame (e.g., sample rows or statistics).
        :param rag_context: Context fetched from a Retrieval-Augmented Generation (RAG) system.
        :return: A combined prompt string.
        """
        # Extract sample rows or statistics from the DataFrame
        sample_data = dataframe.to_string(index=False)

        # Combine all components into a single prompt
        prompt = (
            "Model Interpretability Analysis:\n"
            f"{shap_summary}\n\n"
            "Relevant Data:\n"
            f"{sample_data}\n\n"
            "Additional Context:\n"
            f"{rag_context}\n\n"
            "Based on the above information, provide a detailed explanation and insights."
        )
        return prompt

    def query_gemini(self, prompt):
        """
        Send the prompt to the Gemini LLM and get a response.

        :param prompt: The input prompt string.
        :return: The generated response from Gemini LLM.
        """
        try:
            chat = self.model.start_chat()
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Error querying Gemini LLM: {str(e)}")


    def fetch_rag_context(self, query):
        """
        Query a Retrieval-Augmented Generation (RAG) system to fetch additional context.

        :param query: The query string for the RAG system.
        :return: Retrieved context as a string.
        """
        # Simulate RAG system query (replace with actual implementation)
        rag_context = f"Retrieved context for query: {query}"
        return rag_context

    def generate_insights(self, shap_summary, dataframe, rag_query):
        """
        Generate insights by combining SHAP summary, DataFrame data, and RAG context.

        :param shap_summary: Text summary of SHAP analysis.
        :param dataframe: Relevant data from a DataFrame.
        :param rag_query: Query string for the RAG system.
        :return: Generated insights from Gemini LLM.
        """
        rag_context = self.fetch_rag_context(rag_query)
        prompt = self.create_prompt(shap_summary, dataframe, rag_context)
        return self.query_gemini(prompt)