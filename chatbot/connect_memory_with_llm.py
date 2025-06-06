import os
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())



# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN=os.environ.get("HF_TOKEN")
print("HuggingFace Token: ", HF_TOKEN)
HUGGINGFACE_REPO_ID="mistralai/Mistral-7B-Instruct-v0.3"

# def load_llm(huggingface_repo_id):
#     llm=HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         model_kwargs={"token":HF_TOKEN,
#                       "max_length":"512"}
#     )
#     return llm

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        # max_new_tokens=512,
        huggingfacehub_api_token=HF_TOKEN  # Correct location for token
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
        You are a Dengue‐Data Assistant. 
        Use only the pieces of information provided in the Context to answer the user’s question—no outside knowledge. 
        If you don’t see the answer in the Context, just say “I don’t know.”

        Format your answer exactly as follows (use Markdown):

        1. **Overview**  
        2–3 sentences summarizing dengue in the requested country/region (mention endemic status, data sources, typical case ranges).

        2. **Year‐by‐Year Reported Cases**  
        Create a Markdown table with four columns:
        | Year | Reported Cases (Countrywide) | Region / Notes         | Source ID |
        |:----:|:----------------------------:|:----------------------:|:---------:|
        – If a snippet mentions only regional data (e.g., “rural southern Sri Lanka”), put that number under “Region / Notes” and leave “Reported Cases” as “N/A.”
        – If multiple snippets give the same year+number, merge them into a single row.
        – Sort rows by year (oldest → newest).

        3. **Key Sources**  
        Provide a bullet list of full citations (Author(s), Year, Title, Journal/Publisher). 
        After each citation, include the matching `[Source ID]` that corresponds to the table’s “Source ID” column.

        Context Snippets (with Source IDs):  
        {context}

        Question: {question}

        Start your answer immediately (no extra preamble, no small talk).  
        """

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain
qa_chain=RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query=input("Write Query Here: ")
response=qa_chain.invoke({'query': user_query})
print("RESULT: ", response["result"])
print("SOURCE DOCUMENTS: ", response["source_documents"])
