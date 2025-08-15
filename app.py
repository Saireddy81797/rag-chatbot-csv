import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app title
st.title("📄 RAG Chatbot on CSV using LangChain + OpenAI")
st.write("Ask questions from your CSV data!")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load CSV into LangChain
    loader = CSVLoader(file_path="temp.csv")
    docs = loader.load()

    # Create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create QA chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    # User input
    query = st.text_input("Enter your question:")

    if st.button("Ask"):
        if query:
            result = qa_chain(query)
            st.write("**Answer:**", result["result"])
        else:
            st.warning("Please enter a question.")
