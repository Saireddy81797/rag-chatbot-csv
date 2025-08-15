import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Streamlit UI
st.title("📄 RAG Chatbot on CSV using LangChain + OpenAI")
st.write("Upload a CSV and ask questions about its data.")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load CSV as LangChain documents
    loader = CSVLoader(file_path="temp.csv")
    docs = loader.load()

    # Create FAISS vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Create RetrievalQA chain
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
            st.subheader("Answer")
            st.write(result["result"])
        else:
            st.warning("Please enter a question.")
