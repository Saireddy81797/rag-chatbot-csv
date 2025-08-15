import os
import pandas as pd
from dotenv import load_dotenv
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load API Key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Load CSV data
loader = CSVLoader(file_path="data.csv")
docs = loader.load()

# Step 2: Create embeddings & store in FAISS
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 3: Create Retrieval QA chain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

print("RAG Chatbot is ready! Type 'exit' to quit.\n")
while True:
    query = input("Ask a question: ")
    if query.lower() == "exit":
        break
    result = qa_chain(query)
    print("Answer:", result['result'])
