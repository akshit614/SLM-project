import streamlit as st
import faiss
import os
import torch
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA
import requests


# Step 1: Load and Process Manual
def load_manual(file_path):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

# Step 2: Convert Text into Embeddings
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# Step 3: Initialize Retrieval-Augmented Generation (RAG) Model
def load_rag_model(vector_db):
    retriever = vector_db.as_retriever()
    llm = GPT4All(model='gpt4all-falcon-newbpe-q4_0.gguf', backend='gptj', verbose=True)
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain


def download_model(model_url, model_path):
    if not os.path.exists(model_path):
        print(f"Downloading model from {model_url}...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded to {model_path}")
    else:
        print(f"Model already exists at {model_path}")

# Streamlit UI
st.title("Customer Support Chatbot")

uploaded_file = st.file_uploader("Upload Operations & Maintenance Manual (PDF)", type=["pdf"])
if uploaded_file:
    with open("manual.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("Manual uploaded successfully!")

    # Process document
    chunks = load_manual("manual.pdf")
    vector_db = create_vector_db(chunks)

    try:
        model_url = "https://gpt4all.io/models/gguf/gpt4all-falcon-newbpe-q4_0.gguf"
        model_path = "gpt4all-falcon-newbpe-q4_0.gguf"
        download_model(model_url, model_path)
        llm = GPT4All(model=model_path, backend='gptj', verbose=True)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())
        st.write("Chatbot is ready! Ask your question below.")
        query = st.text_input("Enter your query:")
        if query:
            response = qa_chain.run(query)
            st.write("### Response:")
            st.write(response)

    except Exception as e:
        st.error(f"An error occurred: {e}")