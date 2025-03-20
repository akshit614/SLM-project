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
    qa_chain = load_rag_model(vector_db)

    st.write("Chatbot is ready! Ask your question below.")
    query = st.text_input("Enter your query:")
    if query:
        response = qa_chain.run(query)
        st.write("### Response:")
        st.write(response)


# # Step 1: Load and Process Manual
# def load_manual(file_path):
#     loader = PyMuPDFLoader(file_path)
#     documents = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = text_splitter.split_documents(documents)
#     return chunks

# # Step 2: Convert Text into Embeddings
# def create_vector_db(chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_db = FAISS.from_documents(chunks, embeddings)
#     return vector_db

# # Step 3: Initialize Retrieval-Augmented Generation (RAG) Model
# def load_rag_model(vector_db):
#     retriever = vector_db.as_retriever()
#     llm = GPT4All(model='ggml-gpt4all-j-v1.3-groovy.bin', backend='gptj', verbose=True)
#     qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
#     return qa_chain

# # Streamlit UI
# st.title("Customer Support Chatbot")

# uploaded_file = st.file_uploader("Upload Operations & Maintenance Manual (PDF)", type=["pdf"],key=1)
# if uploaded_file:
#     with open("manual.pdf", "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     st.success("Manual uploaded successfully!")

#     # Process document
#     chunks = load_manual("manual.pdf")
#     vector_db = create_vector_db(chunks)
#     qa_chain = load_rag_model(vector_db)

#     st.write("Chatbot is ready! Ask your question below.")
#     query = st.text_input("Enter your query:")
#     if query:
#         response = qa_chain.run(query)
#         st.write("### Response:")
#         st.write(response)
# def create_vector_db(chunks):
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_db = FAISS.from_documents(chunks, embeddings)
#     return vector_db

# # Step 3: Initialize Retrieval-Augmented Generation (RAG) Model
# def load_rag_model(vector_db):
#     retriever = vector_db.as_retriever()
#     llm = GPT4All(model='ggml-gpt4all-j-v1.3-groovy.bin', backend='gptj', verbose=True)
#     qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
#     return qa_chain

# # Streamlit UI
# st.title("Customer Support Chatbot")

# uploaded_file = st.file_uploader("Upload Operations & Maintenance Manual (PDF)", type=["pdf"])
# if uploaded_file:
#     with open("manual.pdf", "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     st.success("Manual uploaded successfully!")

#     # Process document
#     chunks = load_manual("manual.pdf")
#     vector_db = create_vector_db(chunks)
#     qa_chain = load_rag_model(vector_db)

#     st.write("Chatbot is ready! Ask your question below.")
#     query = st.text_input("Enter your query:")
#     if query:
#         response = qa_chain.run(query)
#         st.write("### Response:")
#         st.write(response)
