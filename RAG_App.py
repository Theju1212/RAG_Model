# ======================================================
# 📄 Dynamic RAG with Groq, FAISS, and LLaMA3 (Streamlit)
# ======================================================

import streamlit as st
from dotenv import load_dotenv
import os
import time
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------
# Load environment variables
# -----------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", "")

# -----------------------
# Streamlit UI Config
# -----------------------
st.set_page_config(page_title="Dynamic RAG with Groq", layout="wide")
if os.path.exists("my.png"):
    st.image("my.png", width=150)
st.title("📄 Dynamic RAG with Groq, FAISS & LLaMA3")

# -----------------------
# Session State
# -----------------------
if "vector" not in st.session_state:
    st.session_state.vector = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------
# Sidebar: Document Upload
# -----------------------
with st.sidebar:
    st.header("📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                docs = []
                for file in uploaded_files:
                    temp_path = file.name
                    with open(temp_path, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(temp_path)
                    docs.extend(loader.load())

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1800,
                    chunk_overlap=200
                )
                final_documents = text_splitter.split_documents(docs)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.vector = FAISS.from_documents(final_documents, embeddings)
                st.success("✅ Documents processed successfully!")
        else:
            st.warning("⚠ Please upload at least one PDF document.")

# -----------------------
# Main Chat Interface
# -----------------------
st.header("💬 Chat with Your Documents")

if not groq_api_key:
    st.error("🚨 GROQ API key not found. Please set it in your .env file.")
else:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="LLaMA3-8b-8192")

    # Display previous chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt_input = st.chat_input("Ask a question about your documents...")

    if prompt_input:
        if st.session_state.vector is None:
            st.warning("⚠ Please upload and process documents first.")
        else:
            with st.chat_message("user"):
                st.markdown(prompt_input)
            st.session_state.chat_history.append({"role": "user", "content": prompt_input})

            with st.spinner("Thinking..."):
                prompt = ChatPromptTemplate.from_template(
                    "Answer the questions based on the provided context only.\n\n"
                    "<context>\n{context}\n</context>\n\n"
                    "Question: {input}\n"
                    "Please provide the most accurate response."
                )
                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = st.session_state.vector.as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                start_time = time.process_time()
                response = retrieval_chain.invoke({"input": prompt_input})
                end_time = time.process_time()

            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})

            st.caption(f"⏱ Response time: {end_time - start_time:.2f} seconds")
