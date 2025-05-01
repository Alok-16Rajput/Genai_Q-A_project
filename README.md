# 🧠 Custom AI-Powered Q&A System

This project is a Generative AI-powered Question-Answering system that uses **fine-tuned BERT**, **LangChain + FAISS** for Retrieval-Augmented Generation (RAG), and a REST API served via **FastAPI**, with an optional chatbot UI using **Streamlit**.

---

## 🔧 Features

- Fine-tuned BERT model for QA using Hugging Face Transformers
- Semantic search using FAISS and Sentence-Transformers
- Retrieval-Augmented Generation (RAG) pipeline
- FastAPI backend for inference
- Streamlit UI chatbot 

Components:
- FAISS: Retrieves relevant chunks based on the user query
- BERT: Answers question based on retrieved context
