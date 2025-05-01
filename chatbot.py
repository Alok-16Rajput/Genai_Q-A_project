import streamlit as st
import requests

st.title("🧠 GenAI Q&A Chatbot")
question = st.text_input("Ask something...")

if st.button("Get Answer"):
    res = requests.post("http://127.0.0.1:8000/ask", json={"question": question})
    st.write("Answer:", res.json()["answer"])
