from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os

# Medical data with optional metadata
data = [
    Document(page_content="Diabetes is a chronic condition...", metadata={"topic": "Diabetes"}),
    Document(page_content="Hypertension, or high blood pressure...", metadata={"topic": "Hypertension"}),
    Document(page_content="Cancer treatment includes...", metadata={"topic": "Cancer"}),
    Document(page_content="Asthma is a chronic respiratory condition...", metadata={"topic": "Asthma"}),
    Document(page_content="Heart disease refers to...", metadata={"topic": "Heart Disease"}),
    Document(page_content="Alzheimer's disease is a progressive...", metadata={"topic": "Alzheimer's"}),
    Document(page_content="Flu is a contagious...", metadata={"topic": "Flu"}),
    Document(page_content="COVID-19 is a respiratory illness...", metadata={"topic": "COVID-19"}),
    Document(page_content="Pneumonia is an infection...", metadata={"topic": "Pneumonia"}),
    Document(page_content="Arthritis is a condition...", metadata={"topic": "Arthritis"})
]

# Use updated HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Create FAISS index
db = FAISS.from_documents(data, embeddings)

# Save vector store
os.makedirs("vector_store", exist_ok=True)
db.save_local("vector_store/faiss_index")

print("FAISS index has been saved locally to 'vector_store/faiss_index'.")
