from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import torch

# Define model path for the fine-tuned model
model_path = "model/fine_tuned" 

# Load the tokenizer and model 
tokenizer = AutoTokenizer.from_pretrained(model_path)  
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
model.eval() 

# Load the embeddings model for FAISS vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local("vector_store/faiss_index", embeddings)

# Function to answer a question based on similarity search and fine-tuned model
def answer_question(question: str) -> str:
    # Retrieve the most relevant document (context) from FAISS vector store
    docs = db.similarity_search(question, k=1)  # k=1 retrieves the most relevant document
    context = docs[0].page_content  # Assuming the context is in 'page_content' field

    # Tokenize the question and context together for the model
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move the model and inputs to GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform inference (no gradients needed)
    with torch.no_grad():
        outputs = model(**inputs)
        start = torch.argmax(outputs.start_logits)
        end = torch.argmax(outputs.end_logits) + 1

    # Decode and return the answer from the tokenized output
    answer = tokenizer.decode(inputs["input_ids"][0][start:end], skip_special_tokens=True)
    return answer

# taking an example questions:
question = "What are the symptoms of diabetes?"
answer = answer_question(question)
print(f"Answer: {answer}")
