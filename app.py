from fastapi import FastAPI
from pydantic import BaseModel
from model.rag_inference import answer_question

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = answer_question(request.question)
    return {"answer": answer}
