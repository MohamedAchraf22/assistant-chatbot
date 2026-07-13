from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from rag import build_rag_chain
from typing import Optional
from scheduler import start_scheduler, stop_scheduler

_rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _rag_pipeline
    print("⏳ Warming up RAG pipeline...")
    _rag_pipeline = build_rag_chain()
    print("✅ RAG pipeline ready.")
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return {"message": "Hello FastAPI"}


class ChatRequest(BaseModel):
    question: str
    history: Optional[list[dict]] = None


class ChatResponse(BaseModel):
    answer: str


def normalize_answer(answer) -> str:
    if isinstance(answer, str):
        return answer
    if isinstance(answer, list):
        parts = []
        for item in answer:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(answer) if answer is not None else ""


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        answer = _rag_pipeline(request.question, request.history or [])
        return ChatResponse(answer=normalize_answer(answer))
    except Exception as e:
        return ChatResponse(answer=f"Error in RAG: {str(e)}")