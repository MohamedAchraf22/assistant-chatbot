from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from rag import ask_question
from typing import Optional
from scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def home():
    return {"message": "Hello FastAPI"}


class ChatRequest(BaseModel):
    question: str
    # Recent session turns passed from the UI layer.
    # Each entry is {"role": "user"|"assistant", "content": "..."}
    history: Optional[list[dict]] = None


class ChatResponse(BaseModel):
    answer: str


def normalize_answer(answer) -> str:
    """
    LLM .content can be a plain string OR a list of content-part dicts
    (e.g. [{"type": "text", "text": "..."}]) depending on the model/response.
    This safely converts either shape into a single string.
    """
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
        answer = ask_question(request.question, history=request.history or [])
        return ChatResponse(answer=normalize_answer(answer))
    except Exception as e:
        return ChatResponse(answer=f"Error in RAG: {str(e)}")