from langchain_google_genai import ChatGoogleGenerativeAI
from config import API_KEY, MODEL_NAME

def get_llm():
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=API_KEY,
        temperature=0,
        max_tokens=256
    )