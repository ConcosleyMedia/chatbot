from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import json
import time
from my_vectordb import retriever
from openai import OpenAI

CHAT_HISTORY_DIR = os.path.join(os.path.dirname(__file__), '../chat_histories')
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

client = OpenAI()
app = FastAPI()

class ChatRequest(BaseModel):
    question: str
    user_id: str
    ticket_id: Optional[str] = None  # If None, create a new ticket

@app.post("/user_chat")
def user_chat(req: ChatRequest):
    # Determine ticket_id
    ticket_id = req.ticket_id or f"{req.user_id}_{int(time.time())}"
    history_path = os.path.join(CHAT_HISTORY_DIR, f"{ticket_id}.json")
    # Load or create chat history
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []
    # Append new question
    history.append({"role": "user", "content": req.question})
    # Retrieve context from RAG
    docs = retriever.get_relevant_documents(req.question)
    context = "\n\n".join(d.page_content for d in docs[:4])
    # Compose messages for OpenAI
    messages = history[-10:]  # Last 10 exchanges
    # Insert system prompt and context
    messages = (
        [{"role": "system", "content": "You are a helpful assistant. Use the provided context to answer as accurately as possible."},
         {"role": "system", "content": f"Context:\n{context}"}]
        + messages
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=messages
    )
    answer = resp.choices[0].message.content
    # Append answer to history
    history.append({"role": "assistant", "content": answer})
    # Save updated history
    with open(history_path, "w") as f:
        json.dump(history, f)
    return {"answer": answer, "ticket_id": ticket_id, "history": history}

@app.get("/user_tickets/{user_id}")
def list_tickets(user_id: str):
    # List all ticket files for this user
    tickets = []
    for fname in os.listdir(CHAT_HISTORY_DIR):
        if fname.startswith(user_id + "_") and fname.endswith(".json"):
            ticket_id = fname[:-5]
            tickets.append(ticket_id)
    return {"tickets": tickets}
