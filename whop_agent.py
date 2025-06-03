from fastapi import FastAPI
from pydantic import BaseModel
from my_vectordb import vectordb, retriever  # Assumes your retriever is set up as in rag_server.py
from openai import OpenAI

client = OpenAI()
app = FastAPI()

class Q(BaseModel):
    question: str
    user_id: str | None = None  # Optional user ID for tracking

@app.post("/whop_ask")
def whop_ask(q: Q):
    docs = retriever.get_relevant_documents(q.question)
    context = "\n\n".join(d.page_content for d in docs[:4])
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Whop users."},
            {"role": "user", "content": f"{q.question}\n\nContext:\n{context}"}
        ]
    )
    return {"answer": resp.choices[0].message.content}
