import os
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from langchain.vectorstores import FAISS

load_dotenv()

INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index.pkl')

with open(INDEX_PATH, 'rb') as f:
    vectordb = pickle.load(f)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

client = OpenAI()
app = FastAPI()

class Q(BaseModel):
    question: str
    user_id: str | None = None

@app.post("/ask")
def ask(q: Q):
    docs = retriever.get_relevant_documents(q.question)
    context = "\n\n".join(d.page_content for d in docs[:4])
    sources = [d.metadata.get("source", "") for d in docs[:4]]
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are an n8n expert. Cite docs at the end."},
            {"role": "user", "content": f"{q.question}\n\nContext:\n{context}"}
        ]
    )
    answer = resp.choices[0].message.content
    sources_section = "\n\nSources:\n" + "\n".join(sources)
    return {"answer": answer + sources_section, "sources": sources}
