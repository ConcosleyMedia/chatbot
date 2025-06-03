# n8n Knowledge Base RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot for n8n documentation, exposing a `/ask` API endpoint via FastAPI.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Prepare docs:**
   Place all `.md`, `.pdf`, and `.html` docs into the `raw/` folder.
3. **Build the vector DB:**
   ```bash
   python scripts/build_vectordb.py
   ```
4. **Run the server:**
   ```bash
   uvicorn rag_server:app --reload
   ```

## API
- `POST /ask` with JSON `{ "question": "..." }` returns an answer and cited doc links.

## Environment Variables
- `OPENAI_API_KEY` required in a `.env` file.

## Files
- `rag_server.py`: FastAPI app
- `scripts/build_vectordb.py`: Script to chunk, embed, and index docs
- `faiss_index.pkl`: Vector DB

