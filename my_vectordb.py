import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# This path assumes that 'my_vectordb.py' is at the root of your app on Render (e.g., /app/my_vectordb.py)
# and your FAISS index disk is mounted as a directory named 'faiss_index' also at the root (e.g., /app/faiss_index).
FAISS_INDEX_PATH = "faiss_index"

retriever = None

# Critical: OpenAIEmbeddings() will raise an error if OPENAI_API_KEY is not set.
if os.getenv("OPENAI_API_KEY") is None:
    print("CRITICAL ERROR: OPENAI_API_KEY environment variable is not set. Retriever will not be initialized.")
else:
    try:
        embeddings = OpenAIEmbeddings()
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.isdir(FAISS_INDEX_PATH) or not os.listdir(FAISS_INDEX_PATH):
            print(f"WARNING: FAISS index directory not found, is not a directory, or is empty at '{FAISS_INDEX_PATH}'.")
            print("Please ensure you have built and uploaded the FAISS index (index.faiss and index.pkl files) to this location on your Render disk.")
            print("The application might not function correctly without the vector database.")
        else:
            vectordb = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            retriever = vectordb.as_retriever()
            print(f"FAISS index loaded successfully from '{FAISS_INDEX_PATH}'. Retriever is ready.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize OpenAIEmbeddings or load FAISS index from '{FAISS_INDEX_PATH}': {e}")
        print("The application might not function correctly. Check OPENAI_API_KEY and FAISS index path/contents.")
