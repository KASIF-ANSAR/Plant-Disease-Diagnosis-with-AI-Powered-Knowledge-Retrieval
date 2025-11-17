# rag_answer.py

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

from llm_answer import generate_gemini_answer



# ------------------------
# Paths to vector store
# ------------------------
STORE = Path(__file__).parent / "vector_store"

EMB_FILE = STORE / "embeddings.npy"
META_FILE = STORE / "metadata.json"
FAISS_FILE = STORE / "index.faiss"


# ------------------------
# Load Embedding Model + FAISS Index + Metadata
# ------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(MODEL_NAME)

# Load FAISS index
index = faiss.read_index(str(FAISS_FILE))

# Load metadata
with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)


# ------------------------
# Retrieve Top-K chunks via FAISS
# ------------------------
def retrieve_chunks(query: str, top_k: int = 3):
    """Embed query, search FAISS, return top-k text chunks."""

    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = normalize(q_emb)

    distances, indices = index.search(q_emb, top_k)

    chunks = []
    for idx in indices[0]:
        if idx < 0:
            continue
        chunks.append(metadata[idx]["text"])

    return "\n\n".join(chunks)


# ------------------------
# Full RAG Pipeline: Retrieval + Gemini
# ------------------------
def get_rag_response(disease_name: str, question: str):
    """Retrieve relevant chunks + call Gemini with context."""

    context = retrieve_chunks(question, top_k=3)

    # Debug print (optional)
    print("\n--- RETRIEVED CHUNKS ---")
    print(context)
    print("------------------------\n")

    # Call Gemini LLM
    result = generate_gemini_answer(
        question=question,
        context_text=context,
        disease_name=disease_name
    )

    return result

# # ------------------------
# # Manual test runner
# # ------------------------
# if __name__ == "__main__":
#     q = input("Enter your question: ")
#     disease = input("Enter disease name (optional): ")

#     result = get_rag_response(disease, q)

#     print("\nANSWER:\n", result["answer"])
#     print("\nSUMMARY:\n", result["summary"])
#     print("\nSTORAGE:\n", result["store_text"])

