import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from pathlib import Path

# ----- Setup -----
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
STORE = Path("vector_store")

EMB_FILE = STORE / "embeddings.npy"
META_FILE = STORE / "metadata.json"
FAISS_FILE = STORE / "index.faiss"

# ----- Load model -----
model = SentenceTransformer(MODEL)

# ----- Load saved objects -----
embeddings = np.load(EMB_FILE)
index = faiss.read_index(str(FAISS_FILE))

with open(META_FILE, "r") as f:
    metadata = json.load(f)

# ----- Query function -----
def search(query_text, top_k=3):
    q_emb = model.encode([query_text], convert_to_numpy=True)
    q_emb = normalize(q_emb)

    D, I = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        results.append({
            "id": metadata[idx]["id"],
            "text": metadata[idx]["text"],
            "score": float(score)
        })

    return results

# ----- Test -----
if __name__ == "__main__":
    q = input("Enter your query: ")
    res = search(q, top_k=3)
    for r in res:
        print("--------")
        print("ID:", r["id"])
        print("Score:", round(r["score"], 3))
        print("Text:", r["text"])
