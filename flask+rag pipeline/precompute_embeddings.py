# precompute_embeddings.py

import os
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# --- Configuration ---
DOCS_FOLDER = "docs"
FAISS_INDEX_FILE = "faiss_index.idx"
DOCS_MAPPING_FILE = "doc_texts.npy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Step 1: Read PDFs and extract text ---
documents = []
filenames = []

for filename in os.listdir(DOCS_FOLDER):
    if filename.endswith(".pdf"):
        path = os.path.join(DOCS_FOLDER, filename)
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        if text.strip():
            documents.append(text)
            filenames.append(filename)

print(f"Extracted text from {len(documents)} PDFs")

# --- Step 2: Generate embeddings ---
print("Generating embeddings...")
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(documents, convert_to_numpy=True)

# --- Step 3: Create FAISS index ---
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors")

# --- Step 4: Save FAISS index and mapping ---
faiss.write_index(index, FAISS_INDEX_FILE)
np.save(DOCS_MAPPING_FILE, np.array(documents))
print(f"FAISS index saved to {FAISS_INDEX_FILE}")
print(f"Document mapping saved to {DOCS_MAPPING_FILE}")

# Optional: Save filenames if you want to keep track
np.save("doc_filenames.npy", np.array(filenames))
print("Filenames saved to doc_filenames.npy")
