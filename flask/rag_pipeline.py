import os
import numpy as np
from google import genai
from langchain.schema import HumanMessage, AIMessage
import nest_asyncio
import re
import faiss
from sentence_transformers import SentenceTransformer

nest_asyncio.apply()

_chat_history = []
_rag_answer_cache = {}
_client = None  # Gemini client
_faiss_index = None
_documents = None
_embedding_model = None

# --- Configuration ---
FAISS_INDEX_FILE = "faiss_index.idx"
DOCS_MAPPING_FILE = "doc_texts.npy"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3  # Number of most relevant documents to retrieve

# ------------------
# Initialize Gemini client
# ------------------
def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key missing!")
        _client = genai.Client(api_key=api_key)
    return _client

# ------------------
# Load FAISS index and documents
# ------------------
def load_faiss():
    global _faiss_index, _documents, _embedding_model
    if _faiss_index is None:
        _faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        _documents = np.load(DOCS_MAPPING_FILE, allow_pickle=True)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# ------------------
# Retrieve top-k relevant documents
# ------------------
def retrieve_relevant_docs(query: str, k=TOP_K):
    load_faiss()
    query_embedding = _embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = _faiss_index.search(query_embedding, k)
    docs = [str(_documents[idx]) for idx in indices[0] if idx != -1]
    return "\n\n".join(docs)  # concatenate top-k docs as context

# ------------------
# Enhanced Gemini response with FAISS context
# ------------------
def get_rag_response(disease_name: str, question: str, context_text: str = ""):
    # Retrieve FAISS-based context if not explicitly provided
    if not context_text:
        context_text = retrieve_relevant_docs(question)

    context_question = f"Disease: {disease_name}\nQuestion: {question}"
    if context_question in _rag_answer_cache:
        return _rag_answer_cache[context_question]

    client = get_client()
    prompt = f"""
    You are a humble, helpful plant pathology expert. Answer the user's question in three sections: Answer, Summary, and Storage.
    Disease: {disease_name}
    User Question: {question}
    Context (if any): {context_text}

    Guidelines:
    - **Answer:** Friendly, practical, well-structured for the user.
    - **Summary:** Short concise summary, for display.
    - **Storage:** A clear, canonical version of the answer suitable for storing in the database and future RAG retrieval.
    Do NOT include unnecessary fluff.
    - Use markdown headings and bullet points in Answer.
    - End each section with a newline.

    Format strictly as:
    Answer: <full answer here>
    Summary: <summary here>
    Storage: <text to store here>
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if not hasattr(response, "text") or not response.text:
            raise ValueError("Empty response from Gemini")

        text = response.text.strip()

        # Extract sections
        answer_match = re.search(r"Answer:\s*(.*?)\nSummary:", text, flags=re.IGNORECASE | re.DOTALL)
        summary_match = re.search(r"Summary:\s*(.*?)\nStorage:", text, flags=re.IGNORECASE | re.DOTALL)
        storage_match = re.search(r"Storage:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)

        answer = answer_match.group(1).strip() if answer_match else text
        summary = summary_match.group(1).strip() if summary_match else ""
        store_text = storage_match.group(1).strip() if storage_match else text

        # Cache all three
        _rag_answer_cache[context_question] = {
            "answer": answer,
            "summary": summary,
            "store_text": store_text
        }

        _chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])

        return {
            "answer": answer,
            "summary": summary,
            "store_text": store_text
        }

    except Exception as e:
        raise RuntimeError(f"Gemini RAG generation failed → {e}")
