
# -----------  3  ------------
import os
import nest_asyncio
nest_asyncio.apply() 

from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage

# --- Lazy embeddings ---
_embedding_function = None
def get_embeddings():
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return _embedding_function

# --- Load documents ---
def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents

# --- Initialize Chroma vectorstore ---
_chroma_vectorstore = None
def get_vectorstore(documents: List[Document]):
    global _chroma_vectorstore 
    if _chroma_vectorstore is None:
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        _chroma_vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=get_embeddings(),
            persist_directory="./chroma_db"
        )
    return _chroma_vectorstore

# --- Create RAG retrieval chain using Gemini ---
_rag_chain = None
def get_rag_chain(vectorstore):
    global _rag_chain
    if _rag_chain is None:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        _rag_chain = RetrievalQA.from_chain_type(
            ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0),  
            retriever=retriever
        )
    return _rag_chain

# --- Main function to query RAG ---
_chat_history = []
def get_rag_response(question: str):
    documents_folder = "./docs" 
    docs = load_documents(documents_folder)
    vectorstore = get_vectorstore(docs)
    rag_chain = get_rag_chain(vectorstore)
    answer = rag_chain.run(question)
    _chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    return answer


# # rag_pipeline.py
# """
# RAG pipeline for Streamlit app.

# Features:
# - Uses Google Gemini for embeddings and chat (langchain_google_genai).
# - Persistent Chroma vector DB reuse to avoid re-embedding on every run.
# - Conversational retrieval chain with in-memory conversation buffer.
# - Safe fallbacks and simple API-key / service-account usage (set in .env via app.py).
# """

# import os
# import nest_asyncio
# from typing import List, Optional

# # Patch asyncio so LangChain works inside Streamlit
# nest_asyncio.apply()

# # LangChain / community imports
# from langchain_core.documents import Document
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory

# # --- Configuration ---
# DOCS_FOLDER = "./docs"
# CHROMA_DIR = "./chroma_db"
# EMBEDDING_MODEL = "models/embedding-001"       # embedding model name for Gemini
# CHAT_MODEL = "gemini-2.5-pro"          # chat model to use
# RETRIEVAL_K = 2                               # number of docs to retrieve

# # --- Globals to hold singletons in memory during the Streamlit session ---
# _embedding = None
# _vectorstore = None
# _rag_chain = None


# def get_embeddings() -> GoogleGenerativeAIEmbeddings:
#     """Lazily instantiate the Google Generative AI embeddings object."""
#     global _embedding
#     if _embedding is None:
#         _embedding = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
#     return _embedding


# def load_documents(folder_path: str = DOCS_FOLDER) -> List[Document]:
#     """Load supported documents from folder (pdf, docx)."""
#     documents: List[Document] = []
#     if not os.path.isdir(folder_path):
#         return documents

#     for filename in sorted(os.listdir(folder_path)):
#         file_path = os.path.join(folder_path, filename)
#         if not os.path.isfile(file_path):
#             continue
#         if filename.lower().endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         elif filename.lower().endswith(".docx"):
#             loader = Docx2txtLoader(file_path)
#         else:
#             continue
#         try:
#             documents.extend(loader.load())
#         except Exception as e:
#             print(f"Warning: failed to load {file_path}: {e}")
#     return documents


# def build_or_load_vectorstore() -> Optional[Chroma]:
#     """Build Chroma vectorstore if not present, otherwise load the persisted one."""
#     global _vectorstore
#     if _vectorstore is not None:
#         return _vectorstore

#     if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
#         try:
#             _vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=get_embeddings())
#             return _vectorstore
#         except Exception as e:
#             print(f"Warning: failed to load persisted Chroma DB: {e}. Will rebuild.")

#     docs = load_documents(DOCS_FOLDER)
#     if not docs:
#         print("No documents found in docs/ — RAG will be disabled.")
#         return None

#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = splitter.split_documents(docs)

#     _vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=get_embeddings(),
#         persist_directory=CHROMA_DIR
#     )
#     try:
#         _vectorstore.persist()
#     except Exception:
#         pass

#     return _vectorstore


# def get_rag_chain() -> Optional[ConversationalRetrievalChain]:
#     """Create or return a cached ConversationalRetrievalChain using Gemini chat + Chroma retriever."""
#     global _rag_chain
#     if _rag_chain is not None:
#         return _rag_chain

#     vectorstore = build_or_load_vectorstore()
#     if vectorstore is None:
#         return None

#     retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
#     llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)

#     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#     _rag_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         return_source_documents=False
#     )

#     return _rag_chain


# def get_rag_response(user_query: str, disease_label: Optional[str] = None, chat_history: Optional[List] = None) -> str:
#     """Top-level function used by the Streamlit app."""
#     try:
#         rag_chain = get_rag_chain()
#         if rag_chain is None:
#             return "No knowledge base loaded (no docs found in ./docs). Please add reference documents."

#         prefix = ""
#         if disease_label:
#             prefix = f"Disease: {disease_label}\n"

#         inputs = {"question": prefix + user_query}
#         if chat_history:
#             inputs["chat_history"] = chat_history

#         result = rag_chain.run(inputs)

#         if isinstance(result, str):
#             return result.strip()
#         if isinstance(result, dict):
#             return result.get("answer") or result.get("result") or str(result)
#         return str(result)

#     except Exception as e:
#         print(f"Error in get_rag_response: {e}")
#         return f"Error generating answer: {e}"
