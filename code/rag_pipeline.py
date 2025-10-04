import os
from google import genai
from langchain.schema import HumanMessage, AIMessage

# ------------------------
# Global caches and client
# ------------------------
import nest_asyncio
nest_asyncio.apply()


_client = None
_chat_history = []
_rag_answer_cache = {}  # cache for repeated questions

# ------------------------
# Gemini client init
# ------------------------
def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE"))
    return _client

# ------------------------
# Disease-only Gemini QA (fallback)
# ------------------------
def get_simple_response(disease_name: str, question: str):
    context_question = f"Disease: {disease_name}\nQuestion: {question}"
    if context_question in _rag_answer_cache:
        return _rag_answer_cache[context_question]

    client = get_client()
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=context_question
        )
        answer = response.text
        _rag_answer_cache[context_question] = answer
        _chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        return answer
    except Exception as e:
        return f"Error generating response: {e}"

# ------------------------
# Chroma RAG (documents + embeddings)
# ------------------------
try:
    from langchain_core.documents import Document
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.chains import RetrievalQA

    # Lazy embeddings function
    _embedding_function = None
    def get_embeddings():
        global _embedding_function
        if _embedding_function is None:
            _embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        return _embedding_function

    # Load persisted Chroma DB
    def load_vectorstore():
        if os.path.exists("./chroma_db"):
            return Chroma(
                persist_directory="./chroma_db",
                embedding_function=get_embeddings()
            )
        return None

    # Initialize RAG chain
    _rag_chain = None
    def get_rag_chain():
        global _rag_chain
        if _rag_chain is None:
            vectorstore = load_vectorstore()
            if vectorstore is None:
                return None
            retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
            _rag_chain = RetrievalQA.from_chain_type(
                ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0),
                retriever=retriever
            )
        return _rag_chain

    # Main RAG function
    # def get_rag_response(disease_name: str, question: str):
    #     # First try Chroma RAG
    #     rag_chain = get_rag_chain()
    #     if rag_chain:
    #         context_question = f"Disease: {disease_name}\nQuestion: {question}"
    #         if context_question in _rag_answer_cache:
    #             return _rag_answer_cache[context_question]
    #         try:
    #             answer = rag_chain.run(context_question)
    #             _rag_answer_cache[context_question] = answer
    #             _chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    #             return answer
    #         except Exception as e:
    #             # fallback to simple Gemini QA
    #             return get_simple_response(disease_name, question)
    #     else:
    #         # fallback to simple Gemini QA
    #         return get_simple_response(disease_name, question)


    def get_rag_response(disease_name: str, question: str):
    # Try Chroma RAG first
        rag_chain = get_rag_chain()
        if rag_chain:
            print("DEBUG: Using Chroma embeddings + RAG for answer")  # <-- add this
            context_question = f"Disease: {disease_name}\nQuestion: {question}"
            if context_question in _rag_answer_cache:
                return _rag_answer_cache[context_question]
            try:
                answer = rag_chain.run(context_question)
                _rag_answer_cache[context_question] = answer
                _chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
                return answer
            except Exception as e:
                print(f"DEBUG: Chroma failed, falling back to disease-only QA: {e}")  # <-- add this
                return get_simple_response(disease_name, question)
        else:
            print("DEBUG: Chroma not available, using disease-only QA")  # <-- add this
            return get_simple_response(disease_name, question)


except ImportError:
    print("LangChain / Chroma not installed. Only disease-only QA will work.")
    get_rag_response = get_simple_response








