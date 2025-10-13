import os
from google import genai
from langchain.schema import HumanMessage, AIMessage
import nest_asyncio

nest_asyncio.apply()

_chat_history = []
_rag_answer_cache = {}
_client = None  # define here

# ------------------
# Initialize Gemini client
# ------------------
def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("AIzaSyCbZPjW9FHt9_mJ6I3K_iRlhAQnH6opaNo")
        if not api_key:
            raise ValueError("Gemini API key missing!")
        _client = genai.Client(api_key=api_key)
    return _client


# ------------------
# Enhanced Gemini response with structured output
# ------------------
def get_simple_response(disease_name: str, question: str, context_text: str = ""):
    """
    Calls Gemini and asks it to return a concise, structured answer for plant diseases.

    Args:
        disease_name (str): Name of the disease (from your RAG retrieval).
        question (str): User question.
        context_text (str): Optional context retrieved from RAG.

    Returns:
        str: Structured, readable answer.
    """
    context_question = f"Disease: {disease_name}\nQuestion: {question}"

    if context_question in _rag_answer_cache:
        return _rag_answer_cache[context_question]

    client = get_client()
    # ---------------- Prompt Engineering ----------------
    prompt = f"""
You are an expert plant pathologist. Answer the following question about plant diseases using the structured format below.

Disease Name:
Affected Plants:
Key Symptoms:
Conditions Favoring Disease:
Spread:
Management:

Use bullet points where appropriate. Keep answers concise, clear, and actionable. Avoid unnecessary long explanations. 

Context Information (optional, use to improve accuracy):
{context_text}

Question:
{question}
"""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if not hasattr(response, "text") or not response.text:
            raise ValueError("Empty response from Gemini")

        answer = response.text.strip()
        _rag_answer_cache[context_question] = answer
        _chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=answer)
        ])
        return answer
    except Exception as e:
        raise RuntimeError(f"Gemini RAG generation failed → {e}")


# ------------------
# Wrapper for Flask
# ------------------
def get_rag_response(disease_name: str, question: str, context_text: str = ""):
    return get_simple_response(disease_name, question, context_text)



# import os
# from google import genai
# from langchain.schema import HumanMessage, AIMessage
# import nest_asyncio
# nest_asyncio.apply()

# _chat_history = []
# _rag_answer_cache = {}
# # api_key = "AIzaSyCbZPjW9FHt9_mJ6I3K_iRlhAQnH6opaNo"

# # _client = None

# # ------------------
# # Initialize Gemini client
# # ------------------
# # def get_client():
# #     global _client
# #     if _client is None:
# #         if not api_key:
# #             raise ValueError("API key not set")
# #         _client = genai.Client(api_key=api_key)
# #     return _client

# def get_client():
#     global _client
#     if _client is None:
#         _client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", "AIzaSyCbZPjW9FHt9_mJ6I3K_iRlhAQnH6opaNo"))
#     return _client


# # ------------------
# # Simple Gemini response
# # ------------------
# def get_simple_response(disease_name: str, question: str):
#     context_question = f"Disease: {disease_name}\nQuestion: {question}"
#     if context_question in _rag_answer_cache:
#         return _rag_answer_cache[context_question]

#     client = get_client()
#     try:
#         response = client.models.generate_content(
#             model="gemini-2.5-flash",
#             contents=context_question
#         )
#         answer = response.text
#         _rag_answer_cache[context_question] = answer
#         _chat_history.extend([HumanMessage(content=question), AIMessage(content=answer)])
#         return answer
#     except Exception as e:
#         return f"Error generating response: {e}"

# # ------------------
# # Wrapper for Flask
# # ------------------
# def get_rag_response(disease_name: str, question: str):
#     # You can add Chroma / embedding retrieval here if needed
#     return get_simple_response(disease_name, question)
