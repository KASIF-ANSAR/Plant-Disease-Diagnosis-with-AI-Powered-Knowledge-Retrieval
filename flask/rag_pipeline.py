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
        api_key = os.environ.get("GOOGLE_API_KEY", "AIzaSyCbZPjW9FHt9_mJ6I3K_iRlhAQnH6opaNo")
        if not api_key:
            raise ValueError("Gemini API key missing!")
        _client = genai.Client(api_key=api_key)
    return _client


# ------------------
# Enhanced Gemini response with structured output
# ------------------
import re

def get_rag_response(disease_name: str, question: str, context_text: str = ""):
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


