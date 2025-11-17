# llm_answer.py

import re
from google import genai 

from config import GEMINI_API_KEY



# ------------------------
# Initialize Gemini client (single global client)
# ------------------------
client = genai.Client(api_key=GEMINI_API_KEY)


# ------------------------
# Generate answer from chunks + question
# ------------------------
def generate_gemini_answer(question: str, context_text: str = "", disease_name: str = ""):
    prompt = f""" 
You are a strict RAG assistant. Use ONLY the retrieved context to answer.

If the context does NOT contain the information needed, reply:
"Insufficient information in the retrieved documents to answer this question."

Do NOT use your own knowledge.

Disease (if any): {disease_name}
User Question: {question}

Retrieved Context:
{context_text}

Follow this EXACT format:

Answer: <answer based ONLY on context or 'Insufficient information...'>

Summary: <short summary>

Storage: <canonical version>
"""

 
# You are a strict RAG assistant. Use ONLY the retrieved context to answer.

# If the context does NOT contain the information needed, reply:
# "Insufficient information in the retrieved documents to answer this question."

# Do NOT use your own knowledge.

# Disease (if any): {disease_name}
# User Question: {question}

# Retrieved Context:
# {context_text}

# Follow this EXACT format:

# Answer: <answer based ONLY on context or 'Insufficient information...'>

# Summary: <short summary>

# Storage: <canonical version>
# """

# You are a helpful subject-matter expert. Answer the user's question in three sections.

# Disease (if any): {disease_name}
# User Question: {question}

# Retrieved Context:
# {context_text}

# Follow this EXACT format:

# Answer: <full detailed answer here>

# Summary: <short summary here>

# Storage: <canonical version for database storage>
 

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = response.text.strip()

        # Extract 3 parts
        answer_match = re.search(r"Answer:\s*(.*?)\nSummary:", text, re.DOTALL | re.IGNORECASE)
        summary_match = re.search(r"Summary:\s*(.*?)\nStorage:", text, re.DOTALL | re.IGNORECASE)
        storage_match = re.search(r"Storage:\s*(.*)", text, re.DOTALL | re.IGNORECASE)

        answer = answer_match.group(1).strip() if answer_match else text
        summary = summary_match.group(1).strip() if summary_match else ""
        storage = storage_match.group(1).strip() if storage_match else text

        return {
            "answer": answer,
            "summary": summary,
            "store_text": storage
        }

    except Exception as e:
        raise RuntimeError(f"❌ Gemini LLM generation failed: {e}")
