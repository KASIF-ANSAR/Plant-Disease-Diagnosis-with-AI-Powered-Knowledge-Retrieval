# app.py (RAG SERVICE)
from flask import Flask, request, jsonify
from flask_cors import CORS

from rag_answer import get_rag_response

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "RAG service is running!", 200


@app.route("/rag", methods=["POST"])
def rag_handler():
    data = request.json

    question = data.get("question")
    disease = data.get("disease", "")

    if not question:
        return jsonify({"error": "Missing question"}), 400

    try:
        result = get_rag_response(disease_name=disease, question=question)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
