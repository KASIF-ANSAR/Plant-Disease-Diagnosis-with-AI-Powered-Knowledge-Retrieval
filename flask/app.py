# from flask import Flask, request, jsonify
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io
# from rag_pipeline import get_rag_response  # our RAG function

# app = Flask(__name__)

# # Load model once
# model = tf.keras.models.load_model("models/my_model.h5")

# def preprocess_image(image):
#     image = image.resize((224, 224))
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     return image

# # ---- Routes ----

# @app.route("/predict", methods=["POST"])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files['image']
#     img = Image.open(io.BytesIO(file.read()))
#     img = preprocess_image(img)

#     prediction = model.predict(img)
#     label = int(np.argmax(prediction))
#     confidence = float(np.max(prediction))

#     # Optional: map label index to actual class name
#     class_names = [
#         "Potato___Early_blight",
#         "Potato___Late_blight",
#         "Potato___healthy",
#         "Tomato_Leaf_Mold",
#         "Tomato__Tomato_YellowLeaf__Curl_Virus",
#         "Tomato_Bacterial_spot",
#         "Tomato_healthy"
#     ]
#     return jsonify({
#         "label": class_names[label],
#         "confidence": round(confidence, 2)
#     })


# @app.route("/rag", methods=["POST"])
# def rag():
#     data = request.json
#     disease = data.get("disease")
#     question = data.get("question")

#     if not disease or not question:
#         return jsonify({"error": "Missing disease or question"}), 400

#     try:
#         answer = get_rag_response(disease, question)
#         return jsonify({"answer": answer})
#     except Exception as e:
#         return jsonify({"error": f"RAG request failed: {str(e)}"}), 500


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)




from flask import Flask, request, jsonify
from rag_pipeline import get_rag_response
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# --- 1. Load your model ---
model = tf.keras.models.load_model("models/my_model.h5")

# --- 2. Class names ---
potato_classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Leaf_Mold",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_healthy"
]

# --- 3. Preprocessing function ---
def preprocess_image(image):
    IMAGE_SIZE = 256   # match the training size
    image = image.convert("RGB")             # ensure 3 channels
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)                  # keep 0–255 if model trained that way
    image = np.expand_dims(image, axis=0)    # add batch dimension
    return image

# --- 4. Routes ---
@app.route("/")
def home():
    return "Flask ML server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read and preprocess image
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    img = preprocess_image(img)

    # Predict
    prediction = model.predict(img)
    pred_index = int(np.argmax(prediction))
    label = potato_classes[pred_index]
    confidence = round(float(np.max(prediction)), 2)  # 2 decimal places


    print(f"Prediction: {label}, Confidence: {confidence:.4f}")  # logs in terminal

    return jsonify({"label": label, "confidence": confidence})
 

@app.route("/rag", methods=["POST"])
def rag():
    data = request.json
    disease = data.get("disease")
    question = data.get("question")

    if not disease or not question:
        return jsonify({"error": "Missing disease or question"}), 400

    try:
        answer = get_rag_response(disease, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": f"RAG request failed: {str(e)}"}), 500

# --- 5. Run Flask ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
