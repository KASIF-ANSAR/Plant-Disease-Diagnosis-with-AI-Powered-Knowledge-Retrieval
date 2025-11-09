# # app.py
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from werkzeug.security import generate_password_hash, check_password_hash
# from db import db
# from models import User, Upload, Question
# from rag_pipeline import get_rag_response
# from PIL import Image
# import numpy as np
# import io
# import tensorflow as tf
# from sqlalchemy import text

# # ---------------- Flask Setup ----------------
# app = Flask(__name__)
# CORS(app)

# # ---------------- Database Config ----------------
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Me1%40sql.com@localhost/leaf_rag'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db.init_app(app)

# # Test DB connection
# with app.app_context():
#     try:
#         result = db.session.execute(text("SELECT DATABASE();"))
#         db_name = result.fetchone()[0]
#         print(f"✅ Connected to database: {db_name}")
#     except Exception as e:
#         print(f"❌ Database connection failed: {e}")

#     # Create tables if not exist
#     db.create_all()

# # ---------------- Load Model ----------------
# model = tf.keras.models.load_model("models/my_model.h5")
# potato_classes = [
#     "Potato___Early_blight",
#     "Potato___Late_blight",
#     "Potato___healthy",
#     "Tomato_Leaf_Mold",
#     "Tomato__Tomato_YellowLeaf__Curl_Virus",
#     "Tomato_Bacterial_spot",
#     "Tomato_healthy"
# ]

# def preprocess_image(image):
#     IMAGE_SIZE = 256
#     image = image.convert("RGB")
#     image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
#     image = np.array(image)
#     image = np.expand_dims(image, axis=0)
#     return image

# # ---------------- Routes ----------------
# @app.route("/")
# def home():
#     return "Flask ML server is running!"

# # --------- Register ---------
# @app.route("/register", methods=["POST"])
# def register():
#     data = request.json
#     username = data.get("username")
#     email = data.get("email")
#     password = data.get("password")

#     if not username or not email or not password:
#         return jsonify({"error": "Missing fields"}), 400

#     if User.query.filter((User.username==username)|(User.email==email)).first():
#         return jsonify({"error": "User already exists"}), 400

#     password_hash = generate_password_hash(password)
#     user = User(username=username, email=email, password_hash=password_hash)
#     db.session.add(user)
#     db.session.commit()

#     return jsonify({"message": "User registered successfully"}), 201

# # --------- Login ---------
# @app.route("/login", methods=["POST"])
# def login():
#     data = request.json
#     email = data.get("email")
#     password = data.get("password")

#     if not email or not password:
#         return jsonify({"error": "Missing fields"}), 400

#     user = User.query.filter_by(email=email).first()  # only email
#     if not user or not check_password_hash(user.password_hash, password):
#         return jsonify({"error": "Invalid credentials"}), 401

#     return jsonify({"message": "Login successful", "user_id": user.id, "username": user.username})

# # --------- Predict ---------
# @app.route("/predict", methods=["POST"])
# def predict():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     user_id = request.form.get("user_id")
#     if not user_id:
#         return jsonify({"error": "Missing user_id"}), 400

#     file = request.files['image']
#     img = Image.open(io.BytesIO(file.read()))
#     img = preprocess_image(img)

#     prediction = model.predict(img)
#     pred_index = int(np.argmax(prediction))
#     label = potato_classes[pred_index]
#     confidence = round(float(np.max(prediction)), 2)

#     try:
#         upload = Upload(
#             user_id=int(user_id),
#             image_path=file.filename,
#             prediction=label,
#             confidence=confidence
#         )
#         db.session.add(upload)
#         db.session.commit()
#     except Exception as e:
#         print(f"Error saving upload: {e}")

#     return jsonify({
#         "label": label,
#         "confidence": confidence,
#         "upload_id": upload.id
#     })

# # --------- RAG ---------
# import json
# @app.route("/rag", methods=["POST"])
# def rag():
#     data = request.json
#     disease = data.get("disease")
#     question = data.get("question")
#     user_id = data.get("user_id")
#     upload_id = data.get("upload_id")

#     if not disease or not question or not user_id or not upload_id:
#         return jsonify({"error": "Missing required fields"}), 400

#     try:
#         # get_rag_response now returns {"answer": ..., "summary": ..., "store_text": ...}
#         answer_data = get_rag_response(disease, question)

#         # Save to DB: store the "store_text" part as canonical text
#         new_question = Question(
#             user_id=user_id,
#             upload_id=upload_id,
#             question=question,
#             answer=answer_data["store_text"],  # store canonical text for retrieval
#             summary=disease
#         )
#         db.session.add(new_question)
#         db.session.commit()

#         # Return user-facing answer and summary to frontend
#         return jsonify({
#             "answer": answer_data["answer"],       # full friendly answer
#             "summary": answer_data.get("summary", ""),  # concise summary
#             "question_id": new_question.id
#         })

#     except Exception as e:
#         return jsonify({"error": f"RAG request failed: {str(e)}"}), 500


# # ---------------- Run ----------------
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)









# from db import db
# from models import User, Upload, Question
# from flask_cors import CORS  # allow cross-origin requests from React
# from flask import Flask, request, jsonify
# from rag_pipeline import get_simple_response
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io 
# from werkzeug.security import generate_password_hash, check_password_hash  


# from sqlalchemy import text  # add this at the top

# app = Flask(__name__)

# # Enable CORS
# CORS(app)

# # Database config
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Me1%40sql.com@localhost/leaf_rag'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # Initialize DB
# db.init_app(app)


# with app.app_context():
#     try:
#         result = db.session.execute(text("SELECT DATABASE();"))
#         db_name = result.fetchone()[0]
#         print(f"✅ Connected to database: {db_name}")
#     except Exception as e:
#         print(f"❌ Database connection failed: {e}")



# # Create tables (first time)
# with app.app_context():
#     db.create_all()


# # --- 1. Load your model ---
# model = tf.keras.models.load_model("models/my_model.h5")

# # --- 2. Class names ---
# potato_classes = [
#     "Potato___Early_blight",
#     "Potato___Late_blight",
#     "Potato___healthy",
#     "Tomato_Leaf_Mold",
#     "Tomato__Tomato_YellowLeaf__Curl_Virus",
#     "Tomato_Bacterial_spot",
#     "Tomato_healthy"
# ]

# # --- 3. Preprocessing function ---
# def preprocess_image(image):
#     IMAGE_SIZE = 256   # match the training size
#     image = image.convert("RGB")             # ensure 3 channels
#     image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
#     image = np.array(image)                  # keep 0–255 if model trained that way
#     image = np.expand_dims(image, axis=0)    # add batch dimension
#     return image

# # --- 4. Routes ---
# @app.route("/")
# def home():
#     return "Flask ML server is running!"


# # ---------------- Register Endpoint ----------------
# @app.route("/register", methods=["POST"])
# def register():
#     data = request.json
#     username = data.get("username")
#     email = data.get("email")
#     password = data.get("password")

#     if not username or not email or not password:
#         return jsonify({"error": "Missing fields"}), 400

#     # Check if user exists
#     if User.query.filter((User.username==username)|(User.email==email)).first():
#         return jsonify({"error": "User already exists"}), 400

#     # Hash the password
#     password_hash = generate_password_hash(password)

#     # Create user
#     user = User(username=username, email=email, password_hash=password_hash)
#     db.session.add(user)
#     db.session.commit()

#     return jsonify({"message": "User registered successfully"}), 201

# # ---------------- Login Endpoint ----------------
# @app.route("/login", methods=["POST"])
# def login():
#     data = request.json
#     email = data.get("email")
#     password = data.get("password")

#     if not email or not password:
#         return jsonify({"error": "Missing fields"}), 400

#     # Find user
#     user = User.query.filter_by(email=email).first()
#     if not user or not check_password_hash(user.password_hash, password):
#         return jsonify({"error": "Invalid credentials"}), 401

#     # For now, we just return user id and username (later JWT or session)
#     return jsonify({"message": "Login successful", "user_id": user.id, "username": user.username})


# @app.route("/predict", methods=["POST"])
# def predict():
#     # --- 1. Check image ---
#     if 'image' not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     # --- 2. Read user_id from form (sent from React) ---
#     user_id = request.form.get("user_id")
#     if not user_id:
#         return jsonify({"error": "Missing user_id"}), 400

#     # --- 3. Read and preprocess image ---
#     file = request.files['image']
#     img = Image.open(io.BytesIO(file.read()))
#     img = preprocess_image(img)

#     # --- 4. Make prediction ---
#     prediction = model.predict(img)
#     pred_index = int(np.argmax(prediction))
#     label = potato_classes[pred_index]
#     confidence = round(float(np.max(prediction)), 2)  # 2 decimals

#     print(f"Prediction: {label}, Confidence: {confidence:.4f}")  # terminal log

#     # --- 5. Save prediction to DB ---
#     try:
#         upload = Upload(
#             user_id=int(user_id),
#             image_path=file.filename,  # can be saved path if you store images
#             prediction_label=label,
#             prediction_confidence=confidence
#         )
#         db.session.add(upload)
#         db.session.commit()
#     except Exception as e:
#         print(f"Error saving upload: {e}")

#     # --- 6. Return prediction to frontend ---
#     return jsonify({
#         "label": label,
#         "confidence": confidence,
#         "upload_id": upload.id  # send back so you can link questions to it
#     })


# @app.route("/rag", methods=["POST"])
# def rag():
#     data = request.json
#     disease = data.get("disease")
#     question = data.get("question")
#     user_id = data.get("user_id")
#     upload_id = data.get("upload_id")  # link to the image prediction

#     if not disease or not question or not user_id or not upload_id:
#         return jsonify({"error": "Missing required fields"}), 400

#     try:
#         # --- 1. Generate RAG answer ---
#         answer = get_simple_response(disease, question)

#         # --- 2. Store question + answer in DB ---
#         from models import UserQuestion  # assuming you have this model

#         new_question = UserQuestion(
#             user_id=user_id,
#             upload_id=upload_id,
#             question=question,
#             answer=answer,
#             disease=disease
#         )
#         db.session.add(new_question)
#         db.session.commit()

#         # --- 3. Return answer to frontend ---
#         return jsonify({
#             "answer": answer,
#             "question_id": new_question.id  # optional: return DB ID
#         })

#     except Exception as e:
#         return jsonify({"error": f"RAG request failed: {str(e)}"}), 500

# # --- 5. Run Flask ---
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)






 

# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from db import db
from models import User, Upload, Question
from rag_pipeline import get_rag_response
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from sqlalchemy import text

# ---------------- Flask Setup ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Database Config ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Me1%40sql.com@localhost/leaf_rag'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# Test DB connection
with app.app_context():
    try:
        result = db.session.execute(text("SELECT DATABASE();"))
        db_name = result.fetchone()[0]
        print(f"✅ Connected to database: {db_name}")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")

    # Create tables if not exist
    db.create_all()

# ---------------- Load Model ----------------
model = tf.keras.models.load_model("models/my_model.h5")
potato_classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Leaf_Mold",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_healthy"
]

def preprocess_image(image):
    IMAGE_SIZE = 256
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

# ---------------- Routes ----------------
@app.route("/")
def home():
    return "Flask ML server is running!"

# --------- Register ---------
@app.route("/register", methods=["POST"])
def register():
    data = request.json
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"error": "Missing fields"}), 400

    if User.query.filter((User.username==username)|(User.email==email)).first():
        return jsonify({"error": "User already exists"}), 400

    password_hash = generate_password_hash(password)
    user = User(username=username, email=email, password_hash=password_hash)
    db.session.add(user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201

# --------- Login ---------
@app.route("/login", methods=["POST"])
def login():
    data = request.json
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Missing fields"}), 400

    user = User.query.filter_by(email=email).first()  # only email
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid credentials"}), 401

    return jsonify({"message": "Login successful", "user_id": user.id, "username": user.username})

# --------- Predict ---------
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    user_id = request.form.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))
    img = preprocess_image(img)

    prediction = model.predict(img)
    pred_index = int(np.argmax(prediction))
    label = potato_classes[pred_index]
    confidence = round(float(np.max(prediction)), 2)

    try:
        upload = Upload(
            user_id=int(user_id),
            image_path=file.filename,
            prediction=label,
            confidence=confidence
        )
        db.session.add(upload)
        db.session.commit()
    except Exception as e:
        print(f"Error saving upload: {e}")

    return jsonify({
        "label": label,
        "confidence": confidence,
        "upload_id": upload.id
    })

# --------- RAG ---------
import json
@app.route("/rag", methods=["POST"])
def rag():
    data = request.json
    disease = data.get("disease")
    question = data.get("question")
    user_id = data.get("user_id")
    upload_id = data.get("upload_id")

    if not disease or not question or not user_id or not upload_id:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # get_rag_response now returns {"answer": ..., "summary": ..., "store_text": ...}
        answer_data = get_rag_response(disease, question)

        # Save to DB: store the "store_text" part as canonical text
        new_question = Question(
            user_id=user_id,
            upload_id=upload_id,
            question=question,
            answer=answer_data["store_text"],  # store canonical text for retrieval
            summary=disease
        )
        db.session.add(new_question)
        db.session.commit()

        # Return user-facing answer and summary to frontend
        return jsonify({
            "answer": answer_data["answer"],       # full friendly answer
            "summary": answer_data.get("summary", ""),  # concise summary
            "question_id": new_question.id
        })

    except Exception as e:
        return jsonify({"error": f"RAG request failed: {str(e)}"}), 500


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
