import streamlit as st

# --- 0. Set page config first ---
st.set_page_config(page_title="🌿 Leaf Disease Classifier + RAG", layout="centered")

# --- 1. Other imports ---
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# --- 2. Set Google credentials ---
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\kasif\Desktop\leaf_rag\leaf-rag-f21a2c4777ea.json"
import os

# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\kasif\Desktop\leaf_rag\leaf-rag-8f4bb7ec1fe6.json"
 
# Set your API key (plain string)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCbZPjW9FHt9_mJ6I3K_iRlhAQnH6opaNo"


from rag_pipeline import get_rag_response

# --- 3. Constants ---
IMAGE_SIZE = 256
potato_classes = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Leaf_Mold",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_healthy"
]

# --- 4. Load model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("leaf_disease_model_1.keras")

model = load_model()

# --- 5. Streamlit UI ---
st.title("🌿 Leaf Disease Classification App")
st.write("Upload a leaf image, see its disease, and ask questions about it!")

# --- 6. Upload button ---
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption="Uploaded Image", use_container_width=True)
    st.image(image, caption="Uploaded Image")  # just this


    # Initialize session state
    if "pred_class" not in st.session_state:
        st.session_state.pred_class = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Predict"):
        st.write("Predicting...")

        # Preprocess
        img_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        img_batch = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_batch)
        pred_index = np.argmax(predictions[0])
        st.session_state.pred_class = potato_classes[pred_index]
        confidence = predictions[0][pred_index]

        # Display result
        st.success(f"Predicted Class: {st.session_state.pred_class}")
        st.info(f"Confidence: {confidence*100:.2f}%")

    # --- RAG QA ---
    if st.session_state.pred_class:
        st.write(f"Ask questions about **{st.session_state.pred_class}**:")
        user_question = st.text_input("Your question:", key="rag_input")
        
        if user_question:
            st.write("Hold on...")
            context_question = f"Disease: {st.session_state.pred_class}\nQuestion: {user_question}"
            answer = get_rag_response(context_question)
            st.session_state.chat_history.append((user_question, answer))

        # Display chat history
        for q, a in st.session_state.chat_history:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")



# # app.py
# import os
# from dotenv import load_dotenv
# import streamlit as st
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# # Load .env
# load_dotenv()  # reads .env in project root

# # --- 0. Page config ---
# st.set_page_config(page_title="🌿 Leaf Disease Classifier + RAG", layout="centered")

# # --- 1. Google credentials (choose one) ---
# # 1) If you have a service account JSON: set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json in .env
# # 2) If you have AI Studio API key: set GOOGLE_API_KEY=your_api_key in .env
# sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# api_key = os.getenv("AIzaSyCbZPjW9FHt9_mJ6I3K_iRlhAQnH6opaNo")

# if sa_path:
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path
# elif api_key:
#     os.environ["GOOGLE_API_KEY"] = api_key
# else:
#     st.warning("No Google credentials found. Put GOOGLE_API_KEY or GOOGLE_APPLICATIONS_CREDENTIALS in a .env file.")

# # --- 2. App imports that depend on credentials ---
# from rag_pipeline import get_rag_response  # local module

# # --- 3. Constants & classes ---
# IMAGE_SIZE = 256
# potato_classes = [
#     "Potato___Early_blight",
#     "Potato___Late_blight",
#     "Potato___healthy",
#     "Tomato_Leaf_Mold",
#     "Tomato__Tomato_YellowLeaf__Curl_Virus",
#     "Tomato_Bacterial_spot",
#     "Tomato_healthy"
# ]

# # --- 4. Load model (cached) ---
# @st.cache_resource
# def load_model(path="leaf_disease_model_1.keras"):
#     return tf.keras.models.load_model(path)

# model = load_model()

# # --- 5. UI ---
# st.title("🌿 Leaf Disease Classification + RAG")
# st.write("Upload a leaf image, get a prediction, and ask follow-up questions about the disease.")

# uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

# # initialize session state
# if "pred_class" not in st.session_state:
#     st.session_state.pred_class = None
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# if uploaded_file:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     if st.button("Predict"):
#         st.write("Predicting...")
#         # Preprocess: resize + normalize
#         img_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE))).astype(np.float32) / 255.0
#         img_batch = np.expand_dims(img_array, axis=0)

#         preds = model.predict(img_batch)
#         pred_index = int(np.argmax(preds[0]))
#         st.session_state.pred_class = potato_classes[pred_index]
#         confidence = float(preds[0][pred_index])

#         st.success(f"Prediction: {st.session_state.pred_class} (confidence {confidence:.2f})")

# # --- 6. Chat RAG ---
# if st.session_state.pred_class:
#     st.subheader("💬 Ask about the disease")
#     query = st.text_input("Your question:", key="user_query")

#     if st.button("Ask"):
#         if query.strip():
#             response = get_rag_response(query, st.session_state.pred_class, st.session_state.chat_history)
#             st.session_state.chat_history.append(("You", query))
#             st.session_state.chat_history.append(("Assistant", response))

#     # Display chat history
#     for role, msg in st.session_state.chat_history:
#         if role == "You":
#             st.markdown(f"**You:** {msg}")
#         else:
#             st.markdown(f"**Assistant:** {msg}")
