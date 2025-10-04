import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from rag_pipeline import get_rag_response
import os

# ------------------------
# Page setup
# ------------------------
st.set_page_config(page_title="🌿 Leaf Disease Classifier + RAG", layout="centered")
st.title("🌿 Leaf Disease Classification App")
st.write("Upload a leaf image, see its disease, and ask questions about it!")

# ------------------------
# Set your API key
# ------------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyCbZPjW9FHt9_mJ6I3K_iRlhAQnH6opaNo"

# ------------------------
# Constants
# ------------------------
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

# ------------------------
# Load TF model
# ------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("leaf_disease_model_1.keras")

model = load_model()

# ------------------------
# Streamlit UI
# ------------------------
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Session state
    if "pred_class" not in st.session_state:
        st.session_state.pred_class = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ------------------------
    # Predict disease
    # ------------------------
    if st.button("Predict"):
        st.write("Predicting...")
        img_array = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))
        img_batch = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_batch)
        pred_index = np.argmax(predictions[0])
        st.session_state.pred_class = potato_classes[pred_index]
        confidence = predictions[0][pred_index]

        st.success(f"Predicted Class: {st.session_state.pred_class}")
        st.info(f"Confidence: {confidence*100:.2f}%")

    # ------------------------
    # RAG / Gemini QA
    # ------------------------
    if st.session_state.pred_class:
        st.write(f"Ask questions about **{st.session_state.pred_class}**:")
        user_question = st.text_input("Your question:", key="rag_input")

        if user_question:
            st.write("Hold on, generating answer...")
            answer = get_rag_response(st.session_state.pred_class, user_question)
            st.session_state.chat_history.append((user_question, answer))

        # Display chat history
        for q, a in st.session_state.chat_history:
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")






