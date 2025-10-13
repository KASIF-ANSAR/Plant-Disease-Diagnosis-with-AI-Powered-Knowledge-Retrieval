# Plant-Disease-Diagnosis-with-AI-Powered-Knowledge-Retrieval

AI-powered plant disease diagnosis combining CNN-based leaf classification with a conversational RAG system for context-aware explanations and recommendations.

## **Problem Statement**
Early detection of plant diseases is critical for crop management. Manual inspection is slow, error-prone, and often inaccessible to farmers. This project automates disease detection and provides a conversational AI system to explain results, answer queries, and offer actionable recommendations.

## **Project Overview**
Plant Disease Diagnosis with AI-Powered Knowledge Retrieval is a web-based system designed to help farmers, researchers, and agricultural enthusiasts quickly identify diseases in potato and tomato leaves. The project combines:

- **Deep Learning (CNN)**: Custom convolutional neural network achieving **94% accuracy** on potato leaf classification.
- **Retrieval-Augmented Generation (RAG)**: Provides context-aware explanations and actionable recommendations from agricultural documents.
- **Interactive Interface**: Users can upload leaf images, view predictions with confidence scores, and interact with a conversational AI agent that maintains context across multiple queries.

## **Key Features**
- **Leaf Disease Classification**: Predicts diseases for potato and tomato leaves with confidence scores.  
- **Contextual Chat with RAG**: Ask questions about detected diseases; system provides informed, context-aware answers.  
- **Interactive Streamlit Interface**: Upload images, view predictions, and chat about diseases in one interface.  
- **Persistent User Context (planned full-stack)**: Store past inputs, predictions, and queries for a personalized experience.  

## **Technologies Used**
- Python 3.13  
- TensorFlow / tf_keras (CNN model)  
- LangChain + Gemini AI (RAG pipeline)  
- Chroma vectorstore for document embeddings  
- Streamlit for front-end interface  
<!-- OpenCV, PIL, NumPy for image processing  -->
- SQL (planned) for storing user history  

## **Dataset**
- [Plant Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) containing labeled images of potato and tomato leaves (healthy and diseased).  
- Used to train the CNN model, which achieved **95.21% test accuracy**.  

## **Usage**
1. Upload a potato or tomato leaf image.  
2. Click **Predict** to see the disease class and confidence score.  
3. Ask questions about the disease in the chat interface; the system maintains context across queries.  
4. *(Future)* Store user inputs, predictions, and queries in a SQL database for persistent history.  

## **Results & Performance**
- CNN model achieved **95.21% test accuracy** with a **loss of 0.1436**.  
- Training accuracy reached **95.64%** with a **loss of 0.1230**.  
- RAG system provides **context-aware answers** and actionable recommendations from agricultural documents.  

## **Future Work**
- Full-stack integration with SQL database for persistent user history.  
- Expand dataset to include more plant species and diseases.  
- Enhance RAG agent with multi-modal responses (image + text) and real-time detection highlights.  
