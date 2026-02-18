import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("job_role_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("AI Job Role Predictor")

resume_input = st.text_area("Paste your resume text here:")

if st.button("Predict Job Role"):
    if resume_input:
        resume_tfidf = vectorizer.transform([resume_input])
        prediction = model.predict(resume_tfidf)
        st.success(f"Predicted Job Role: {prediction[0]}")
    else:
        st.warning("Please enter resume text.")