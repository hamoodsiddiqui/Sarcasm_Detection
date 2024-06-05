import streamlit as st
import joblib

model = joblib.load("./Model/naive_bayes_classifier.pkl")


def predict(sentence):
    sentence = sentence.lower()
    prediction = model.predict([sentence])[0]
    return prediction


st.title("Sarcasm Detector")

st.divider()


sentence = st.text_area("Enter your sentence", placeholder="Enter your sentence")

if st.button("Detect", type="primary"):
    if sentence:
        prediction = predict(sentence)
        if prediction == 1:
            st.info("This sentence is sarcastic.")
        else:
            st.info("This sentence is not sarcastic.")
    else:
        st.error("Please enter any sentence first.")
