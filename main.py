# import libraries
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb

# Load the model
@st.cache_resource
def load_my_model():
    return load_model("simple_rnn_imdb.h5")

model = load_my_model()


# Load the data word index
word_index = imdb.get_word_index()

# Preprocess the text
def preprocess_text(txt):
    words = txt.strip().lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

# prediction
def predict(review):
    encoded_review = preprocess_text(review)
    prediction = model.predict(encoded_review)
    sentiment = "Positive" if prediction[0][0] > 0.5  else "Negative"
    return sentiment, prediction[0][0]

# Give title for streamlit app
st.title("IMDB Sentiment Analysis")

# take input from streamlit app
review = st.text_area("Enter the movie review:")

if st.button("Predict"):
    if review.strip() != "":
        sentiment, score = predict(review)
        if sentiment=="Positive":
            st.success(f"{sentiment} (Confidence : {score:.2f})")
        else:
            st.error(f"{sentiment} (Confidence : {score:.2f})")
    else:
        st.write("Please enter your review")

    