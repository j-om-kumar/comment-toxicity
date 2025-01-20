import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import os
import pandas as pd

st.title("Recognise Comment")

def score_comment(comment, df):
    # Load the pre-trained model
    new_model = load_model(os.path.join('models', 'toxicity.h5'))
    
    # Adapt the TextVectorization layer
    X = df["comment_text"]
    MAX_FEATURES = 200000
    vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                                   output_sequence_length=1800,
                                   output_mode='int')
    vectorizer.adapt(X.values)
    vectorized_comment = vectorizer([comment])
    
    # Predict
    results = new_model.predict(vectorized_comment)
    
    # Prepare results as a dictionary
    df.drop('threat', axis=1, inplace=True)  # Remove 'threat' column as per your logic
    output_dict = {}
    for idx, col in enumerate(df.columns[2:]):  # Adjust based on your CSV structure
        output_dict[col] = results[0][idx] > 0.5  # Convert probability to Boolean
    
    return output_dict

# Streamlit input/output
input_comment = st.text_input('Your comment')
submit = st.button('Submit')

if submit:
    df = pd.read_csv("./data/train.csv")  # Ensure the file path is correct
    predictions = score_comment(input_comment, df)
    
    st.write("Results:")
    for label, is_present in predictions.items():
        st.checkbox(label, value=is_present, key=label, disabled=True)
