import re
import string
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import joblib

# Initialize spaCy
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.keep_words = {
            "not", "no", "never", "none",
            "can", "could", "may", "might", "must", "should",
            "all", "any", "some", "few", "many",
            "now", "soon", "immediately", "recently",
            "if", "unless", "until",
            "high", "critical", "low", "medium", "zero"
        }
        
        # Update stop words
        for word in self.keep_words:
            if word in STOP_WORDS:
                STOP_WORDS.remove(word)
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_text(self, text):
        doc = nlp(text)
        tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop 
            and not token.is_punct 
            and token.is_alpha
            and len(token.lemma_) > 1
        ]
        return " ".join(tokens)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(lambda x: self.preprocess_text(self.clean_text(x)))
        elif isinstance(X, np.ndarray):
            return np.array([self.preprocess_text(self.clean_text(text)) for text in X])
        elif isinstance(X, str):
            return np.array([self.preprocess_text(self.clean_text(X))])
        else:
            raise ValueError("Input must be string, NumPy array, or Pandas Series")

# Load the trained pipeline
@st.cache_resource
def load_model():
    return joblib.load('incident_classifier_pipeline.pkl')

@st.cache_resource
def load_label_encoder():
    # You'll need to save and load your label encoder
    # For now, we'll create a dummy one (replace with your actual LE)
    le = LabelEncoder()
    # Add your classes here if needed
    le.classes_ = np.array(['high', 'low', 'medium', 'no'])  # Replace with your actual classes
    return le

pipeline = load_model()
le = load_label_encoder()

# Streamlit app
def main():
    st.title("Security Incident Classifier")
    st.write("This app classifies security incident reports using a trained ML model.")

    # Input text
    user_input = st.text_area("Enter the security incident text to classify:", 
                             "Bandits kill 19, raze houses in fresh Katsina attack")

    if st.button("Classify"):
        if user_input:
            # Make prediction
            processed_text = TextPreprocessor().transform(user_input)
            prediction = pipeline.predict(processed_text)
            predicted_class = le.inverse_transform(prediction)[0]
            
            # Display results
            st.subheader("Prediction Result")
            st.success(f"Predicted Class: {predicted_class}")
            
            # Show confidence scores (if your model supports predict_proba)
            if hasattr(pipeline, 'predict_proba'):
                probabilities = pipeline.predict_proba(processed_text)[0]
                st.write("Classification Probabilities:")
                for i, prob in enumerate(probabilities):
                    st.write(f"{le.classes_[i]}: {prob:.2f}")
        else:
            st.warning("Please enter some text to classify.")

if __name__ == '__main__':
    main()