# app.py
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load the BERT model and tokenizer
model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Initialize the NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

# Streamlit App Interface
st.title("Named Entity Recognition (NER) with BERT")
st.write("This app uses a BERT-based model to recognize entities in the text you enter below.")

# Text input
text = st.text_area("Enter Text:", "")

# Analyze button
if st.button("Analyze"):
    if text:
        # Run NER pipeline
        ner_results = ner_pipeline(text)
        
        # Display Results
        st.write("**Entities Found:**")
        for entity in ner_results:
            st.write(f"Entity: `{entity['word']}`, Type: `{entity['entity']}`, Score: `{entity['score']:.4f}`")
    else:
        st.write("Please enter some text to analyze.")
