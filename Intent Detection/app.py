import zipfile
import os
import torch
import streamlit as st
import nltk

from nltk.corpus import stopwords
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --- File and Directory Configuration ---
ZIP_FILE_PATH = 'distilbert-model.zip'
EXTRACT_DIR = 'extracted_model'
MODEL_DIR = None  

try:
    # Check if we need to extract
    if not os.path.exists(EXTRACT_DIR):
        st.info(f'Starting extraction from "{ZIP_FILE_PATH}".')
        
        # Check if the source zip file is present
        if not os.path.exists(ZIP_FILE_PATH):
             st.error(f'Critical Error: Zip file "{ZIP_FILE_PATH}" not found. Please ensure it is uploaded.')
             st.stop()

        # Perform the extraction
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            zip_ref.extractall(path=EXTRACT_DIR)
        st.success('Model successfully extracted!')
    else:
        st.info('Model directory exists. Skipping extraction.')
    # Searches recursively for huggingface model & primary files.
    for root, dirs, files in os.walk(EXTRACT_DIR):
        if 'config.json' in files and 'pytorch_model.bin' in files:
            MODEL_DIR = root
            break
    
    # If not found using pytorch_model.bin, try model.safetensors
    if MODEL_DIR is None:
        for root, dirs, files in os.walk(EXTRACT_DIR):
            if 'config.json' in files:
                MODEL_DIR = root
                break
    
    if MODEL_DIR is None:
        st.error(f'Could not find model files in extracted directory. Please check the zip file contents.')
        st.stop()
    st.info(f'Model directory found at: {MODEL_DIR}')
        
except FileNotFoundError:
    st.error(f'File not found! Make sure {ZIP_FILE_PATH} is present.')
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during extraction: {e}")
    st.exception(e)

# --- 2. Load Model and Tokenizer ---
try:
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    st.success('✅ Model and tokenizer loaded successfully!')
except Exception as e:
    st.error(f"Failed to load model or tokenizer from directory '{MODEL_DIR}'. Ensure extracted files are correct.")
    st.exception(e)
    st.stop()

# --- NLTK Setup ---
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# --- Preprocessing and Prediction Functions ---

def preprocess_text(text):
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Configuration
max_length = 128

# Create ID to label mapping - using fixed mapping.
id_to_label = {
    0: 'get weather',
    1: 'search creative work',
    2: 'search screening event',
    3: 'add to playlist',
    4: 'book restaurant',
    5: 'rate book',
    6: 'play music',
}

def predict_intent(text):
    text = preprocess_text(text)
    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        truncation=True, 
        padding='max_length', 
        max_length=max_length
    )
    
    with torch.no_grad():
        outputs = model(**inputs) # here, inputs contain input_ids and attention_mask.
        logits = outputs.logits # logits are un-normalize scores for each class.
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return predicted_class_id

# --- Streamlit UI and Logic ---

st.title("🎯 Intent Classification App")
text = st.text_area("Enter a sentence to classify its intent:", height=150, placeholder="e.g., What's the weather like today?")

if st.button('Predict Intent', type="primary", use_container_width=True):
    if text.strip():
        with st.spinner('Analyzing your text...'):
            # Get prediction
            predicted_id = predict_intent(text)
            intent = id_to_label.get(predicted_id, 'Unknown Intent') # Here, Unknown is showing default.
            
            # Display result in a nice format
            st.success("✅ Classification Complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Predicted Intent", intent)
            with col2:
                st.metric("Intent ID", predicted_id)
    else:
        st.warning("⚠️ Please enter a sentence to classify.")

# Optional: Showing available intents
with st.expander("📋 View Available Intents"):
    st.write("The model can classify the following intents:")
    for idx, label in sorted(id_to_label.items()):
        st.write(f"**{idx}.** {label}")

# Add footer
st.markdown("---")
st.caption("Built with Streamlit")