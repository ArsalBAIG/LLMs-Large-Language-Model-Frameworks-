import os
import zipfile
import torch
import tempfile
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def extract_and_load_model(zip_path, extract_dir=None):
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp() # Create a temp directory with unique name.
    st.info(f"Extracting model to: {extract_dir}")
    
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)    
    st.success("Extraction complete!")
    
    # Find the model directory (handle nested folders)
    model_dir = extract_dir
    for root, dirs, files in os.walk(extract_dir):
        if 'config.json' in files: # Essential for loading the model.
            model_dir = root
            st.info(f"Found model config at: {model_dir}")
            break
    
    # Load model and tokenizer
    st.info("Loading model and tokenizer...")
    loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    loaded_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return loaded_model, loaded_tokenizer, extract_dir


def generate_code(query, model, tokenizer):
    query = query.lower().strip()
    input_text = 'Generate Code: ' + query
    input_ids = tokenizer.encode(
        input_text, 
        padding='max_length', 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    )
    outputs = model.generate(
        input_ids, 
        max_length=128, 
        num_beams=5, 
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Initialize session state for model and tokenizer
if 'model' not in st.session_state: # Checks if 'model' key exists in streamlit session.
    st.session_state.model = None
    st.session_state.tokenizer = None

# Load model section
if st.session_state.model is None:
    st.warning("⚠️ Model not loaded. Please load the model first.")
    zip_path = st.text_input("Enter path to fine_tuned_model.zip:", "fine_tuned_model.zip")
    if st.button("Load Model from Zip"):
        if os.path.exists(zip_path):
            model, tokenizer, extract_dir = extract_and_load_model(zip_path)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.success("✅ Model loaded! You can now generate code.")
            st.rerun()
        else:
            st.error("File not found!")
    st.stop()

# Streamlit UI for code generation
st.title("Code Generation with T5 Model")
user_query = st.text_area("Enter your code description within C++, JS, Java, PY:")

if st.button("Generate Code"):
    if user_query:
        with st.spinner('Generating code...'):
            generated_code = generate_code(user_query, st.session_state.model, st.session_state.tokenizer)
            st.subheader("Generated Code:")
            st.code(generated_code)
    else:
        st.warning("Please enter a code description to generate code.")