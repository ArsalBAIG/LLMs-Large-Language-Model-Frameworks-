import os
import zipfile 
import streamlit as st
import torch
import tempfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def extract_and_load_model(zip_path, extract_dir=None):
    if extract_dir is None:
        extract_dir = tempfile.mkdtemp() # Create a temp directory with unique name.
        st.info(f'Extracting model to: {extract_dir}')

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)    
    st.success("Extraction complete!")

    model_dir = extract_dir
    for root, dirs, files in os.walk(extract_dir):
        if 'config.json' in files:
            model_dir = root
            st.info(f'Found model config at: {model_dir}')
            break
    st.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return model, tokenizer, extract_dir


def generate_recipe(prompt, model, tokenizer, max_output_length= 128):
    input_text = tokenizer(
        prompt,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )

    input_ids = input_text['input_ids'].to(torch.device('cpu'))
    attention_mask = input_text['attention_mask'].to(torch.device('cpu'))

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_output_length,
        num_beams=5,
        early_stopping=True
    )

    generate_recipe = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generate_recipe


if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.tokenizer = None

if st.session_state.model is None:
    zip_path = st.text_input("Enter path to fine_tuned_model.zip:", "fine_tuned_model.zip")
    if st.button('Load Model from Zip'):
        if os.path.exists(zip_path):
            model, tokenizer, extract_dir = extract_and_load_model(zip_path)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer 
            st.success("Model and tokenizer loaded successfully!")
            st.rerun()
        else:
            st.error("Zip file not found. Please check the path and try again.")
    st.stop()

st.title("Recipe Generation App")
prompt = st.text_area("Enter your recipe prompt here:", height= 100)

if st.button('Generate Recipe'):
    if prompt.strip() == "":
        st.warning("Please enter a valid recipe prompt.")
    else:
        with st.spinner("Generating recipe..."):
            generated_recipe = generate_recipe(prompt, st.session_state.model, st.session_state.tokenizer)
        st.subheader("Generated Recipe:")
        st.text(generated_recipe)
