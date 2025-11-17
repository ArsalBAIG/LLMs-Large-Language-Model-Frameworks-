import streamlit as st
from transformers import pipeline
import torch

# --- Model Loading with Caching ---
@st.cache_resource
def load_summarizer_pipeline(model_name: str):
    """Load the summarization model with proper configuration."""
    try:
        # Check if CUDA is available
        device = 0 if torch.cuda.is_available() else -1
        
        summarizer = pipeline(
            "summarization",
            model=model_name,
            device=device,
            torch_dtype=torch.float32,  # Explicitly set dtype
            model_kwargs={"low_cpu_mem_usage": True}  # More efficient loading
        )
        return summarizer, None
    except Exception as e:
        return None, str(e)

# --- Streamlit App UI ---

st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("📄 LLM Text Summarizer")
st.markdown("Enter a paragraph below to generate a summary.")

# Model selection
MODEL_NAME = "facebook/bart-large-cnn"

# Load the model once using the cached function
with st.spinner("Loading model..."):
    model_pipeline, error = load_summarizer_pipeline(MODEL_NAME)

if model_pipeline is None:
    st.error(f"Failed to load the model pipeline: {error}")
else:
    st.success("✅ Model loaded successfully!")

# Text input
input_text = st.text_area("Enter Text to Summarize:", height=200)

if st.button("✨ Generate Summary"):
    if model_pipeline is None:
        st.error("Cannot generate summary because the model failed to load.")
    elif not input_text.strip():
        st.warning("⚠️ Please enter some text to summarize.")
    elif len(input_text.split()) < 45:
        st.warning("⚠️ Text is too short. Please provide at least 45 words for better results.")
    else:
        with st.spinner("Generating summary..."):
            try:
                # Calculate input length
                input_word_count = len(input_text.split())
                
                # increase the input_word_count by 30% to estimate tokens
                input_tokens = int(input_word_count * 1.3) 
                
                # Set max_length to 50% of input tokens (ensures shorter output)
                max_length = max(10, int(input_tokens * 0.5))
                # Set min_length to 30% of input tokens (ensures summary is not too short)
                min_length = max(5, int(input_tokens * 0.3))
                
                # Generate summary
                result = model_pipeline(
                    input_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True,  # Handle long texts
                    no_repeat_ngram_size=3  # Prevent repetition
                )
                
                summary = result[0]['summary_text']
                summary_word_count = len(summary.split())
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📝 Original Text")
                    st.write(f"**Word count:** {len(input_text.split())}")
                    st.text_area("Original", input_text, height=200)
                
                with col2:
                    st.subheader("✨ Summary")
                    st.write(f"**Word count:** {len(summary.split())}")
                    st.text_area('Summary', summary, height=200)
            except Exception as e:
                st.error(f"❌ An error occurred during summarization: {str(e)}")

st.markdown("---")