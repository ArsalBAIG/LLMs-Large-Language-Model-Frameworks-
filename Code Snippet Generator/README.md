# Code Snippet Generation with Fine-Tuned T5

A natural language to code generation system built by fine-tuning Google's `t5-small` model on a 10,000-sample dataset. Users describe what they want in plain English, and the model returns a corresponding code snippet. An interactive Streamlit web app makes the model accessible without any coding knowledge.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Training Results](#training-results)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Train the Model (Notebook)](#1-train-the-model-notebook)
  - [2. Run the Streamlit App](#2-run-the-streamlit-app)
  - [3. Generate Code](#3-generate-code)
- [Supported Languages](#supported-languages)
- [Example Queries](#example-queries)

---

## Overview

This project fine-tunes a T5 (Text-to-Text Transfer Transformer) model on a labeled dataset of natural language queries paired with code snippets across four programming languages. The trained model is then served through a Streamlit web application that allows users to:

- Load the fine-tuned model from a `.zip` file
- Enter a plain-English description of what they want coded
- Receive a generated code snippet instantly

---

## Project Structure

```
├── Code_Snippet_Generation.ipynb   # End-to-end training notebook (Colab)
├── app.py                          # Streamlit inference application
├── fine_tuned_model/               # Saved model directory (generated after training)
├── fine_tuned_model.zip            # Zipped model for portability
└── README.md
```

---

## Dataset

| Property        | Detail                                      |
|-----------------|---------------------------------------------|
| File            | `code_dataset_10000.csv`                    |
| Total Samples   | 10,000                                      |
| Columns         | `Query`, `Code_Snippet`, `Language`         |
| Missing Values  | None (fully clean)                          |

**Language distribution:**

| Language   | Samples |
|------------|---------|
| C++        | 2,525   |
| JavaScript | 2,522   |
| Python     | 2,484   |
| Java       | 2,469   |

Sample rows from the dataset:

| Query                                         | Code Snippet                                      | Language   |
|-----------------------------------------------|---------------------------------------------------|------------|
| How to print in JavaScript?                   | `console.log('Hello');`                           | JavaScript |
| How to create a vector in C++?                | `std::vector<int> v = {1,2,3};`                   | C++        |
| How to create a list comprehension in Python? | `[x*2 for x in range(10)]`                        | Python     |
| How to fetch API data in JavaScript?          | `fetch(url).then(r=>r.json())...`                 | JavaScript |

**Preprocessing steps applied:**
- All `Query`, `Code_Snippet`, and `Language` columns lowercased for consistency
- Input sequences prefixed with `"Generate code: "` before tokenization

---

## Model Architecture

The base model is **`t5-small`** from Hugging Face, a compact Seq2Seq transformer:

| Component       | Detail                        |
|-----------------|-------------------------------|
| Architecture    | T5ForConditionalGeneration    |
| Embedding dim   | 512                           |
| FFN hidden dim  | 2048                          |
| Encoder layers  | 6 (1 + 5 shared)              |
| Decoder layers  | 6 (1 + 5 shared)              |
| Attention heads | 8                             |
| Vocab size      | 32,128                        |
| Dropout         | 0.1                           |
| Activation      | ReLU                          |
| Tokenizer       | T5Tokenizer (SentencePiece)   |

---

## Training Details

Training was conducted on **Google Colab** with GPU (CUDA) acceleration using Hugging Face's `Trainer` API.

| Hyperparameter                 | Value            |
|-------------------------------|------------------|
| Epochs                         | 3                |
| Learning rate                  | 5e-5             |
| Train batch size (per device)  | 1                |
| Eval batch size (per device)   | 4                |
| Gradient accumulation steps    | 4 (effective batch size = 4) |
| Weight decay                   | 0.01             |
| Warmup steps                   | 500              |
| Evaluation strategy            | Per epoch        |
| Save strategy                  | Per epoch        |
| Mixed precision (fp16)         | ✅ Enabled        |
| Gradient checkpointing         | ✅ Enabled        |
| Best model loading at end      | ✅ Enabled        |
| Logging steps                  | 10               |
| Output directory               | `./results`      |
| Logging directory              | `./logs`         |

A `DataCollatorForSeq2Seq` was used to handle dynamic padding during training.

---

## Training Results

| Metric                      | Value      |
|-----------------------------|------------|
| Final Training Loss         | **0.3229** |
| Total Global Steps          | 7,500      |
| Total Training Time         | ~56 min (3,374 sec) |
| Train Samples/sec           | 8.892      |
| Train Steps/sec             | 2.223      |
| Total FLOPs                 | 1.67 × 10¹⁴ |
| Epochs Completed            | 3          |

---

## Installation

Clone the repository and install dependencies:

```bash
pip install streamlit transformers torch pandas
```

> **Note:** Training was done on Google Colab with GPU. For local training, ensure CUDA is available. The Streamlit app runs on CPU by default; a GPU will significantly speed up inference.

---

## Usage

### 1. Train the Model (Notebook)

Open `Code_Snippet_Generation.ipynb` in **Google Colab** and run all cells in order:

1. **Import libraries & load dataset** — loads `code_dataset_10000.csv` into a Pandas DataFrame.
2. **EDA** — checks for null values and reviews language distribution.
3. **Preprocessing** — lowercases all text; prepends `"Generate code: "` to each query.
4. **Tokenization** — encodes inputs and targets with `T5Tokenizer` (truncation + padding).
5. **Dataset class** — wraps encodings in a PyTorch `Dataset` (`CodeDataset`).
6. **Training** — runs 3 epochs with the `Trainer` API using the hyperparameters above.
7. **Save model** — saves model and tokenizer to `./fine_tuned_model/`.
8. **Zip for export** — archives the model directory into `fine_tuned_model.zip` for use in the app.

```python
# The model is saved and zipped automatically at the end of the notebook
import shutil
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')
shutil.make_archive('fine_tuned_model', 'zip', './fine_tuned_model')
```

Download `fine_tuned_model.zip` from Colab to your local machine.

---

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

On first launch, the app will prompt you for the path to `fine_tuned_model.zip`. Once loaded, the model is stored in Streamlit's `session_state` — it will not reload between interactions.

**Model loading flow:**
1. Enter the path to `fine_tuned_model.zip` (default: `fine_tuned_model.zip`)
2. Click **"Load Model from Zip"**
3. The app extracts the zip to a temp directory, scans for `config.json` to locate the model, and loads it with `AutoModelForSeq2SeqLM` and `AutoTokenizer`
4. The app reloads and the code generation interface appears

---

### 3. Generate Code

Once the model is loaded:

1. Type a natural language description in the text area (specify the language for best results)
2. Click **"Generate Code"**
3. The generated snippet appears in a formatted code block

**Inference settings used in the app:**

| Parameter       | Value           |
|-----------------|-----------------|
| Max input length | 128 tokens     |
| Max output length | 128 tokens    |
| Decoding strategy | Beam search   |
| Number of beams | 5               |
| Early stopping  | ✅ Enabled       |

---

## Supported Languages

| Language   | Suggested Prompt Phrasing          |
|------------|------------------------------------|
| Python     | `"...in Python"`                   |
| JavaScript | `"...in JavaScript"`               |
| Java       | `"...in Java"`                     |
| C++        | `"...in C++"`                      |

---

## Demo Video

https://github.com/user-attachments/assets/67b45176-c947-467a-8b16-988fe88b105b

---
