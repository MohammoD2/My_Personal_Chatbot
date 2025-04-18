# Environment fixes FIRST
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Now other imports
import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from safetensors import safe_open
import gdown

# Rest of your original code...

# --- Page Configuration ---
st.set_page_config(page_title="Personal Chatbot", page_icon="🤖", layout="centered")

# --- Model Management ---
MODEL_FOLDER_ID = st.secrets["keys"]["MODEL_FOLDER_ID"]
MODEL_DIR = "model_files"

@st.cache_resource
def setup_model():
    # Download model files from Google Drive
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)
        url = f"https://drive.google.com/drive/folders/{MODEL_FOLDER_ID}"
        gdown.download_folder(url, output=MODEL_DIR, quiet=False)
    
    # Load tokenizer and base model
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
    
    # Load fine-tuned weights
    with safe_open(os.path.join(MODEL_DIR, "model.safetensors"), framework="pt") as f:
        for name, param in model.named_parameters():
            if name in f.keys():
                param.data = f.get_tensor(name)
    
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# --- Initialize Model ---
model, tokenizer = setup_model()

# --- Chat Interface ---
with st.sidebar:
    st.title('Chat Settings')
    def clear_chat_history():
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Assalamu alaikum 🍁, I'm your personal chatbot, here to assist you! 😊"
        }]
    st.button('Clear Chat History', on_click=clear_chat_history)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Assalamu alaikum 🍁, I'm your personal chatbot, here to assist you! 😊"
    }]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Response generation function
def generate_response(input_text):
    input_prompt = f"User: {input_text} <|endoftext|> Response:"
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
        repetition_penalty=1.1
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Response:" in response:
        response = response.split("Response:")[1].strip()
    return response

# Chat input handling
if prompt := st.chat_input("Type a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                response = generate_response(prompt)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


