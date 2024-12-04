import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from safetensors import safe_open
import os

# App title
st.set_page_config(page_title="Personal Chatbot", page_icon="🤖", layout="centered")

# Access MODEL_KEY from Streamlit secrets
MODEL_KEY = st.secrets["keys"]["MODEL_KEY"]
MODEL_PATH = "model.safetensors"

# Download model from Google Drive if it doesn't exist locally
def download_model():
    import gdown
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(f"https://drive.google.com/uc?id={MODEL_KEY}", MODEL_PATH, quiet=False)

# Load the model and tokenizer
@st.cache_resource
def load_model():
    download_model()
    model = GPT2LMHeadModel.from_pretrained('gpt2')  # Use the base model config
    with safe_open(MODEL_PATH, framework="pt") as f:
        for name, param in model.named_parameters():
            param.data = f.get_tensor(name)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model()

# Sidebar for Clear Chat History
with st.sidebar:
    st.title('Chat Settings')
    def clear_chat_history():
        st.session_state.messages = [{"role": "assistant", "content": "Assalamu alaikum 🍁, I'm your personal chatbot, here to assist you! 😊"}]
    st.button('Clear Chat History', on_click=clear_chat_history)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Assalamu alaikum 🍁, I'm your personal chatbot, here to assist you! 😊"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to generate response
def generate_response(input_text):
    input_prompt = f"User: {input_text} <|endoftext|> Response:"
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Response:" in response:
        response = response.split("Response:")[1].strip()
    return response

# User input and chatbot response generation
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

