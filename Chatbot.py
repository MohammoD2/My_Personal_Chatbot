import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from safetensors import safe_open
import os

# Set page configuration
st.set_page_config(page_title="Personal Chatbot", page_icon="🤖", layout="centered")

# Get the Google Drive model key from Streamlit secrets
MODEL_KEY = st.secrets["keys"]["MODEL_KEY"]
MODEL_PATH = "model.safetensors"

# Download model from Google Drive if not locally available
def download_model():
    import gdown
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(f"https://drive.google.com/uc?id={MODEL_KEY}", MODEL_PATH, quiet=False)
            st.success("Model downloaded successfully.")

# Load the model and tokenizer from safe tensors, with caching for speed
@st.cache_resource(show_spinner=False)
def load_model():
    download_model()
    # Load the base GPT2 model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    # Load weights from the safetensors file
    with safe_open(MODEL_PATH, framework="pt") as f:
        for name, param in model.named_parameters():
            if f.has_tensor(name):  # Check if the tensor exists in the file
                param.data = f.get_tensor(name)
            else:
                st.warning(f"Missing tensor: {name}")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set EOS token as pad token
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

# Sidebar: Chat settings
with st.sidebar:
    st.title('Chat Settings')
    def clear_chat_history():
        st.session_state.messages = [{
            "role": "assistant", 
            "content": "Assalamu alaikum 🍁, I'm your personal chatbot, here to assist you! 😊"
        }]
    st.button('Clear Chat History', on_click=clear_chat_history)

# Initialize chat history if it does not exist yet
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Assalamu alaikum 🍁, I'm your personal chatbot, here to assist you! 😊"
    }]

# Display chat messages in order
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Function to generate a response using the loaded model
def generate_response(input_text):
    input_prompt = f"User: {input_text} <|endoftext|> Response:"
    input_ids = tokenizer.encode(input_prompt, return_tensors='pt')
    # Create attention mask manually if needed
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

# User input prompt and response generation
if prompt := st.chat_input("Type a message"):
    # Append user message to chat history and show it immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    # Ensure we generate a response only when the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                response = generate_response(prompt)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})


