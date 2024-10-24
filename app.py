import streamlit as st
from pathlib import Path
from model import Transformer
from tokenizers import Tokenizer
import torch
import time

device = "cpu"

@st.cache_resource
def load_transformer():
    transformer = Transformer(dmodel=512, vocab_size=15797, device=device, hidden_size=2048, N=8, h=8).to(device)
    transformer.load_state_dict(torch.load("weights/1824_steps.pth", map_location=torch.device('cpu')))
    tokenizer_path = Path('./tokenizer/')
    tokenizer = Tokenizer.from_file(str(tokenizer_path / 'tokenizer.json'))
    return transformer, tokenizer

def generate_text(model, tokenizer, input_text, num_tokens=50, temperature=0.6):
    global device
    generated_tokens = tokenizer.encode(input_text).ids
    block_size = 128  # Define your block size
    yield input_text
    for _ in range(num_tokens):
        input_tensor = torch.tensor(generated_tokens[-block_size:]).unsqueeze(0).to(device)  # Keep only the last block_size tokens
        with torch.no_grad():
            predictions = model(input_tensor)
        
        # Get the last prediction and apply softmax
        next_token_logits = predictions[:, -1, :]  # Get the last token logits
        next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
        
        # Sample a token from the predicted probabilities
        next_token = torch.multinomial(next_token_probs, num_samples=1).item()
        
        # Append the predicted token to the sequence
        generated_tokens.append(next_token)
        
        yield tokenizer.decode([next_token])

# Streamlit app layout
st.write("""
# Harry Potter Text Generator

For more control over the outputs, check the options in the sidebar.
""")

st.sidebar.image("images/harryPT.webp",width=330)

st.sidebar.write(
    """
This is a simple replicate of the `GPT-2` Architecture, It's a decoder based text 
generator. Our decoder architecture was taken from the ```Attention is all you Need paper``` from google and with some minor modifications.
It was trained on aroud 1.9k steps and on all the Harry Potter books dataset.
    """
)
st.sidebar.divider()
st.divider()
# Load the model and tokenizer only once
if 'model' not in st.session_state or 'tokenizer' not in st.session_state:
    st.session_state.model, st.session_state.tokenizer = load_transformer()

# Create a form for text input
with st.form(key='text_input_form', border=False):
    temperature = st.sidebar.slider(
    "Select randomness (0.1 = Least random, 1.0 = most random):",
    0.1,
    1.0
    )

    num_tokens = st.sidebar.slider("How many tokkens do you want to generate ?", 50, 300, 100)
    text_input = st.text_input("Enter some text to be completed 👇")
    _, mid, _ = st.columns([3,3,1])
    submit_button = mid.form_submit_button(label='Generate')
    
st.sidebar.markdown(
    """
    <div style='text-align: center;'>
        <hr style='width:100%;'>
        <p style='color:gray; font-size:13px' >Built with love using Pytorch and Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Generate text only when the form is submitted
if submit_button:
    if text_input:
        with st.spinner("Generating .."):
            time.sleep(1)
            st.divider()
            output = generate_text(st.session_state.model, st.session_state.tokenizer, text_input, num_tokens=num_tokens, temperature=temperature)
            st.write_stream(output)
            st.success("Done")
    else:
        st.warning("Please enter some text to complete.")
