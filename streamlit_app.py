
import streamlit as st
import openai
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle # For loading chunks and embeddings

# --- Suppress TensorFlow/Keras Warnings ---
# These environment variables help to silence verbose TensorFlow logging and specific warnings.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress oneDNN custom operations warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show warnings and errors (1=info, 2=warning, 3=error)

# --- 1. Configure your LLM API (GroqCloud) ---
# Your provided GroqCloud API credentials
GROQ_API_BASE = "https://api.groq.com/openai/v1"
# IMPORTANT: Retrieve API key from environment variable for security and deployment
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # THIS LINE WAS MODIFIED TO REMOVE THE HARDCODED KEY
# GROQ_API_KEY = "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"
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Set API base and key for openai==0.28.0 (compatible with Groq's OpenAI-like API)
openai.api_base = GROQ_API_BASE
openai.api_key = GROQ_API_KEY

# --- 2. Load RAG Components (Data, Model, Index, Chunks) ---
# These components will be loaded only once when the Streamlit app starts
# We use st.cache_resource to cache heavy objects like models and FAISS indexes
@st.cache_resource
def load_rag_components():
    """
    Loads the Sentence Transformer model, FAISS index, and data chunks.
    This function is cached to run only once when the Streamlit app starts.
    """
    model = None
    index = None
    chunks = []

    # Load the Sentence Transformer model
    st.info("Loading Sentence Transformer model... This may take a moment.")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        st.success("Sentence Transformer model loaded.")
    except Exception as e:
        st.error(f"Error loading Sentence Transformer model: {e}")
        st.error("Please ensure 'sentence-transformers' is installed: `pip install sentence-transformers`")
        return None, None, None

    # Load the FAISS index
    index_file_name = 'charging_data_index.faiss'
    if os.path.exists(index_file_name):
        try:
            index = faiss.read_index(index_file_name)
            st.success(f"FAISS index loaded with {index.ntotal} vectors.")
        except Exception as e:
            st.error(f"Error loading FAISS index from '{index_file_name}': {e}. Please ensure it's not corrupted or regenerate it.")
            return model, None, None
    else:
        st.error(f"Error: FAISS index file '{index_file_name}' not found.")
        st.error("Please run the 'Embedding Generation and FAISS Indexing' step first to create this file.")
        return model, None, None

    # Load the chunks
    chunks_file_name = 'all_chunks.pkl'
    if os.path.exists(chunks_file_name):
        try:
            with open(chunks_file_name, 'rb') as f:
                chunks = pickle.load(f)
            st.success(f"Loaded {len(chunks)} text chunks.")
        except Exception as e:
            st.error(f"Error loading chunks from '{chunks_file_name}': {e}. Please ensure it's not corrupted or regenerate it.")
            return model, index, []
    else:
        st.error(f"Error: Chunks file '{chunks_file_name}' not found.")
        st.error("Please run the 'Data Loading and Chunk Generation' step first to create this file.")
        return model, index, []

    return model, index, chunks

# Load components globally for Streamlit (cached automatically)
rag_model, rag_index, all_chunks = load_rag_components()

# --- 3. Define the Retrieval Function ---
def retrieve_top_k_chunks(question, model, index, all_chunks, top_k=3): # top_k set to 3 for brevity
    """
    Retrieves the top-k most relevant text chunks from the FAISS index
    based on the semantic similarity to the user's question.
    """
    if model is None or index is None or not all_chunks:
        # These errors are already handled by load_rag_components
        return []

    try:
        question_embedding = model.encode([question]).astype('float32')
        if question_embedding.shape[1] != index.d:
            st.warning(f"Warning: Question embedding dimension ({question_embedding.shape[1]}) does not match index dimension ({index.d}). Retrieval might be inaccurate.")
            return []

        distances, indices = index.search(question_embedding, top_k)
        results = [all_chunks[i] for i in indices[0] if i < len(all_chunks)]
        return results
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return []

# --- 4. Define the LLM answer generation function (using GroqCloud) ---
def generate_answer(question, context_text):
    """
    Generates an answer to the user's question using GroqCloud,
    based on the provided context. Includes robust prompt engineering.
    """
    prompt = f"""You are an expert assistant for Electric Vehicle (EV) charging data in India. Your goal is to provide **precise, accurate, and specific answers** to user questions based *only* on the provided context.

    **Instructions for your answer:**
    1.  **Direct Answers:** If the context directly provides the answer, state it clearly with numerical values and units.
    2.  **Inference/Approximation:** If an exact numerical match is not found but similar data is present (e.g., asking for 40kWh, but context has 38kWh or 42kWh), provide an *approximate* answer or a range based on the closest relevant data. Clearly state that it's an approximation or based on similar data.
    3.  **Summarize Relevant Data:** If a direct numerical answer isn't possible but the context contains relevant information, summarize the key data points that are related to the question.
    4.  **Explain Limitations:** If the question cannot be answered even approximately from the given context, clearly state that the information is not available in the provided data and briefly explain *why* (e.g., "The data does not contain information about X battery capacity" or "The context does not provide average values for this specific combination").
    5.  **Prioritize Aggregated Data:** If both individual session data and aggregated summaries are available for a question (e.g., "average charging duration for X charger type"), prioritize using the aggregated summary.
    6.  **Maintain Professional Tone:** Be helpful, informative, and concise.
    7.  **Do NOT make up information.** Stick strictly to the provided context.

    Context:
    {context_text}

    Question:
    {question}

    Answer:"""

    try:
        # For openai==0.28.0, use openai.ChatCompletion.create directly
        response = openai.ChatCompletion.create(
            model=GROQ_MODEL_NAME, # Use the specified Groq model name
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500, # Groq models are generally generous, but adjust if you hit limits
            temperature=0.2, # Lower temperature for more factual answers
        )
        answer = response['choices'][0]['message']['content'] # Access content using dictionary keys
        return answer
    except openai.error.InvalidRequestError as e:
        st.error(f"API Request Error: {e}. This might be due to prompt length, invalid parameters, or issues with the Groq model.")
        return "Sorry, I encountered an API error while processing your request. Please try rephrasing."
    except openai.error.AuthenticationError as e:
        st.error(f"Authentication Error: {e}. Please check your Groq API key in the code.")
        return "Authentication failed. Please check your API key."
    except openai.error.APIConnectionError as e:
        st.error(f"API Connection Error: {e}. Please check your internet connection or GroqCloud status.")
        return "Could not connect to the API. Please check your internet connection."
    except openai.error.RateLimitError as e:
        st.warning(f"Rate Limit Exceeded: {e}. Please wait a moment and try again, or check your GroqCloud plan.")
        return "You've sent too many requests too quickly. Please wait a moment."
    except openai.error.OpenAIError as e:
        st.error(f"GroqCloud API Error: {e}. There was an issue with the API service.")
        return "An error occurred with the GroqCloud API. Please try again later."
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return "An unexpected error occurred. Please try again."

# --- 5. Streamlit UI Setup ---
st.set_page_config(page_title="EV Charging Chatbot", page_icon="⚡️", layout="centered")

# Custom CSS for a dark theme and better visibility
st.markdown("""
<style>
    /* Overall app background - Dark theme */
    .reportview-container {
        background: #282a36; /* Dracula Theme background */
        color: #f8f8f2; /* Light text for dark background */
    }
    .main .block-container {
        max-width: 768px;
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background-color: #282a36; /* Consistent dark background */
        color: #f8f8f2;
    }

    /* Streamlit widgets text color */
    .stMarkdown, .stText, .stJson, .stDataFrame {
        color: #f8f8f2;
    }

    /* Input text box styling - Dark theme */
    .stTextInput > div > div > input {
        border-radius: 0.75rem;
        border: 1px solid #44475a; /* Darker border */
        padding: 0.75rem 1rem;
        font-size: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2); /* Stronger shadow for contrast */
        color: #f8f8f2; /* Light text */
        background-color: #333444; /* Slightly lighter dark background for input */
    }
    .stTextInput > div > div > input::placeholder {
        color: #6272a4; /* Lighter placeholder text */
    }

    /* Button styling - Adapted for dark theme */
    .stButton > button {
        background: linear-gradient(90deg, #50fa7b 0%, #bd93f9 100%); /* Green-purple gradient for vibrancy */
        color: #282a36; /* Dark text on vibrant button */
        padding: 0.75rem 1.5rem;
        border-radius: 0.75rem;
        font-weight: 600; /* Bolder font for buttons */
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3); /* Pronounced shadow */
        border: none;
    }
    .stButton > button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.4);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Chat bubbles - Dark theme */
    .chat-message {
        padding: 10px 15px;
        border-radius: 1rem;
        margin-bottom: 10px;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        color: #f8f8f2; /* Light text */
    }
    .user-message {
        background-color: #44475a; /* Darker gray-blue for user */
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 0.25rem;
    }
    .bot-message {
        background-color: #3f4c4a; /* Darker green-gray for bot */
        align-self: flex-start;
        margin-right: auto;
        border-bottom-left-radius: 0.25rem;
    }
    .loading-indicator {
        background-color: #333444; /* Darker background */
        padding: 10px 15px;
        border-radius: 1rem;
        margin-bottom: 10px;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        color: #bd93f9; /* Light purple for loading text */
        font-style: italic;
    }

    /* Header styling - Dark theme */
    h1 {
        color: #50fa7b; /* Vibrant green for header */
        text-align: center;
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }

    /* Adjusting chat container for better layout - Dark theme */
    .st-emotion-cache-z5f0fr { /* This is a dynamic class for the main content area */
        display: flex;
        flex-direction: column;
        min-height: 70vh;
        justify-content: space-between;
        background-color: #2e303e; /* Slightly lighter dark for main content area */
        border-radius: 1.5rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.1);
    }
    /* Ensure chat messages take full width within their container */
    .st-emotion-cache-1c7y2kl { /* This is a dynamic class for the chat message container */
        flex-grow: 1;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1.5rem;
    }
    /* Adjusting the input container to ensure it's visible - Dark theme */
    .st-emotion-cache-13ln4gm { /* This is a dynamic class for the chat input container */
        background-color: #282a36; /* Matches app background */
        padding: 1.5rem;
        border-top: 1px solid #44475a; /* Darker border */
        border-bottom-left-radius: 1.5rem;
        border-bottom-right-radius: 1.5rem;
    }
    .retrieved-context-box {
        background-color: #38424a; /* Darker blue-gray for context box */
        border-left: 5px solid #8be9fd; /* Vibrant light blue accent border */
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        color: #f8f8f2; /* Light text */
        max-height: 200px;
        overflow-y: auto;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
    }
    .retrieved-context-box h5 {
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #f1fa8c; /* Vibrant yellow for context header */
    }

    /* Adjust Streamlit specific elements for dark theme */
    .st-dg { /* DataFrame */
        color: #f8f8f2;
    }
    .st-bd { /* Border around markdown/text */
        border-color: #44475a;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        color: #f8f8f2; /* Adjust expander text color */
    }
    div[data-testid="stStatusWidget"] {
        background-color: #44475a; /* Info/success/error banners */
        color: #f8f8f2;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡️ EV Charging Chatbot")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! I'm your EV Charging Chatbot. I can answer questions about EV charging times, battery capacities, and more, based on Indian EV data. How can I assist you today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-message user-message">{message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot-message">{message["content"]}</div>', unsafe_allow_html=True)
    
    # Conditionally display retrieved context if it exists for bot messages
    if message["role"] == "bot" and "retrieved_context" in message and message["retrieved_context"]:
        with st.expander("See Retrieved Context"):
            # Use st.code for better display of multi-line context, or just st.markdown with <pre>
            st.markdown(f'<div class="retrieved-context-box"><h5>Context Used:</h5><pre>{message["retrieved_context"]}</pre></div>', unsafe_allow_html=True)


# Input for user question
user_question = st.chat_input("Ask me about EV charging in India...")

if user_question:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_question})
    # Display the user message immediately after adding to history
    st.markdown(f'<div class="chat-message user-message">{user_question}</div>', unsafe_allow_html=True)

    # Display loading indicator
    # Use st.empty() to create a placeholder that can be updated or cleared
    loading_placeholder = st.empty()
    loading_placeholder.markdown('<div class="loading-indicator">Chatbot is thinking...</div>', unsafe_allow_html=True)

    bot_response_content = ""
    retrieved_context_for_display = "N/A - Component loading issues."

    # --- Perform RAG Retrieval and LLM Generation ---
    if rag_model is None or rag_index is None or not all_chunks:
        bot_response_content = "Error: RAG components failed to load. Please check console for details and ensure all prerequisites are met (data, index, chunks files)."
        retrieved_context_for_display = "N/A - RAG components not loaded."
    else:
        # Retrieve top K chunks based on the user's question
        # top_k is set in retrieve_top_k_chunks, but can be overridden here
        top_k_chunks = retrieve_top_k_chunks(user_question, rag_model, rag_index, all_chunks, top_k=3)
        context_text = "\n\n".join(top_k_chunks)
        retrieved_context_for_display = context_text # Store for display

        # Get bot response from LLM using the retrieved context
        bot_response_content = generate_answer(user_question, context_text)

    # Remove loading indicator
    loading_placeholder.empty()

    # Add bot response to chat history, including retrieved context for optional display
    st.session_state.messages.append({
        "role": "bot",
        "content": bot_response_content,
        "retrieved_context": retrieved_context_for_display
    })
    # Display the bot response after it's generated
    st.markdown(f'<div class="chat-message bot-message">{bot_response_content}</div>', unsafe_allow_html=True)

    # Display retrieved context in an expander
    if retrieved_context_for_display and retrieved_context_for_display != "N/A - Component loading issues.":
        with st.expander("See Retrieved Context"):
            st.markdown(f'<div class="retrieved-context-box"><h5>Context Used:</h5><pre>{retrieved_context_for_display}</pre></div>', unsafe_allow_html=True)


# Add a "Clear Chat" button at the bottom of the page
st.markdown("---") # Separator
if st.button("Clear Chat History"):
    st.session_state.messages = [
        {"role": "bot", "content": "Hello! I'm your EV Charging Chatbot. I can answer questions about EV charging times, battery capacities, and more, based on Indian EV data. How can I assist you today?"}
    ]
    st.rerun() # Rerun the app to clear the chat display



