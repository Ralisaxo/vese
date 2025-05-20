import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import ast
import os # For API key if set as environment variable locally

# --- Configuration ---
# Ensure your CSV file is named this, or change the name here.
# This file should be in the SAME FOLDER as app.py in your GitHub repo.
CSV_FILE_PATH = "website_embeddings.csv"
URL_COLUMN_NAME = "url"
EMBEDDING_COLUMN_NAME = "Vector Embedding text-embedding-004" # Make sure this matches your CSV
MODEL_NAME = "text-embedding-004"
TOP_N_RESULTS = 5

# --- Load Data (Cached for performance in Streamlit) ---
@st.cache_data # Streamlit's caching decorator
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if URL_COLUMN_NAME not in df.columns:
            st.error(f"Error: Column '{URL_COLUMN_NAME}' not found in the CSV. Please check the file and column name configuration.")
            return None, None, None
        if EMBEDDING_COLUMN_NAME not in df.columns:
            st.error(f"Error: Column '{EMBEDDING_COLUMN_NAME}' not found in the CSV. Please check the file and column name configuration.")
            return None, None, None

        # Convert string embeddings to numerical lists/arrays
        df['embedding_vector'] = df[EMBEDDING_COLUMN_NAME].astype(str).apply(ast.literal_eval)
        df['embedding_vector'] = df['embedding_vector'].apply(np.array)

        urls_list = df[URL_COLUMN_NAME].tolist()
        # Filter out any rows where embedding conversion might have failed or is not a numpy array
        valid_embeddings = [emb for emb in df['embedding_vector'] if isinstance(emb, np.ndarray) and emb.ndim == 1]

        if not valid_embeddings:
            st.error("No valid embeddings found after processing. Check CSV embedding format.")
            return urls_list, np.array([]), df # Return empty array for embeddings

        embeddings_matrix = np.array(valid_embeddings) # Stack valid 1D arrays

        # Basic validation of embedding dimension (assuming text-embedding-004 which is 768)
        expected_dim = 768
        if embeddings_matrix.shape[1] != expected_dim:
            st.warning(f"Embeddings dimension is {embeddings_matrix.shape[1]}, but expected {expected_dim} for '{MODEL_NAME}'. Results might be inaccurate.")

        return urls_list, embeddings_matrix, df
    except FileNotFoundError:
        st.error(f"Error: The CSV file '{csv_path}' was not found. Make sure it's in the same directory as app.py in your GitHub repository.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        return None, None, None

# --- Configure Gemini API ---
# For Streamlit Community Cloud, set GEMINI_API_KEY as a secret in the app settings.
# For local development, you can set it as an environment variable
# or temporarily hardcode it (NOT RECOMMENDED FOR GITHUB).
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") # Recommended for Streamlit Cloud

# Fallback for local development if you set it as an environment variable
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

api_configured_successfully = False
if not GEMINI_API_KEY:
    st.error("Gemini API Key not found. Please set it in Streamlit secrets (for deployed app) or as an environment variable GEMINI_API_KEY (for local run).")
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        api_configured_successfully = True
    except Exception as e:
        st.error(f"Error configuring Gemini API: {e}. Please check your API key.")

# --- Main Streamlit UI ---
st.title("📝 Semantic Search Engine")
st.write("Enter a query to find the most relevant pages from the indexed website.")

# Load data
urls, page_embeddings, _ = load_data(CSV_FILE_PATH)

if urls is not None and page_embeddings is not None:
    if page_embeddings.size > 0:
        st.success(f"Successfully loaded and processed {len(urls)} pages from '{CSV_FILE_PATH}'.")
    else:
        st.warning("Data loaded, but no valid page embeddings were found. Search will not function.")

    if api_configured_successfully:
        user_query = st.text_input("Your search query:", placeholder="e.g., information about data privacy")

        if st.button("Search") or (user_query and not st.session_state.get('search_clicked_once', False)):
            st.session_state['search_clicked_once'] = True # Handle initial auto-submit on text input if desired
            if user_query:
                if page_embeddings.size == 0:
                    st.warning("No page embeddings available to search. Please check the data source.")
                else:
                    try:
                        with st.spinner(f"Embedding query and searching..."):
                            # Embed the query
                            result = genai.embed_content(
                                model=f"models/{MODEL_NAME}",
                                content=user_query,
                                task_type="RETRIEVAL_QUERY" # Important for search
                            )
                            query_embedding_list = result['embedding']
                            query_embedding_np = np.array(query_embedding_list).reshape(1, -1)

                            # Calculate similarities
                            similarities = cosine_similarity(query_embedding_np, page_embeddings)
                            similarity_scores = similarities[0]

                            # Rank and Display Results
                            scored_urls = sorted(list(zip(similarity_scores, urls)), key=lambda x: x[0], reverse=True)

                        st.subheader(f"Top {min(TOP_N_RESULTS, len(scored_urls))} results for '{user_query}':")
                        if not scored_urls:
                            st.write("No results found.")
                        else:
                            for i, (score, url) in enumerate(scored_urls[:TOP_N_RESULTS]):
                                st.markdown(f"{i+1}. **[{url}]({url})** (Similarity: {score:.4f})")
                                # st.write(f"   URL: {url}")
                                # st.write(f"   Similarity: {score:.4f}")
                                st.divider()

                    except Exception as e:
                        st.error(f"An error occurred during search: {e}")
            elif st.session_state.get('search_clicked_once'): # if button was clicked with no query
                st.warning("Please enter a search query.")
    else:
        st.warning("Gemini API not configured. Search functionality is disabled.")
else:
    st.error("Failed to load data. Please check the CSV file and configurations.")

st.markdown("---")
st.markdown("Powered by Streamlit and Gemini")