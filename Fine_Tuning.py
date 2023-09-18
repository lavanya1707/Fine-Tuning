import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load your dataset (replace 'your_dataset.csv' with the actual dataset file path)
df = pd.read_csv('wiki_movie.csv')

# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('msmarco-distilbert-base-v2')

# Load the Faiss index
index = faiss.read_index('movie_plot.index')

# Define a function to perform semantic search
def search(query, top_k, index, model):
    query_vector = model.encode([query])
    top_k = index.search(query_vector, top_k)
    
    top_k_ids = top_k[1].tolist()[0]
    top_k_ids = list(np.unique(top_k_ids))
    results = [fetch_movie_info(idx) for idx in top_k_ids]
    return results

# Define a function to fetch movie info based on dataframe index
def fetch_movie_info(dataframe_idx):
    info = df.iloc[dataframe_idx]
    meta_dict = {}
    meta_dict['Title'] = info['Title']
    meta_dict['Plot'] = info['Plot'][:500]
    return meta_dict

# Streamlit app title
st.title('Fine Tuning')

# Input text area for user query
query = st.text_input('Enter your movie plot query:', 'Terrors attack and people running around for help')

# Button to trigger semantic search
if st.button('Search'):
    # Perform semantic search
    results = search(query, top_k=3, index=index, model=model)
    
    # Display search results
    st.subheader('Search Results:')
    for result in results:
        st.write(f'Title: {result["Title"]}')
        st.write(f'Plot: {result["Plot"]}')
        st.write('---')
