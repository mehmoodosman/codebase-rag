import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from langchain.schema import Document


client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# Set the PINECONE_API_KEY as an environment variable
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
os.environ['PINECONE_API_KEY'] = pinecone_api_key

# Initialize Pinecone and get list of namespaces
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
pinecone_index = pc.Index("codebase-rag")
namespaces = pinecone_index.describe_index_stats()['namespaces'].keys()

# Add namespace selection to sidebar
st.sidebar.title("Settings")
selected_namespace = st.sidebar.selectbox(
    "Select Repository Namespace",
    options=list(namespaces),
    index=0
)

def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)


def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    # Update query to use selected namespace
    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(), 
        top_k=10, # Can change this to 10 or 20 to get more context  
        include_metadata=True,
        namespace=selected_namespace  # Use the selected namespace
    )

    # Get the list of retrieved texts
    contexts = [item['metadata']['text'] for item in top_matches['matches']]

    augmented_query = "<CONTEXT>\n" + "\n\n-------\n\n".join(contexts[ : 10]) + "\n-------\n</CONTEXT>\n\n\n\nMY QUESTION:\n" + query

    # Modify the prompt below as need to improve the response quality
    system_prompt = f"""You are a Senior Software Engineer.

    Answer any questions I have about the codebase, based on the code provided. Always consider all of the context provided when forming a response. Also take a step by step approach in your problem-solving.
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # TODO: Change to llama-3.1-8b-instant, or llama-3.1-70b-versatile       
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": augmented_query}
        ]
    )

    return llm_response.choices[0].message.content


st.title("Codebase RAG")
st.caption(f"Currently browsing: {selected_namespace}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your codebase..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response using RAG
    with st.chat_message("assistant"):
        response = perform_rag(prompt)
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})