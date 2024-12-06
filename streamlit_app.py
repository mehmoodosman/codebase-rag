import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=st.secrets["GROQ_API_KEY"]
)

# Set the PINECONE_API_KEY as an environment variable
pinecone_api_key = st.secrets["PINECONE_API_KEY"]

# Initialize Pinecone and get list of namespaces
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
pinecone_index = pc.Index("codebase-rag")
# Convert namespaces to sorted list for consistent ordering
namespace_list = sorted(list(pinecone_index.describe_index_stats()['namespaces'].keys()))

# Initialize session state for namespace if it doesn't exist
if "selected_namespace" not in st.session_state:
    st.session_state.selected_namespace = namespace_list[0]

# Update sidebar selection to use session state
selected_namespace = st.sidebar.selectbox(
    "Select Repository Namespace",
    options=namespace_list,  # Use the sorted list
    key="selected_namespace"
)


def get_huggingface_embeddings(text, model_name="sentence-transformers/all-mpnet-base-v2"):
    model = SentenceTransformer(model_name)
    return model.encode(text)


def perform_rag(query):
    raw_query_embedding = get_huggingface_embeddings(query)

    # Update query to use selected namespace and include more metadata
    top_matches = pinecone_index.query(
        vector=raw_query_embedding.tolist(), 
        top_k=10,  # Reduced since we're getting more focused chunks
        include_metadata=True,
        namespace=selected_namespace
    )

    # Enhanced context building with metadata
    contexts = []
    for item in top_matches['matches']:
        metadata = item['metadata']
        context_header = f"File: {metadata.get('filepath', 'Unknown')}"
        if 'type' in metadata:
            context_header += f"\n{metadata['type']}: {metadata['name']} (Line {metadata['line_number']})"
        
        contexts.append(f"{context_header}\n```\n{metadata['text']}\n```")

    augmented_query = (
        "<CODE_CONTEXT>\n" + 
        "\n\n---\n\n".join(contexts) + 
        "\n</CODE_CONTEXT>\n\n" +
        "QUESTION:\n" + query
    )

    system_prompt = """You are a Senior Software Engineer specializing in code analysis.
    
    Analyze the provided code context carefully, considering:
    1. The structure and relationships between code components
    2. The specific implementation details and patterns
    3. The filepath and location of each code segment
    4. The type of code segment (e.g., function, class, etc.)
    5. The name of the code segment
    
    When answering questions:
    - Reference specific parts of the code and their locations
    - Explain the reasoning behind the implementation
    - Suggest improvements if relevant to the question
    - Consider the broader context of the codebase
    - Always use the code context to answer the question
    - Take a step by step approach in your problem-solving
    """

    llm_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
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