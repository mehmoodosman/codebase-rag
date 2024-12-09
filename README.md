# Codebase RAG

A Retrieval Augmented Generation (RAG) system for querying and understanding codebases using Streamlit, Pinecone, and LLMs.

## Overview

This project implements a conversational interface for querying codebases using RAG technology. It combines vector similarity search through Pinecone with Large Language Models to provide contextually relevant answers about your code.
![image](https://github.com/user-attachments/assets/021f60e1-c367-40c0-b44f-7ca8d39b73f7)


## Features

- 🔍 **Semantic Code Search**: Uses HuggingFace embeddings to find relevant code snippets
- 💬 **Conversational Interface**: Built with Streamlit for an intuitive chat experience
- 📚 **Multiple Repository Support**: Switch between different codebases using namespaces which are added to Pinecone
- 🤖 **Advanced LLM Integration**: Powered by Groq's LLama models
- 🔄 **Context-Aware Responses**: Maintains chat history for coherent conversations

## Usage

1. Visit the Streamlit web application: https://rag-my-codebase.streamlit.app/
2. Select a repository namespace from the sidebar
3. Start asking questions about your codebase in the chat interface

## Architecture

- **Frontend**: Streamlit web interface
- **Embedding**: HuggingFace Sentence Transformers
- **Vector Store**: Pinecone
- **LLM**: Groq (LLama models)
- **RAG Framework**: Custom implementation with context augmentation

## Future Work & Challenges

- Implement AST parsing of codebase embeddings rather than dumping the whole codebase to embeddings to allow for more accurate and relevant answers as code follows a different structure to natural language.
- Add a way to update the Pinecone index when you push any new commits to your repo. This would be done through a webhook that's triggered on each commit, where the codebase is re-embedded and added to Pinecone.
- Add a way to chat with multiple codebases at the same time.
- Add support for image uploads when chatting with the codebase this is called Multimodal RAG.
