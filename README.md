# Production-Grade RAG System

A modular, production-ready Retrieval-Augmented Generation (RAG) system built with Python, LangChain, FAISS, and Groq LLM.

## Features

- **Universal File Support**: Deterministic parsing for PDF, DOCX, TXT, HTML, CSV, MD, and source code. LLM-assisted parsing for unknown formats.
- **Hybrid GenAI Architecture**:
  - **Local Embeddings**: Generates vectors locally using `sentence-transformers` (no external API calls for embedding).
  - **Groq LLM**: High-speed generation using Groq's API.
- **FAISS Vector Database**: Per-upload, sesQsion-based vector store for efficient retrieval.
- **Strict Grounding**: Prompt templates enforce answers based solely on provided context with zero hallucination.
- **Token & Memory Safety**: Strict enforcement of token limits for queries, context, and output.
- **Structured Logging**: Production-friendly JSON logs for tracking performance, retrieval scores, and LLM latency.
- **Exportable Sessions**: Download the entire RAG session (original file, cleaned text, metadata, FAISS index, chat history) as a ZIP.

## Prerequisites

- Python 3.10 or higher.
- A [Groq API Key](https://console.groq.com/keys).

## Installation

1. **Clone the repository** (or navigate to the project directory).

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. **Set your Groq API Key**:
   - **Windows (PowerShell)**:
     ```powershell
     $env:GROQ_API_KEY="your_api_key_here"
     ```
   - **Linux/macOS**:
     ```bash
     export GROQ_API_KEY="your_api_key_here"
     ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Use the App**:
   - Upload a document via the sidebar or main area.
   - Wait for the indexing process (extraction, cleaning, chunking, embedding).
   - Start chatting with your document.
   - Use the "Sources" expander to verify citations.
   - Download the session ZIP if needed.

## Project Structure

- `app/`: Streamlit UI and session management.
- `rag/`: Core RAG logic.
  - `parsing.py`: Deterministic and LLM-assisted file parsing.
  - `chunking.py`: Token-based text segmentation.
  - `embeddings.py`: Local vector generation.
  - `vector_store.py`: FAISS index management.
  - `retrieval.py`: Similarity search and filtering.
  - `generation.py`: Groq-powered grounded answer generation.
  - `export.py`: ZIP bundle creation.
  - `config.py`: Centralized configuration and limits.
- `sessions/`: (Generated) Temporary storage for active RAG sessions.

## Deployment

To deploy to **Streamlit Cloud**:
1. Push this project to a GitHub repository.
2. Connect the repository to Streamlit Cloud.
3. Add `GROQ_API_KEY` to the app's **Secrets**.
4. Set the main file path to `app/streamlit_app.py`.
