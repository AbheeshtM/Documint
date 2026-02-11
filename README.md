Perfect 👍
Here’s a clean, final, recruiter-ready README for DocuMint, followed by a short product description you can reuse on GitHub, Streamlit Cloud, and LinkedIn.

You can copy–paste this directly.

📘 DocuMint

DocuMint is a production-grade, grounded Retrieval-Augmented Generation (RAG) system that allows users to query large documents with strict citations, zero hallucination, and full auditability.

It is designed for real-world PDFs (policies, reports, technical docs) and supports documents up to 100 pages / 25 MB, deployed as a live Streamlit application.

🚀 Overview

DocuMint enables reliable document question-answering by combining:

Local embeddings

FAISS vector search

Groq LLMs for fast inference

Every answer is:

Grounded only in retrieved document text

Traceable to exact chunks and pages

Safely refused when evidence is missing

✨ Key Features
📂 Universal Document Support

Deterministic parsing for PDF, DOCX, TXT, HTML, CSV, Markdown, and source code

LLM-assisted fallback parsing for unsupported formats

🧠 Hybrid GenAI Architecture

Local Embeddings
Uses sentence-transformers locally (no external embedding APIs)

Groq LLMs
Fast, free-tier friendly inference for grounded answer generation

🔎 FAISS-Based Retrieval

Session-scoped FAISS vector index

Distance-based similarity search

Safe fallback to prevent false “no context” refusals

🧾 Strict Grounding & Safety

Answers generated only from retrieved context

Explicit refusal for out-of-scope questions

No hallucinations or external knowledge leakage

🔐 Token & Memory Safety

Enforced limits on:

Query length

Context size

Output tokens

Prevents prompt overflow and instability

📊 Structured Logging

JSON logs for:

Ingestion steps

Retrieval scores

LLM latency

Useful for debugging and evaluation

📦 Exportable RAG Sessions

Download a complete session ZIP containing:

Original document

Cleaned text

Chunk metadata

FAISS index

Chat history

Configuration snapshot

📏 System Limits
Constraint	Value
Max PDF pages	100 pages
Max file size	25 MB
Vector store	FAISS (local)
Embeddings	Local (sentence-transformers)

Limits are intentional to ensure stable retrieval quality and performance.

🛠️ Prerequisites

Python 3.10+

Groq API Key
https://console.groq.com/keys

⚙️ Installation
git clone https://github.com/AbheeshtM/Documint.git
cd Documint

python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt

▶️ Running Locally
Set Groq API Key

Windows (PowerShell)

$env:GROQ_API_KEY="your_api_key_here"


Linux / macOS

export GROQ_API_KEY="your_api_key_here"

Launch App
streamlit run app/streamlit_app.py

☁️ Deployment (Streamlit Cloud – Free)

Push the repository to GitHub

Go to Streamlit Cloud

Create a new app from the repo

Add GROQ_API_KEY under Secrets

Set entry point:

app/streamlit_app.py


Your app will be live with a public URL.

🗂️ Project Structure
Documint/
│
├── app/
│   └── streamlit_app.py
│
├── rag/
│   ├── parsing.py
│   ├── cleaning.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── export.py
│   ├── config.py
│   └── logging_utils.py
│
├── sessions/
├── requirements.txt
└── README.md

🧠 Design Philosophy

Grounded over clever

Auditable over opaque

Stable over experimental

DocuMint prioritizes trust, traceability, and correctness.
