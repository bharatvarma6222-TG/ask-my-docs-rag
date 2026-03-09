A Retrieval-Augmented Generation (RAG) application that allows users to query documents and receive answers grounded in retrieved evidence with citations.

The system combines hybrid retrieval (BM25 + vector search), cross-encoder reranking, and citation enforcement to ensure reliable responses.

🚀 Features

📄 PDF Document Ingestion

🔍 Hybrid Retrieval

BM25 keyword search

Vector similarity search (FAISS)

🧠 Cross-Encoder Reranking

📑 Citation-Based Answers

❌ Refusal Mechanism

Returns "I don't know" when evidence is insufficient

📊 Confidence Score

🧪 Evaluation Pipeline

💬 Streamlit UI for interaction

🏗️ Architecture
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embeddings
    ↓
Hybrid Retrieval
(BM25 + Vector Search)
    ↓
Cross-Encoder Reranker
    ↓
Answer Generation
    ↓
Citations + Confidence Score
🖥️ Demo Workflow

1️⃣ Upload a PDF
2️⃣ System ingests and indexes the document
3️⃣ Ask a question
4️⃣ Receive:

Extractive answer

Evidence citations

Confidence score

📂 Project Structure
ask-my-docs-rag
│
├── app
│   ├── api.py
│   ├── rag.py
│   ├── ingestion.py
│   ├── retrieval_bm25.py
│   ├── retrieval_vector.py
│   ├── retrieval_hybrid.py
│   ├── reranker.py
│   └── ui.py
│
├── eval
│   ├── run_eval.py
│   └── golden.jsonl
│
├── data
│
├── requirements.txt
└── README.md
⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/ask-my-docs-rag.git
cd ask-my-docs-rag

Create virtual environment:

python -m venv .venv

Activate environment:

Windows

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt
▶️ Running the Application

Start the API server:

python -m uvicorn app.api:app --reload

Start the UI:

streamlit run app/ui.py
📊 Evaluation

Run evaluation tests:

python eval/run_eval.py

The evaluation pipeline checks:

answer correctness

citation validity

regression detection

📸 Example Output

Example query:

What is RAG and how does it work?

Response:

RAG stands for Retrieval Augmented Generation.
A RAG system retrieves relevant documents before generating answers.

Citations:

rag_test_document.pdf_p1_c0

Confidence:

0.50
🛠️ Tech Stack

Python

FastAPI

Streamlit

FAISS

Sentence Transformers

BM25

NumPy

📈 Future Improvements

Add local LLM support via Ollama

Streaming responses

Multi-document indexing

Vector database integration

Observability metrics

Latency tracking

🎯 Purpose

This project demonstrates how to build a production-oriented RAG system that emphasizes:

reliability

transparency

evaluation

system design

👤 Author

Bharat Varma

AI / ML Enthusiast focused on building production-ready AI systems.
