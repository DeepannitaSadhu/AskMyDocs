# 📄 AskMyDocs

> A smart Retrieval-Augmented Generation (RAG) system that allows you to **ingest PDFs and ask questions from them using AI**.

---

## 🚀 Overview

This project is a **unified agentic pipeline** that can:

* 📥 Ingest PDF documents
* ✂️ Chunk and process text
* 🧠 Store embeddings in a vector database (ChromaDB)
* 🔍 Retrieve relevant context
* 💬 Answer user queries based ONLY on the document

⚠️ Note: This system is **PDF-only** (not multimodal). It processes **text content from PDFs**.

---

## 🧠 How It Works

### 🔹 Ingestion Pipeline

1. Upload PDF to `documents/`
2. Extract text from PDF
3. Chunk text into smaller pieces
4. Convert text into embeddings
5. Store in ChromaDB with metadata

### 🔹 Query Pipeline

1. User asks a question
2. Check if query is ambiguous
3. Retrieve relevant chunks from DB
4. Generate answer using context
5. Validate answer relevance

---

## 🏗️ Project Structure

```
project/
│
├── agent.py              # Main agent definition
├── tools.py              # All pipeline tools (ingestion + retrieval)
├── documents/            # Place your PDFs here
├── chroma_db/            # Vector database storage
├── extracted_images/     # (optional, auto-created)
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* 🧩 Google ADK (Agent framework)
* 🧠 Gemini (LLM)
* 🔎 ChromaDB (Vector DB)
* 📊 Sentence Transformers (Embeddings)
* 📄 PyMuPDF (PDF parsing)

---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/your-username/pdf-rag-agent.git
cd pdf-rag-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Setup

Set your Google API key:

```bash
export GOOGLE_API_KEY="your_api_key_here"
# Windows
set GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Run the Agent

```bash
adk web
```

---

## 💡 Usage

### 📥 Ingest a PDF

Example input:

```
upload sample.pdf
```

### ❓ Ask Questions

Example:

```
What is the main topic of the document?
```

---

## ✨ Features

* ✅ Unified ingestion + query agent
* ✅ Automatic ambiguity detection
* ✅ Context-aware answering
* ✅ Relevance validation loop
* ✅ Clean modular tool design

---

## ⚠️ Limitations

* ❌ Only works with PDF text (no images, audio, video)
* ❌ Depends on quality of extracted text
* ❌ Requires API key for LLM

---

## 📌 Future Improvements

* Support for multiple document formats
* Better ranking strategies
* Streaming responses
* UI improvements

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open issues for suggestions or bugs.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it with others!
