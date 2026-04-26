# AskMyDocs

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent PDF document querying.

AskMyDocs lets you ingest PDF documents and ask natural language questions against them — powered by Google Gemini, ChromaDB, and Sentence Transformers. It operates as a fully agentic pipeline with built-in ambiguity detection and answer validation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Features](#features)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

AskMyDocs is a unified agentic pipeline built on Google ADK that enables users to:

- Ingest and index PDF documents into a persistent vector store
- Ask natural language questions grounded strictly in the ingested document content
- Receive validated, context-aware responses with automatic ambiguity handling

> Scope: This system processes text content from PDFs only. Images, audio, and video are not supported.

---

## Architecture

AskMyDocs runs two coordinated pipelines:

### Ingestion Pipeline

PDF Upload → Text Extraction → Chunking → Embedding → ChromaDB Storage

1. Place PDF in the documents/ directory
2. Text is extracted using PyMuPDF
3. Content is chunked into semantically meaningful segments
4. Chunks are converted to vector embeddings via Sentence Transformers
5. Embeddings and metadata are persisted in ChromaDB

### Query Pipeline

User Query → Ambiguity Check → Vector Retrieval → LLM Generation → Relevance Validation → Response

1. User submits a natural language question
2. Query is checked for ambiguity before retrieval
3. Relevant chunks are retrieved from ChromaDB via semantic search
4. Gemini generates a grounded answer using retrieved context
5. Answer is validated for relevance before being returned

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | [Google ADK](https://google.github.io/adk-docs/) |
| LLM | [Google Gemini](https://deepmind.google/technologies/gemini/) |
| Vector Database | [ChromaDB](https://www.trychroma.com/) |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) |
| PDF Parsing | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) |

---

## Prerequisites

- Python 3.9 or higher
- A valid [Google API key](https://aistudio.google.com/app/apikey) with Gemini access

---

## Installation

1. Clone the repository

git clone https://github.com/your-username/askmydocs.git
cd askmydocs

2. Create and activate a virtual environment

python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

---

## Configuration

Set your Google API key as an environment variable before running the application:

# macOS / Linux
export GOOGLE_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set GOOGLE_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:GOOGLE_API_KEY="your_api_key_here"

> Tip: For persistent configuration, add this to your shell profile (.bashrc, .zshrc, etc.) or use a .env file with python-dotenv.

---

## Usage

Start the agent

adk web

Ingest a PDF

Place your PDF in the documents/ folder, then send:

upload sample.pdf

Ask questions

Once ingested, query the document in natural language:

What is the main topic of the document?
What methodology is described in section 3?
Summarize the key findings.

The agent will only answer based on content found in the ingested document. If context is insufficient, it will say so rather than hallucinate.

---

## Project Structure
