# PDF Summarizer with RAG (Retrieval-Augmented Generation)

This project implements a PDF Summarization System powered by LangChain, Hugging Face Transformers, and FAISS.


It uses Retrieval-Augmented Generation (RAG) to extract, chunk, embed, and summarize the contents of any PDF into clear and concise bullet points.


## Features

Load and parse PDFs using PyPDFLoader

Split large documents into manageable text chunks

Generate embeddings with sentence-transformers/all-MiniLM-L6-v2

Store and retrieve chunks using FAISS vector database

Summarize context using google/flan-t5-base via Hugging Face pipeline

Built with modular LangChain components (PromptTemplate, RunnablePassthrough, etc.)


## Workflow 

Load PDF → PyPDFLoader

Split text → RecursiveCharacterTextSplitter

Create embeddings → HuggingFaceEmbeddings

Store vectors → FAISS

Retrieve relevant chunks

Summarize context → Flan-T5 model using RAG chain


## Tech Stack 

Python (3.9.3)

LangChain for orchestration

FAISS for vector search

Hugging Face Transformers for summarization

Sentence Transformers for embeddings

PyMuPDF / pypdf for PDF parsing

## Installation 

```
git clone https://github.com/<your-username>/pdf-summarizer-rag.git
cd pdf-summarizer-rag
pip install -r requirements.txt
```

## Usage 

Clone the repository and pass your pdf path into the loader() -> Open terminal in the folder -> Run app.py







