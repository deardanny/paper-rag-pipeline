# Research Paper Q&A System (RAG Pipeline)

An end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that enables semantic search and Q&A over academic PDF documents — ask any question about a paper and get context-aware, document-grounded answers.

## Overview

LLMs hallucinate when answering questions outside their training data. This project grounds every answer in the actual content of uploaded research papers using a RAG architecture.

## How It Works

```
PDF → Chunk → Embed → Vector DB → Query → Retrieve → LLM → Answer
```

1. **Load** — PDF is parsed and split into 1,000-character chunks (200 overlap)
2. **Embed** — Each chunk is converted to a vector using OpenAI Embeddings
3. **Store** — Vectors are stored in ChromaDB for semantic search
4. **Query** — User question is embedded and matched against stored chunks
5. **Generate** — Top 3 relevant chunks + question are passed to GPT-4o-mini for grounded response

## Stack

| Tool | Purpose |
|---|---|
| LangChain | Orchestration |
| ChromaDB | Vector store |
| OpenAI API | Embeddings + GPT-4o-mini |
| PyPDF | PDF parsing |
| Python | Core language |

## Setup

```bash
pip install langchain langchain-openai langchain-community langchain-text-splitters chromadb pypdf openai python-dotenv
```

The CLI reads a PDF from a local path. Building the index sends its text chunks to OpenAI for embeddings. Each question and up to three retrieved chunks are then sent to the chat model. Use a document you can share with the API.

Create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

## Usage

```bash
python app.py
```

```
Enter PDF path (e.g. paper.pdf): your_paper.pdf
Loading paper...
Split into 14 chunks
Building vector store...

Ready! Ask anything about the paper. (type 'quit' to exit)

Question: What is the main contribution of this paper?
Answer: ...
```

## Example

Tested on *How to Read a Paper* (Keshav, 2016):

> **Q:** What is the three-pass approach?  
> **A:** The three-pass approach is a method for reading research papers in up to three passes, each with a specific goal...

## Author

Daniel (Heedong) Jang | [hdjang.com](https://hdjang.com) | CMU Heinz '26
