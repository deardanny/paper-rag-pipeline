# Research Paper Q&A System (RAG Pipeline)

A command-line **Retrieval-Augmented Generation (RAG)** prototype for asking questions about a local research PDF. It retrieves relevant text chunks and passes them to GPT-4o-mini alongside the question.

[Case study and demonstration](https://hdjang.com/projects/rag/) · [Application code](app.py)

## Overview

Retrieval gives the model document context instead of relying only on its training data. The prompt asks the model to answer from that context, but retrieval and prompting do not guarantee factual correctness.

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

Type `quit` at the question prompt to exit. Closing input (EOF, usually Ctrl+D) or pressing Ctrl+C at either prompt also exits cleanly.

```
Enter PDF path (e.g. paper.pdf): your_paper.pdf
Loading paper...
Split into 14 chunks
Building vector store...

Ready! Ask anything about the paper. (type 'quit' to exit)

Question: What is the main contribution of this paper?
Answer: ...
```

## Example question

An illustrative question for a paper describing a three-pass reading approach:

> **Q:** What is the three-pass approach?  
> **A:** The three-pass approach is a method for reading research papers in up to three passes, each with a specific goal...

## Current scope

- One local PDF per run; the vector store is created at startup and no persistence directory is configured.
- Text-based PDFs are supported. There is no OCR pipeline for scanned pages.
- The CLI prints the model's answer, without a separate retrieved-passage or page-citation display.
- No benchmark dataset or accuracy evaluation is bundled. Check important answers against the source PDF.

## Author

Daniel (Heedong) Jang | [hdjang.com](https://hdjang.com) | CMU Heinz '26
