# openwebui-plugins

A collection of powerful tools to enhance [Open WebUI](https://github.com/open-webui/open-webui) with agentic search and retrieval capabilities for multi-step reasoning and ReAct (Reasoning and Acting) workflows.

## Table of Contents

- [Overview](#overview)
- [Tools](#tools)
  - [Document Search](#document-search)
  - [Linkup Web Search](#linkup-web-search)
  - [Perplexity Web Search (OpenRouter)](#perplexity-web-search-openrouter)
- [License](#license)

## Overview

While Open WebUI has built-in document and web search functionality, these tools provide **native tool access** that enables models to use search capabilities in agentic workflows. This allows models to:

- **Decompose complex questions** into focused sub-queries
- **Use search tools iteratively**, refining queries based on results
- **Reason through multi-step problems** using ReAct (Reasoning and Acting) patterns
- **Chain multiple searches** to build comprehensive understanding

This repository provides three specialized tools for Open WebUI:

- **Document Search**: Search document collections with hybrid semantic and keyword matching
- **Linkup Web Search**: Access current web search results with flexible filtering
- **Perplexity Web Search (OpenRouter)**: Access search summaries through OpenRouter

Each tool includes automatic citation generation and is designed for seamless integration into agentic reasoning workflows.

## Tools

### Document Search

**Purpose**: Search through document collections with agentic workflow support.

This tool enables models to iteratively explore knowledge bases through multi-step reasoning. Models can split and chain searches, using results to inform follow-up queries. This approach provides more accurate results, making it perfect for agentic Retrieval-Augmented Generation (RAG).

#### Key Features

- Hybrid semantic and keyword search for better accuracy
- Optionally specify result count (default: 5 results)
- Optionally specify file name to filter results (default: None)
- High-performance vector storage (Qdrant)
- Configurable embedding models (Ollama and HuggingFace)
- Automatic citation generation with sequential indices for inline references
- Build utility which supports multiple document formats (LlamaIndex, PyMuPDF, etc.)

#### Quick Start

1. **Prepare documents** in a folder
2. **Build the document store** using the [build utility][build_document_store.py]
3. **Import the tool** into Open WebUI
4. **Configure the connection** to the document store
5. **Create a model** in the workspace with access to the tool and a custom prompt

#### Tool Parameters

- **Required**: `query` - search query
- **Optional**: `top_k` - number of results (default: 5)
- **Optional**: `file_name` - filter results by filename (default: None)

#### Tips for Better Accuracy

- Parse PDF files as Markdown instead of plain text.
- Use a larger embedding model like the Qwen3-Embedding model series (0.6B, 4B, or 8B). Check the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard), choose English or Multilingual as appropriate, and sort by the "Retrieval" column.
- Set the query instruction for the model, e.g., for Qwen3-Embedding model series: `Given a web search query, retrieve relevant passages that answer the query\nQuery:`
- Provide context on the documents in the system prompt (see [prompt example](#example-system-prompt)).

#### Building the Document Store

Download the [build utility][build_document_store.py].

**Basic setup** (fast, good for testing):
```bash
python utils/build_document_store.py /path/to/documents
```

**Recommended setup** (slower but more accurate):
```bash
python utils/build_document_store.py \
  --embedding_model Qwen/Qwen3-Embedding-0.6B \
  --output_format markdown \
  /path/to/documents
```

To install the required dependencies for the [build utility][build_document_store.py]:

```bash
## Core dependency (required)
pip install llama-index-vector-stores-qdrant

## Choose ONE dense embedding backend (required)

### Option 1: Ollama API
pip install llama-index-embeddings-ollama

### Option 2: DeepInfra API
pip install llama-index-embeddings-deepinfra

### Option 3: HuggingFace SentenceTransformer
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install llama-index-embeddings-huggingface

# GPU (skip the torch line above if using GPU):
pip install llama-index-embeddings-huggingface

## Choose ONE sparse embedding backend (required)

### Option 1: CPU
pip install fastembed

### Option 2: GPU
pip install fastembed-gpu

## Choose ONE PDF extraction format

### Option 1: Markdown (slower but better quality)
pip install pymupdf4llm

### Option 2: plain text (faster)
pip install llama-index-readers-file PyMuPDF

## Other file formats if present (docx, xlsx, xlsx)
pip install docx2txt openpyxl xlrd
```

The [build utility][build_document_store.py] supports several options:

```
usage: build_document_store.py [-h] [--qdrant-url QDRANT_URL]
                               [--qdrant-collection-name QDRANT_COLLECTION_NAME]
                               [--embedding-model EMBEDDING_MODEL]
                               [--embedding-text-instruction EMBEDDING_TEXT_INSTRUCTION]
                               [--ollama-base-url OLLAMA_BASE_URL]
                               [--output-format {plain,markdown}]
                               input_dir

Build a document store using LlamaIndex and Qdrant

positional arguments:
  input_dir             Directory containing input documents

options:
  -h, --help            show this help message and exit
  --qdrant-url QDRANT_URL
                        Path to a local Qdrant directory or remote Qdrant
                        instance (default: ./qdrant_db)
  --qdrant-collection-name QDRANT_COLLECTION_NAME
                        Qdrant collection to build (default: llamacollection)
  --embedding-model EMBEDDING_MODEL
                        HuggingFace model for text embeddings (default:
                        sentence-transformers/all-MiniLM-L6-v2)
  --embedding-text-instruction EMBEDDING_TEXT_INSTRUCTION
                        Instruction to prepend to text before embedding, e.g.,
                        'passage:'. Escape sequences like \n are interpreted.
                        (default: None)
  --ollama-base-url OLLAMA_BASE_URL
                        Base URL for Ollama API. When set, uses Ollama instead
                        of downloading the embedding model from HuggingFace.
                        (default: None)
  --output-format {plain,markdown}
                        Output format for document parsing (default: plain)
```

There are also utilities to [copy from Milvus to Qdrant][copy_milvus_to_qdrant.py] if you're looking to migrate from Milvus as well as to [copy Qdrant collections between servers][copy_qdrant_to_qdrant.py].

[build_document_store.py]: utils/build_document_store.py
[copy_milvus_to_qdrant.py]: utils/copy_milvus_to_qdrant.py
[copy_qdrant_to_qdrant.py]: utils/copy_qdrant_to_qdrant.py

#### Example System Prompt

> You are Danesh, a highly specialized AI assistant and expert query engine for a knowledge base of documents.
>
> SEARCH STRATEGY:
> - Decompose complex questions into focused sub-queries
> - Use the search tool iteratively, refining queries based on results
> - Gather comprehensive context before synthesizing your final answer
>
> RESPONSE REQUIREMENTS:
> - Base answers STRICTLY and EXCLUSIVELY on search result information
> - If insufficient information is found, clearly state this limitation
> - Include inline citations as [1][2][3] when ID numbers are available in search results
> - Provide factual, accurate, and comprehensive responses
>
> SCOPE:
> - Focus on document search results
> - For tangentially related queries, acknowledge the connection but redirect to document-specific aspects

### Linkup Web Search

**Purpose**: Enable agentic web search with real-time information gathering.

This tool empowers models to conduct web research through iterative search strategies. Models can split and chain searches to build comprehensive understanding of topics.

#### Key Features

- Access web search results or AI-generated answers
- Date range filtering
- Domain inclusion/exclusion
- Automatic citation generation

#### Setup

1. **Import the tool** into Open WebUI (Workspace → Tools)
2. **Get a Linkup API key** from [Linkup](https://linkup.so/)
3. **Configure the tool** by clicking the valves settings icon and entering the API key

#### Tool Parameters

- **Required**: `query` - search query
- **Optional**: `from_date` - search results from this date
- **Optional**: `to_date` - search results until this date
- **Optional**: `exclude_domains` - domains to exclude from results
- **Optional**: `include_domains` - only include results from these domains

#### Output Modes

- **searchResults** (default): returns raw search results with full content and individual citations
- **sourcedAnswer**: returns an AI-generated answer with source list and citations

Choose `searchResults` for more accurate model grounding or `sourcedAnswer` to reduce token usage.

### Perplexity Web Search (OpenRouter)

**Purpose**: Access search summaries for multi-step reasoning workflows.

This tool enables models to leverage Perplexity's search summaries. It is adapted from the [Perplexity Web Search Tool](https://openwebui.com/t/abhiactually/perplexity) to support OpenRouter.

#### Key Features

- Access AI-generated answers from web search
- Configurable model selection
- Automatic citation generation

#### Setup

1. **Import the tool** into Open WebUI (Workspace → Tools)
2. **Get an OpenRouter API key** from [OpenRouter](https://openrouter.ai/)
3. **Configure the tool** by clicking the valves settings icon and entering the API key

The tool defaults to Perplexity Sonar Pro but can be configured to use other compatible models.

## License

Copyright © 2025 Dara Adib

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
