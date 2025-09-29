# openwebui-plugins

This repository contains custom tools for Open WebUI.

## Tools

### Document Search

Document Search is a tool that retrieves documents from a Milvus vector store using hybrid search (combining dense vector similarity and BM25 sparse retrieval for improved accuracy). This tool is ideal for Retrieval-Augmented Generation (RAG) applications where you need to search through your own document collection with automatic citations.

The tool accepts `query` as a required argument and `similarity_top_k` (default: 5) and `filters` as optional arguments. The `filters` parameter allows metadata filtering using a list of dictionaries with 'key' and 'value' pairs (e.g., `[{"key": "file_name", "value": "report.pdf"}]`).

#### Setup

1. **Build the Document Store** (optional): Use `utils/build_document_store.py` to create your vector store:

```bash
# Build the document store (default options, fast)
python utils/build_document_store.py /path/to/your/documents

# Build the document store (recommended alternative, slower but more accurate)
python utils/build_document_store.py \
  --embedding_model Snowflake/snowflake-arctic-embed-m-v1.5 \
  --output_format markdown \
  /path/to/your/documents
```

2. **Import the Tool**: Import the tool into Open WebUI (Workspace - Tools).

3. **Configure Settings**: Click the valves settings icon and configure it for the document store that was created (or existing).

To maximize performance, build the document store on a faster computer using a larger model (e.g., `Snowflake/snowflake-arctic-embed-m-v1.5`) for text embeddings. You can still perform queries on a slower computer using a smaller model for query embeddings (e.g., `MongoDB/mdbr-leaf-ir`) that is aligned with the larger model.

For available embeddings, refer to the [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for English or Multilingual and sort by the Retrieval column.

#### Command-Line Utility

`utils/build_document_store.py` supports several configuration options:

```bash
usage: build_document_store.py [-h] [--milvus_uri MILVUS_URI]
                               [--milvus_collection_name MILVUS_COLLECTION_NAME]
                               [--embedding_model EMBEDDING_MODEL]
                               [--output_format {plain,markdown}]
                               input_dir

Build a document store using LlamaIndex and Milvus

positional arguments:
  input_dir             Directory containing input documents

options:
  -h, --help            show this help message and exit
  --milvus_uri MILVUS_URI
                        Path to a Milvus Lite database file or remote Milvus
                        instance (default: ./milvus_llamaindex.db)
  --milvus_collection_name MILVUS_COLLECTION_NAME
                        Milvus collection to build (default: llamalection)
  --embedding_model EMBEDDING_MODEL
                        HuggingFace model for text embeddings
                        (default: sentence-transformers/all-MiniLM-L6-v2)
  --output_format {plain,markdown}
                        Output format for document parsing (default: plain)
```

Installing dependencies for the utility:

```bash
# Install required dependencies
pip install "numpy<2"  # to avoid runtime warnings
pip install torch --index-url https://download.pytorch.org/whl/cpu  # if using CPU-only
pip install llama-index-core llama-index-readers-file \
  llama-index-embeddings-huggingface llama-index-vector-stores-milvus \
  milvus-lite

# Additional dependencies for parsing PDF files
pip install PyMuPDF  # (--output-format=plain)
pip install pymupdf4llm  # (--output-format=markdown)
```

### Linkup Web Search

Linkup Web Search is a tool that provides real-time web search capabilities with citations.

The tool accepts `query` as a required argument. Optional additional arguments are `from_date`, `to_date`, `exclude_domains`, and `include_domains`.

#### Setup

After importing the tool into Open WebUI (Workspace - Tools), click the valves settings icon and set the Linkup API key.

The tool can be configured to return search results for model grounding or a sourced answer for reduced model usage. When output type is set to "searchResults" (default), it returns the raw search results including content and emits a citation for each result. When output type is set to "sourcedAnswer", it returns an answer with a list of sources and emits a citation for each source. "searchResults" will generally provide more accurate model grounding, but use more model context.

### Perplexity Web Search (OpenRouter)

Perplexity Web Search (OpenRouter) is ported from the [Perplexity Web Search Tool](https://openwebui.com/t/abhiactually/perplexity) to use the OpenRouter API with a configurable model.

#### Setup

After importing the tool into Open WebUI (Workspace - Tools), click the valves settings icon and set the OpenRouter API key.

The tool can be configured to use Perplexity Sonar Pro (default) or another model.

## License

Copyright (c) 2025 Dara Adib

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
