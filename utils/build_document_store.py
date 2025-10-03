#!/usr/bin/env python3

import argparse
from urllib.parse import urlparse

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# Installation Notes:
#
# If running on CPU, install CPU variants of dependencies
# pip install fastembed
# pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# If running on GPU, install GPU variants of dependencies
# pip install fastembed-gpu torch
#
# Core dependencies
# pip install llama-index-embeddings-huggingface llama-index-vector-stores-qdrant
#
# For PDF plain text extraction (faster)
# pip install llama-index-readers-file PyMuPDF
#
# For PDF Markdown extraction (recommended)
# pip install pymupdf4llm


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a document store using LlamaIndex and Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required positional argument
    parser.add_argument("input_dir", help="Directory containing input documents")

    # Optional arguments with defaults
    parser.add_argument(
        "--qdrant-url",
        default="./qdrant_db",
        help="Path to a local Qdrant directory or remote Qdrant instance",
    )
    parser.add_argument(
        "--qdrant-collection-name",
        default="llamacollection",
        help="Qdrant collection to build",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model for text embeddings",
    )
    parser.add_argument(
        "--output-format",
        choices=["plain", "markdown"],
        default="plain",
        help="Output format for document parsing",
    )

    return parser.parse_args()


def build_document_store(args: argparse.Namespace) -> None:
    """Build the document store with the given arguments."""
    # Set up document parser based on output format
    if args.output_format == "plain":
        # pip install llama-index-readers-file PyMuPDF
        from llama_index.readers.file import PyMuPDFReader

        parser_obj = PyMuPDFReader()
    elif args.output_format == "markdown":
        # pip install pymupdf4llm
        from pymupdf4llm import LlamaMarkdownReader

        parser_obj = LlamaMarkdownReader()
    else:
        raise NotImplementedError

    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=args.embedding_model)

    # Initialize vector store
    parsed_url = urlparse(args.qdrant_url, scheme="file")
    if parsed_url.scheme == "file":
        client = QdrantClient(path=parsed_url.path)
        kwargs = {"client": client}
    else:
        kwargs = {"url": args.qdrant_url, "api_key": ""}

    vector_store = QdrantVectorStore(
        collection_name=args.qdrant_collection_name,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        **kwargs,
    )

    # Load documents
    file_extractor = {".pdf": parser_obj}
    documents = SimpleDirectoryReader(
        args.input_dir, recursive=True, file_extractor=file_extractor
    ).load_data(show_progress=True)

    # Build index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    print(f"Successfully built document store with {len(documents)} documents")
    print(f"Qdrant URL: {args.qdrant_url}")
    print(f"Qdrant Collection Name: {args.qdrant_collection_name}")
    print(f"Embedding Model: {args.embedding_model}")


def main() -> None:
    """Main entry point of the script."""
    args = parse_arguments()
    build_document_store(args)


if __name__ == "__main__":
    main()
