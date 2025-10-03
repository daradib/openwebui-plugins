#!/usr/bin/env python3

import argparse
import codecs
import multiprocessing
from urllib.parse import urlparse

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


# For dependency installation instructions see README.md


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
        "--embedding-text-instruction",
        default=None,
        help="Instruction to prepend to text before embedding, e.g., 'passage:'. Escape sequences like \\n are interpreted.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Base URL for Ollama API (recommended). When set, uses Ollama instead of downloading the embedding model from HuggingFace.",
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
        from llama_index.readers.file import PyMuPDFReader

        parser_obj = PyMuPDFReader()
    elif args.output_format == "markdown":
        from pymupdf4llm import LlamaMarkdownReader

        parser_obj = LlamaMarkdownReader()
    else:
        raise NotImplementedError

    # Interpret escape sequences, e.g., literal '\n' into an actual newline.
    if args.embedding_text_instruction:
        embed_kwargs = {
            "text_instruction": codecs.decode(
                args.embedding_text_instruction, "unicode_escape"
            )
        }
    else:
        embed_kwargs = {}

    # Initialize embedding model
    if args.ollama_base_url:
        from llama_index.embeddings.ollama import OllamaEmbedding

        embed_model = OllamaEmbedding(
            model_name=args.embedding_model,
            base_url=args.ollama_base_url,
            **embed_kwargs,
        )
    else:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        embed_model = HuggingFaceEmbedding(
            model_name=args.embedding_model,
            **embed_kwargs,
        )

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
    num_workers = multiprocessing.cpu_count()
    documents = SimpleDirectoryReader(
        args.input_dir,
        recursive=True,
        filename_as_id=True,
        file_extractor=file_extractor,
    ).load_data(show_progress=True, num_workers=num_workers)

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
    if args.ollama_base_url:
        print(f"Ollama Base URL: {args.ollama_base_url}")


def main() -> None:
    """Main entry point of the script."""
    args = parse_arguments()
    build_document_store(args)


if __name__ == "__main__":
    main()
