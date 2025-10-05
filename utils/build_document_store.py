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
        "--qdrant-collection",
        default="llamacollection",
        help="Qdrant collection to build",
    )
    parser.add_argument(
        "--qdrant-api-key",
        default=None,
        help="API key for remote Qdrant instance",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model for dense vector embeddings",
    )
    parser.add_argument(
        "--embedding-text-instruction",
        default=None,
        help="Instruction to prepend to text before embedding, e.g., 'passage:'. Escape sequences like \\n are interpreted.",
    )
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument(
        "--ollama-base-url",
        default=None,
        help="Base URL for Ollama API. When set, uses Ollama instead of downloading the embedding model from HuggingFace.",
    )
    provider_group.add_argument(
        "--deepinfra-api-key",
        default=None,
        help="API key for DeepInfra. When set, uses DeepInfra instead of downloading the embedding model from HuggingFace.",
    )
    parser.add_argument(
        "--format",
        choices=["plain", "markdown"],
        default="plain",
        help="Format to parse PDF files into",
    )

    return parser.parse_args()


def build_document_store(args: argparse.Namespace) -> None:
    """Build the document store with the given arguments."""
    # Set up document parser based on output format
    if args.format == "plain":
        from llama_index.readers.file import PyMuPDFReader

        parser_obj = PyMuPDFReader()
    elif args.format == "markdown":
        from pymupdf4llm import LlamaMarkdownReader

        parser_obj = LlamaMarkdownReader()
    else:
        raise NotImplementedError

    # Interpret escape sequences, e.g., literal '\n' into an actual newline.
    if args.embedding_text_instruction:
        text_instruction = str(
            codecs.decode(args.embedding_text_instruction, "unicode_escape", "strict")
        ).strip()
    else:
        text_instruction = None

    # Initialize embedding model
    if args.ollama_base_url:
        from llama_index.embeddings.ollama import OllamaEmbedding

        embed_model = OllamaEmbedding(
            model_name=args.embedding_model,
            base_url=args.ollama_base_url,
            text_instruction=text_instruction,
        )
    elif args.deepinfra_api_key:
        from llama_index.embeddings.deepinfra import DeepInfraEmbeddingModel

        embed_model = DeepInfraEmbeddingModel(
            model_id=args.embedding_model,
            api_token=args.deepinfra_api_key,
            text_prefix=text_instruction + " " if text_instruction else "",
        )
    else:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        embed_model = HuggingFaceEmbedding(
            model_name=args.embedding_model,
            text_instruction=text_instruction,
        )

    # Initialize vector store
    parsed_url = urlparse(args.qdrant_url, scheme="file")
    if parsed_url.scheme == "file":
        client = QdrantClient(path=parsed_url.path)
        kwargs = {"client": client}
    else:
        kwargs = {"url": args.qdrant_url, "api_key": args.qdrant_api_key or ""}

    vector_store = QdrantVectorStore(
        collection_name=args.qdrant_collection,
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
    print(f"Qdrant Collection: {args.qdrant_collection}")
    print(f"Embedding Model: {args.embedding_model}")
    if args.ollama_base_url:
        print(f"Ollama Base URL: {args.ollama_base_url}")
    elif args.deepinfra_api_key:
        print("Using DeepInfra API")


def main() -> None:
    """Main entry point of the script."""
    args = parse_arguments()
    build_document_store(args)


if __name__ == "__main__":
    main()
