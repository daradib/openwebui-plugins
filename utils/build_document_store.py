#!/usr/bin/env python3

import argparse

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction


# Installation Notes:
# pip install "numpy<2" # to downgrade NumPy if needed to avoid runtime warning
# pip install torch --index-url https://download.pytorch.org/whl/cpu # to install torch without gpu dependencies
# pip install llama-index-core llama-index-readers-file llama-index-embeddings-huggingface llama-index-vector-stores-milvus milvus-lite


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build a document store using LlamaIndex and Milvus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required positional argument
    parser.add_argument("input_dir", help="Directory containing input documents")

    # Optional arguments with defaults
    parser.add_argument(
        "--milvus_uri",
        default="./milvus_llamaindex.db",
        help="Path to a Milvus Lite database file or remote Milvus instance",
    )
    parser.add_argument(
        "--milvus_collection_name",
        default="llamacollection",
        help="Milvus collection to build",
    )
    parser.add_argument(
        "--embedding_model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model for text embeddings",
    )
    parser.add_argument(
        "--output_format",
        choices=["plain", "markdown"],
        default="plain",
        help="Output format for document parsing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the existing Milvus collection if it exists",
    )

    return parser.parse_args()


def build_document_store(args: argparse.Namespace):
    """Build the document store with the given arguments."""
    # Set up document parser based on output format
    if args.output_format == "plain":
        # pip install PyMuPDF
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
    milvus_dim = embed_model._model.get_sentence_embedding_dimension()
    vector_store = MilvusVectorStore(
        uri=args.milvus_uri,
        collection_name=args.milvus_collection_name,
        overwrite=args.overwrite,
        dim=milvus_dim,
        enable_sparse=True,
        sparse_embedding_function=BM25BuiltInFunction(),
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
    print(f"Milvus URI: {args.milvus_uri}")
    print(f"Milvus Collection Name: {args.milvus_collection_name}")
    print(f"Embedding Model: {args.embedding_model}")


def main():
    """Main entry point of the script."""
    args = parse_arguments()
    build_document_store(args)


if __name__ == "__main__":
    main()
