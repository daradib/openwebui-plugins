#!/usr/bin/env python3

import argparse
import codecs
from datetime import datetime, timezone
import os
from collections import defaultdict
from typing import Dict, Set
from urllib.parse import urlparse

from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from tqdm import tqdm
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import models, QdrantClient


# For dependency installation instructions see README.md

# List of file extensions to load.
# Other formats like csv, pptx, xlsx could be added if they're useful.
ALLOWED_EXTENSIONS = [
    ".docx",
    ".md",
    ".pdf",
    ".txt",
]


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
        choices=["plain", "markdown", "json"],
        default="plain",
        help="Format to parse PDF files into",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers to use for parsing documents",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compare files between input directory and Qdrant collection without actually adding or deleting documents",
    )

    return parser.parse_args()


def get_existing_documents(
    vector_store: QdrantVectorStore, batch_size: int = 1000
) -> Dict[str, Dict]:
    """
    Retrieve existing documents from Qdrant collection.

    Returns:
        Dict mapping file_path to document metadata:
        - file_size: Size of the file
        - last_modified_date: Last modification date
        - valid: Whether all document IDs have the same metadata
        - doc_ids: Set of document IDs for this file
        - node_ids: Set of node IDs for this file
    """
    client = vector_store.client
    collection_name = vector_store.collection_name
    if not client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' does not exist. Starting fresh.")
        return {}

    document_map = defaultdict(
        lambda: {
            "file_size": None,
            "last_modified_date": None,
            "valid": None,
            "doc_ids": set(),
            "node_ids": list(),
        }
    )

    offset = None
    points_count = client.count(collection_name=collection_name).count

    if points_count == 0:
        print(f"Collection '{collection_name}' is empty. Starting fresh.")
        return {}

    print(f"Retrieving existing documents from collection '{collection_name}'...")

    with tqdm(
        total=points_count,
        desc="Loading existing documents",
        miniters=batch_size,
        disable=None,
    ) as pbar:
        while True:
            points, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=["doc_id", "file_path", "file_size", "last_modified_date"],
                with_vectors=False,
            )
            for point in points:
                doc_id = point.payload["doc_id"]
                file_path = point.payload["file_path"]
                if document_map[file_path]["valid"] is None:
                    # We have not seen this file yet so add metadata.
                    document_map[file_path]["file_size"] = point.payload["file_size"]
                    document_map[file_path]["last_modified_date"] = point.payload[
                        "last_modified_date"
                    ]
                    document_map[file_path]["valid"] = True
                else:
                    if (
                        document_map[file_path]["file_size"]
                        != point.payload["file_size"]
                        or document_map[file_path]["last_modified_date"]
                        != point.payload["last_modified_date"]
                    ):
                        # Metadata is inconsistent.
                        document_map[file_path]["valid"] = False
                document_map[file_path]["doc_ids"].add(doc_id)
                document_map[file_path]["node_ids"].append(point.id)

            pbar.update(len(points))

            if next_page_offset is None:
                break
            offset = next_page_offset

    print(f"Found {len(document_map)} unique files in vector store.")
    return dict(sorted(document_map.items(), key=lambda item: item[0]))


def get_filesystem_files(input_dir: str) -> Dict[str, Dict]:
    """
    Scan filesystem and return file metadata.

    Returns:
        Dict mapping absolute file path to metadata:
        - file_size: Size of the file in bytes
        - last_modified_date: Last modification date UTC string (YYYY-MM-DD)
    """

    def raise_error(e):
        raise e

    filesystem_files = {}

    for root, dirs, files in os.walk(input_dir, onerror=raise_error):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in ALLOWED_EXTENSIONS:
                file_path = os.path.abspath(os.path.join(root, file))
                stat = os.stat(file_path)
                timestamp_dt = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                filesystem_files[file_path] = {
                    "file_size": stat.st_size,
                    "last_modified_date": timestamp_dt.strftime("%Y-%m-%d"),
                }

    return dict(sorted(filesystem_files.items(), key=lambda item: item[0]))


def compare_and_plan_updates(
    existing_docs: Dict[str, Dict],
    filesystem_files: Dict[str, Dict],
) -> tuple[Set[str], Set[str], Set[str]]:
    """
    Compare existing documents with filesystem and plan updates.

    Returns:
        Tuple of (files_to_add, files_to_update, files_to_delete)
        - files_to_add: Files in filesystem but not in vector store
        - files_to_update: Files that exist in both but have different size/mtime
        - files_to_delete: Files in vector store but not in filesystem
    """
    files_to_add = set()
    files_to_update = set()
    files_to_delete = set()

    # Find files to add or update
    for file_path, fs_meta in filesystem_files.items():
        if file_path in existing_docs:
            vs_meta = existing_docs[file_path]
            # Compare file_size and last_modified_date (rsync-like)
            # Also update files with inconsistent metadata
            if (
                vs_meta["file_size"] != fs_meta["file_size"]
                or vs_meta["last_modified_date"] != fs_meta["last_modified_date"]
                or not vs_meta["valid"]
            ):
                files_to_update.add(file_path)
        else:
            files_to_add.add(file_path)

    # Find files to delete
    for file_path in existing_docs:
        if file_path not in filesystem_files:
            files_to_delete.add(file_path)

    return files_to_add, files_to_update, files_to_delete


def delete_documents_by_file_paths(
    vector_store: QdrantVectorStore,
    file_paths: Set[str],
    existing_docs: Dict[str, Dict],
    dry_run: bool = False,
) -> None:
    """Delete all documents associated with the given file paths."""
    client = vector_store.client
    collection_name = vector_store.collection_name

    if not file_paths:
        return

    doc_ids_to_delete = []
    for file_path in file_paths:
        doc_ids_to_delete.extend(existing_docs[file_path]["doc_ids"])

    print(
        f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} {len(doc_ids_to_delete)} document nodes from {len(file_paths)} files..."
    )

    if not dry_run:
        client.delete(
            collection_name=collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id", match=models.MatchAny(any=doc_ids_to_delete)
                    )
                ]
            ),
        )


def delete_nodes_by_file_paths(
    vector_store: QdrantVectorStore,
    file_paths: Set[str],
    existing_docs: Dict[str, Dict],
    dry_run: bool = False,
) -> None:
    """Delete all existing document nodes associated with the given file paths."""
    client = vector_store.client
    collection_name = vector_store.collection_name

    if not file_paths:
        return

    node_ids_to_delete = []
    for file_path in file_paths:
        node_ids_to_delete.extend(existing_docs[file_path]["node_ids"])

    print(
        f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} {len(node_ids_to_delete)} document nodes..."
    )

    if not dry_run:
        client.delete(
            collection_name=collection_name, points_selector=node_ids_to_delete
        )


def build_document_store(args: argparse.Namespace) -> None:
    """Build or update the document store with the given arguments."""
    # Set up document parser based on output format
    if args.format == "plain":
        from llama_index.readers.file import PyMuPDFReader

        parser_obj = PyMuPDFReader()
        file_extractor = {".pdf": parser_obj}
        transformations = [SentenceSplitter()]
    elif args.format == "markdown":
        from pymupdf4llm import LlamaMarkdownReader

        parser_obj = LlamaMarkdownReader()
        file_extractor = {".pdf": parser_obj}
        transformations = [MarkdownNodeParser(), SentenceSplitter()]
    elif args.format == "json":
        from llama_index.node_parser.docling import DoclingNodeParser
        from llama_index.readers.docling import DoclingReader

        parser_obj = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        file_extractor = {
            ".pdf": parser_obj,
            ".docx": parser_obj,
            ".xlsx": parser_obj,
            ".pptx": parser_obj,
        }
        transformations = [DoclingNodeParser(), SentenceSplitter()]
    else:
        raise NotImplementedError

    if args.embedding_text_instruction:
        # Interpret escape sequences, e.g., literal '\n' into an actual newline.
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

    # Initialize Qdrant client and vector store
    parsed_url = urlparse(args.qdrant_url, scheme="file")
    if parsed_url.scheme == "file":
        client = QdrantClient(path=parsed_url.path)
        vs_kwargs = {"client": client}
    else:
        vs_kwargs = {"url": args.qdrant_url, "api_key": args.qdrant_api_key or ""}
    vector_store = QdrantVectorStore(
        collection_name=args.qdrant_collection,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        **vs_kwargs,
    )

    # Scan filesystem
    print(f"Scanning filesystem at '{args.input_dir}'...")
    filesystem_files = get_filesystem_files(args.input_dir)
    print(f"Found {len(filesystem_files)} files in filesystem.")

    # Get existing documents from vector store
    existing_docs = get_existing_documents(vector_store)

    # Compare and plan updates
    files_to_add, files_to_update, files_to_delete = compare_and_plan_updates(
        existing_docs=existing_docs, filesystem_files=filesystem_files
    )
    files_unchanged = len(filesystem_files) - len(files_to_add) - len(files_to_update)

    print("\nUpdate plan:")
    print(f"  Files to add: {len(files_to_add)}")
    print(f"  Files to update: {len(files_to_update)}")
    print(f"  Files to delete: {len(files_to_delete)}")
    print(f"  Files unchanged: {files_unchanged}")

    if args.dry_run:
        print("\n[DRY RUN] No changes will be made.")
        if files_to_add:
            print(f"\nWould add {len(files_to_add)} files:")
            for fp in files_to_add:
                print(f"  + {fp}")
        if files_to_update:
            print(f"\nWould update {len(files_to_update)} files:")
            for fp in files_to_update:
                print(f"  ~ {fp}")
        if files_to_delete:
            print(f"\nWould delete {len(files_to_delete)} files:")
            for fp in files_to_delete:
                print(f"  - {fp}")
        return

    # Prepare files to process (add + update)
    files_to_process = files_to_add | files_to_update

    if files_to_process:
        print(f"\nProcessing {len(files_to_process)} files...")

        # Load only the files that need processing
        num_workers = args.workers
        documents = SimpleDirectoryReader(
            input_files=list(files_to_process),
            filename_as_id=True,
            file_extractor=file_extractor,
        ).load_data(show_progress=True, num_workers=num_workers)

        def is_document_custom_extractor(doc: Document) -> bool:
            """Return whether a Document uses a custom file_extractor."""
            file_extension = os.path.splitext(doc.metadata["file_name"])[1]
            return file_extension.lower() in file_extractor

        # Build index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        VectorStoreIndex.from_documents(
            [doc for doc in documents if is_document_custom_extractor(doc)],
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
            transformations=transformations,
        )
        VectorStoreIndex.from_documents(
            [doc for doc in documents if not is_document_custom_extractor(doc)],
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True,
            transformations=[SentenceSplitter()],
        )

        print(f"Successfully processed {len(documents)} documents")

    # Delete documents that are no longer in filesystem
    if files_to_delete:
        delete_documents_by_file_paths(vector_store, files_to_delete, existing_docs)

    # Delete outdated documents
    if files_to_update:
        delete_nodes_by_file_paths(vector_store, files_to_update, existing_docs)

    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Added/Updated: {len(files_to_process)} files")
    print(f"  Deleted: {len(files_to_delete)} files")
    print(f"  Unchanged: {files_unchanged} files")
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
