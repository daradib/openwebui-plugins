#!/usr/bin/env python3

import argparse
from typing import Optional
from urllib.parse import urlparse

from qdrant_client import QdrantClient


# Installation Notes:
#
# Core dependencies
# pip install qdrant-client


def get_qdrant_client(qdrant_url: str, api_key: Optional[str] = None) -> QdrantClient:
    """
    Connect to Qdrant and return the client object.
    Supports both local (file-based) and remote Qdrant instances.
    """
    parsed_url = urlparse(qdrant_url, scheme="file")
    if parsed_url.scheme == "file":
        # Local file-based Qdrant
        client = QdrantClient(path=parsed_url.path)
    else:
        # Remote Qdrant instance
        client = QdrantClient(url=qdrant_url, api_key=api_key)
    return client


def copy_qdrant(
    src_url: str,
    dst_url: str,
    collection_names: Optional[list[str]] = None,
    recreate_on_collision: bool = False,
    batch_size: int = 100,
    src_api_key: Optional[str] = None,
    dst_api_key: Optional[str] = None,
) -> None:
    """
    Copy collections from source Qdrant to destination Qdrant using the migrate method.

    Args:
        src_url: Source Qdrant URL (local path or remote URL)
        dst__url: Destination Qdrant URL (local path or remote URL)
        collection_names: List of collection names to copy. If None, copies all collections.
        recreate_on_collision: If True, recreate collection if it exists in destination
        batch_size: Batch size for scrolling and uploading vectors
        src_api_key: API key for source Qdrant (if remote)
        dst_api_key: API key for destination Qdrant (if remote)
    """
    print("Connecting to source Qdrant...")
    src_client = get_qdrant_client(src_url, src_api_key)
    print(f"Source Qdrant client connected at '{src_url}'.")

    print("Connecting to destination Qdrant...")
    dst_client = get_qdrant_client(dst_url, dst_api_key)
    print(f"Destination Qdrant client connected at '{dst_url}'.")

    # Get list of collections to migrate
    if collection_names:
        print(f"Migrating collections: {', '.join(collection_names)}")
    else:
        source_collections = src_client.get_collections().collections
        collection_names = [collection.name for collection in source_collections]
        print(f"Migrating all collections: {', '.join(collection_names)}")

    # Use the migrate method
    print("Starting migration...")
    src_client.migrate(
        dest_client=dst_client,
        collection_names=collection_names,
        recreate_on_collision=recreate_on_collision,
        batch_size=batch_size,
    )

    print("\nMigration complete!")
    print(f"Destination Qdrant database: {dst_url}")
    print(f"Migrated collections: {', '.join(collection_names)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy Qdrant collections to another Qdrant instance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src-url",
        type=str,
        required=True,
        help="Source Qdrant URL (local path like './qdrant_db' or remote URL like 'http://localhost:6333')",
    )
    parser.add_argument(
        "--dst-url",
        type=str,
        required=True,
        help="Destination Qdrant URL (local path or remote URL)",
    )
    parser.add_argument(
        "--collection-names",
        type=str,
        nargs="+",
        default=None,
        help="Collection names to migrate (space-separated). If not specified, all collections will be migrated.",
    )
    parser.add_argument(
        "--recreate-on-collision",
        action="store_true",
        help="If set, recreate collection if it already exists in destination",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for scrolling and uploading vectors",
    )
    parser.add_argument(
        "--src-api-key",
        type=str,
        default=None,
        help="API key for source Qdrant instance (if remote)",
    )
    parser.add_argument(
        "--dst-api-key",
        type=str,
        default=None,
        help="API key for destination Qdrant instance (if remote)",
    )

    args = parser.parse_args()

    copy_qdrant(
        src_url=args.src_url,
        dst_url=args.dst_url,
        collection_names=args.collection_names,
        recreate_on_collision=args.recreate_on_collision,
        batch_size=args.batch_size,
        src_api_key=args.src_api_key,
        dst_api_key=args.dst_api_key,
    )
