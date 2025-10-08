#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterator
from typing import Optional
from urllib.parse import urlparse

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Record
from tqdm import tqdm

DOCUMENT_FIELD = "doc_id"
METADATA_FIELDS = [
    "file_path",
    "file_name",
    "file_type",
    "file_size",
    "creation_date",
    "last_modified_date",
]


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


def get_qdrant_points(
    client: QdrantClient, collection_name: str, batch_size: int
) -> Iterator[Record]:
    offset = None
    while True:
        points, next_page_offset = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=[DOCUMENT_FIELD] + METADATA_FIELDS,
            with_vectors=False,
        )
        yield from points
        if next_page_offset is None:
            break
        offset = next_page_offset


def summarize_qdrant_documents(
    qdrant_url: str,
    collection_name: str,
    output_csv_path: str,
    api_key: Optional[str] = None,
    batch_size: int = 1000,
) -> None:
    """
    Queries Qdrant to obtain a list of documents, their metadata,
    and the number of nodes for each document, then outputs this
    information to a CSV file.

    Args:
        qdrant_url: URL or path to the Qdrant instance.
        collection_name: The name of the Qdrant collection to query.
        output_csv_path: The path to the output CSV file.
        api_key: API key for remote Qdrant instance (optional).
        batch_size: Number of points to retrieve per scroll request.
    """
    print(f"Connecting to Qdrant at '{qdrant_url}' ... ", end="")
    client = get_qdrant_client(qdrant_url, api_key)
    print("connection successful.")

    # Use defaultdict to easily group points by DOCUMENT_ID
    # Each entry will store node count and the metadata for the document
    document_summary = defaultdict(
        lambda: {
            "node_count": 0,
            "metadata": {field: None for field in METADATA_FIELDS},
        }
    )

    points = get_qdrant_points(client, collection_name, batch_size)
    points_count = client.count(collection_name=collection_name).count

    for point in tqdm(
        points,
        desc=f"Querying collection '{collection_name}'",
        total=points_count,
        miniters=batch_size,
        disable=None,
    ):
        doc_id = point.payload.get(DOCUMENT_FIELD)
        if doc_id:
            document_summary[doc_id]["node_count"] += 1
            # Store metadata from the first point encountered for this doc_id
            # Assuming metadata is consistent across all nodes of a document
            if document_summary[doc_id]["node_count"] == 1:
                for field in METADATA_FIELDS:
                    document_summary[doc_id]["metadata"][field] = point.payload.get(
                        field
                    )
            else:
                for field in METADATA_FIELDS:
                    assert document_summary[doc_id]["metadata"][
                        field
                    ] == point.payload.get(field)

    document_summary = dict(sorted(document_summary.items(), key=lambda item: item[0]))
    print(
        f"Finished processing points. Found {len(document_summary)} unique documents."
    )

    # Prepare data for CSV
    csv_headers = [DOCUMENT_FIELD, "node_count"] + METADATA_FIELDS
    csv_data = []
    for doc_id, data in document_summary.items():
        row = {DOCUMENT_FIELD: doc_id, "node_count": data["node_count"]}
        row.update(data["metadata"])
        csv_data.append(row)

    # Write to CSV file
    print(f"Writing summary to '{output_csv_path}'...")
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_data)


def main() -> None:
    """Main entry point of the script."""
    parser = argparse.ArgumentParser(
        description="Export a summary of documents stored in Qdrant to a CSV file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="./qdrant_db",
        help="Path to a local Qdrant directory or remote Qdrant instance URL (e.g., 'http://localhost:6333')",
    )
    parser.add_argument(
        "--qdrant-collection",
        type=str,
        default="llamacollection",
        help="Name of the Qdrant collection to query",
    )
    parser.add_argument(
        "--qdrant-api-key",
        type=str,
        default=None,
        help="API key for remote Qdrant instance (optional)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="qdrant_document_summary.csv",
        help="Path to the output CSV file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of points to retrieve per scroll request",
    )

    args = parser.parse_args()

    summarize_qdrant_documents(
        qdrant_url=args.qdrant_url,
        collection_name=args.qdrant_collection,
        api_key=args.qdrant_api_key,
        output_csv_path=args.output_csv,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
