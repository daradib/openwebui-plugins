#!/usr/bin/env python3

import argparse
from collections.abc import Iterator
from urllib.parse import urlparse

from llama_index.core.schema import TextNode
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Installation Notes:
#
# If running on CPU, install fastembed dependency
# pip install fastembed
#
# If running on GPU, install fastembed-gpu dependency
# pip install fastembed-gpu
#
# Core dependencies
# pip install llama-index-vector-stores-milvus llama-index-vector-stores-qdrant milvus-lite


def get_milvus_vector_store(
    milvus_uri: str,
    milvus_collection_name: str,
) -> MilvusVectorStore:
    """
    Connect to Milvus and return the collection object.
    """
    vector_store = MilvusVectorStore(
        uri=milvus_uri,
        collection_name=milvus_collection_name,
        overwrite=False,
    )
    return vector_store


def get_qdrant_vector_store(
    qdrant_url: str,
    qdrant_collection_name: str,
) -> QdrantVectorStore:
    """
    Initialize and return the QdrantVectorStore object.
    """
    parsed_url = urlparse(qdrant_url, scheme="file")
    if parsed_url.scheme == "file":
        client = QdrantClient(path=parsed_url.path)
        kwargs = {"client": client}
    else:
        kwargs = {"url": qdrant_url, "api_key": ""}
    vector_store = QdrantVectorStore(
        collection_name=qdrant_collection_name,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25",
        **kwargs,
    )
    return vector_store


def query_all_milvus(
    milvus_vector_store: MilvusVectorStore, batch_size: int = 1000
) -> Iterator[dict]:
    """
    Fetch all documents from Milvus.
    """
    offset = 0
    while True:
        results = milvus_vector_store.client.query(
            collection_name=milvus_vector_store.collection_name,
            filter="",
            output_fields=["*"],
            limit=batch_size,
            offset=offset,
        )
        if not results:
            break
        yield from results
        offset += batch_size


def convert(
    milvus_uri: str,
    milvus_collection_name: str,
    qdrant_url: str,
    qdrant_collection_name: str,
) -> None:
    """
    Convert Milvus vector store to Qdrant vector store.
    """
    print("Connecting to Milvus...")
    milvus_vector_store = get_milvus_vector_store(milvus_uri, milvus_collection_name)
    print("Milvus vector store loaded.")

    print("Initializing Qdrant vector store...")
    qdrant_vector_store = get_qdrant_vector_store(qdrant_url, qdrant_collection_name)
    print(f"Qdrant vector store connected at '{qdrant_url}'.")

    print("Preparing documents from Milvus...")
    results = query_all_milvus(milvus_vector_store)

    print("Preparing documents for Qdrant...")
    nodes_to_insert = []
    for item in results:
        if item["_node_type"] == "TextNode":
            node = TextNode.from_json(item["_node_content"])
        else:
            raise NotImplementedError
        node.embedding = item["embedding"]
        node.text = item["text"]
        nodes_to_insert.append(node)

    print(f"Adding {len(nodes_to_insert)} documents to Qdrant...")
    qdrant_vector_store.add(nodes_to_insert)
    print("Documents added to Qdrant successfully.")

    print("\nConversion complete!")
    print(f"Qdrant database is saved at: {qdrant_url}")
    print(f"Collection name: {qdrant_collection_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Milvus vector store to Qdrant vector store for llama-index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--milvus-uri",
        type=str,
        default="./milvus_llamaindex.db",
        help="Path to a Milvus Lite database file or remote Milvus instance",
    )
    parser.add_argument(
        "--milvus-collection-name",
        type=str,
        default="llamacollection",
        help="Existing Milvus collection name",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="./qdrant_db",
        help="Path to local Qdrant directory or remote Qdrant instance",
    )
    parser.add_argument(
        "--qdrant-collection-name",
        type=str,
        default="llamacollection",
        help="New Qdrant collection name",
    )

    args = parser.parse_args()

    convert(
        milvus_uri=args.milvus_uri,
        milvus_collection_name=args.milvus_collection_name,
        qdrant_url=args.qdrant_url,
        qdrant_collection_name=args.qdrant_collection_name,
    )
