"""
title: Document Search
author: daradib
author_url: https://github.com/daradib/
git_url: https://github.com/daradib/openwebui-plugins.git
description: Retrieves documents from a Milvus vector store using hybrid search.
requirements: llama-index-core, llama-index-embeddings-huggingface, llama-index-vector-stores-milvus
version: 0.1.0
license: AGPL-3.0-or-later
"""

# Citation indexing is async-safe, but assumes a single-node/worker (default).
# If multi-node/worker deployments of Open WebUI will call this tool from
# separate workers, consider modifying to use Redis for state synchronization.

import asyncio
import json
import re
from typing import Any, Callable, Dict, List, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    FilterCondition,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from pydantic import BaseModel, Field


def parse_filters(
    filters_list: Optional[List[Dict[str, Any]]],
) -> Optional[MetadataFilters]:
    """
    Parses a list of filter dictionaries into a LlamaIndex MetadataFilters object.
    """
    if not filters_list:
        return None

    filter_objects = [
        ExactMatchFilter(key=f["key"], value=f["value"])
        for f in filters_list
        if f.get("key") and f.get("value")
    ]

    if not filter_objects:
        return None

    return MetadataFilters(filters=filter_objects, condition=FilterCondition.AND)


def clean_text(text: str) -> str:
    """
    Remove unwanted formatting and artifacts from text output.
    """
    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", "", text)
    # Replace multiple blank lines with a single blank line.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Remove lines with only whitespace.
    text = re.sub(r"^\s*$", "", text, flags=re.MULTILINE)
    # Remove excessive whitespace within lines.
    text = re.sub(r" +", " ", text)
    # Replace 4 or more periods with just 3 periods.
    text = re.sub(r"\.{4,}", "...", text)
    # Remove citation references.
    # Workaround for https://github.com/open-webui/open-webui/issues/17062
    text = re.sub(r"\[\d+\]", "", text)
    # Strip leading/trailing whitespace.
    return text.strip()


def clean_node(node: NodeWithScore, citation_id: int) -> dict:
    """
    Removes internal LlamaIndex node attributes.
    """
    metadata_fields_to_keep = {
        "file_name",
        "file_type",
        "page",
        "source",
        "title",
        "total_pages",
    }
    cleaned_node = {
        "id": citation_id,
        "id_": node.id_,
        "metadata": {
            k: v for k, v in node.metadata.items() if k in metadata_fields_to_keep
        },
        "text": clean_text(node.text),
        "score": node.score,
    }
    return cleaned_node


class CitationIndex:
    def __init__(self):
        self._set = set()
        self._count = 0
        self._lock = asyncio.Lock()

    async def emit_citation(self, node, __event_emitter__: Callable[[Dict], Any]):
        source_name = node.metadata.get("file_name", "Retrieved Document")
        source_name += f" ({node.id_})"
        page_number = node.metadata.get("page") or node.metadata.get("source")
        if page_number:
            source_name += f" - p. {page_number}"
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [clean_text(node.text)],
                    "metadata": [
                        {
                            "source": source_name,
                        }
                    ],
                    "source": {"name": source_name},
                },
            }
        )

    async def add_if_not_exists(
        self,
        node: NodeWithScore,
        __event_emitter__: Optional[Callable[[Dict], Any]] = None,
    ) -> Optional[int]:
        async with self._lock:
            if node.id_ in self._set:
                return None
            else:
                if __event_emitter__:
                    await self.emit_citation(node, __event_emitter__)
                self._set.add(node.id_)
                self._count += 1
                return self._count


class Tools:
    """
    A toolset for interacting with an existing Milvus vector store for Retrieval-Augmented Generation
    """

    class Valves(BaseModel):
        milvus_uri: str = Field(
            default="./milvus_llamaindex.db",
            description="Path to a Milvus Lite database file or remote Milvus instance",
        )
        milvus_collection_name: str = Field(
            default="llamalection",
            description="Milvus collection containing both dense vectors and BM25 sparse vectors",
        )
        embedding_model: str = Field(
            default="sentence-transformers/all-MiniLM-L6-v2",
            description="HuggingFace model for query embeddings, which must match (or be aligned with) the model used to create the text embeddings",
        )

    def __init__(self):
        """
        Initializes the tool and its valves.
        Disables automatic citation handling to allow for custom citation events.
        """
        self.valves = self.Valves()
        # Disable automatic citations from the retriever, as we will send our own.
        self.citation = False
        self._index = None
        self._last_config = None
        self._lock = asyncio.Lock()
        asyncio.get_running_loop().create_task(self._get_index())

    async def _get_index(
        self, __event_emitter__: Optional[Callable[[Dict], Any]] = None
    ) -> Optional[VectorStoreIndex]:
        """
        Initializes and returns the VectorStoreIndex object, caching it for efficiency.
        """
        current_config = self.valves.model_dump_json()
        if self._index and self._last_config == current_config:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Using cached vector store connection...",
                            "done": False,
                        },
                    }
                )
            return self._index

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Initializing vector store connection...",
                        "done": False,
                    },
                }
            )

        try:
            # Connect to the existing Milvus vector store.
            # `overwrite=False` ensures we don't delete the existing data.
            # `enable_sparse=True` is necessary to signal the use of hybrid search.
            vector_store = MilvusVectorStore(
                uri=self.valves.milvus_uri,
                collection_name=self.valves.milvus_collection_name,
                overwrite=False,
                enable_sparse=True,
                sparse_embedding_function=BM25BuiltInFunction(),
            )

            # Load the specified embedding model from HuggingFace.
            embed_model = HuggingFaceEmbedding(model_name=self.valves.embedding_model)

            # Create the index object from the existing vector store.
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model=embed_model
            )

            self._index = index
            self._last_config = current_config
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Vector store connection successful.",
                            "done": False,
                        },
                    }
                )
            return self._index

        except Exception as e:
            error_message = f"Failed to connect to vector store or load model: {e}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": error_message,
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
            return None

    async def retrieve_documents(
        self,
        query: str,
        similarity_top_k: int = 5,
        filters: Optional[List[Dict[str, Any]]] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Dict], Any]] = None,
    ) -> str:
        """
        Retrieves relevant documents from the Milvus vector store using hybrid search.

        :param query: The natural language search query.
        :param similarity_top_k: The number of top documents to retrieve.
        :param filters: A list of dictionaries for metadata filtering, e.g., [{"key": "file_name", "value": "report.pdf"}].
        :param __metadata__: Injected by Open WebUI with information about the chat.
        :param __event_emitter__: Injected by Open WebUI to send events to the frontend.
        """
        if __metadata__:
            if "document_search_citation_index" not in __metadata__:
                async with self._lock:
                    if "document_search_citation_index" not in __metadata__:
                        __metadata__["document_search_citation_index"] = CitationIndex()
            citation_index = __metadata__["document_search_citation_index"]
        else:
            citation_index = CitationIndex()

        index = await self._get_index(__event_emitter__)

        if not index:
            return "Error: Could not establish a connection with the vector store. Please check the configuration."

        parsed_filters = parse_filters(filters)
        if filters and not parsed_filters:
            return "Error: Invalid filter format. Each filter must be a dictionary with 'key' and 'value'."

        if __event_emitter__:
            filter_desc = " with metadata filters" if parsed_filters else ""
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Performing hybrid search{filter_desc}...",
                        "done": False,
                    },
                }
            )

        try:
            # Create a query engine with hybrid search mode and async execution.
            retriever = index.as_retriever(
                vector_store_query_mode="hybrid",
                similarity_top_k=similarity_top_k,
                filters=parsed_filters,
                use_async=True,
            )

            nodes = await retriever.aretrieve(query)

            if not nodes:
                return "No relevant documents found for the query."

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Search complete.",
                            "done": True,
                            "hidden": True,
                        },
                    }
                )

            documents = []
            for node in nodes:
                citation_id = await citation_index.add_if_not_exists(
                    node, __event_emitter__
                )
                if citation_id:
                    documents.append(clean_node(node, citation_id=citation_id))

            return json.dumps(documents)

        except Exception as e:
            error_message = f"An error occurred during search: {e}"
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": error_message,
                            "done": True,
                            "hidden": False,
                        },
                    }
                )
            return error_message
