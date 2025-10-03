"""
title: Document Search
author: daradib
author_url: https://github.com/daradib/
git_url: https://github.com/daradib/openwebui-plugins.git
description: Retrieves documents from a Milvus vector store. Supports hybrid search for agentic knowledge base RAG.
requirements: llama-index-embeddings-ollama, llama-index-vector-stores-milvus
version: 0.1.2
license: AGPL-3.0-or-later
"""


# Notes:
#
# To use HuggingFace embeddings instead of Ollama, edit requirements line above
# changing llama-index-embeddings-ollama to llama-index-embeddings-huggingface.
#
# Connection caching and citation indexing use async locking, but assume a
# single-node/worker (default). If a multi-node/worker deployment of Open WebUI
# will call this tool from separate workers, consider modifying it to use Redis
# for state synchronization.


import asyncio
import codecs
import json
import re
from typing import Any, Callable, Dict, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    ExactMatchFilter,
    FilterCondition,
    MetadataFilters,
)
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import BM25BuiltInFunction
from pydantic import BaseModel, Field


def get_vector_index(
    milvus_uri: str,
    milvus_collection_name: str,
    embedding_model: str,
    embedding_query_instruction: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
) -> VectorStoreIndex:
    """
    Initialize and return the VectorStoreIndex object.
    """
    # Connect to the existing Milvus vector store.
    # `overwrite=False` ensures we don't delete the existing data.
    # `enable_sparse=True` is necessary to signal the use of hybrid search.
    vector_store = MilvusVectorStore(
        uri=milvus_uri,
        collection_name=milvus_collection_name,
        overwrite=False,
        enable_sparse=True,
        sparse_embedding_function=BM25BuiltInFunction(),
    )

    # Load the specified embedding model from HuggingFace or Ollama.
    # Interpret escape sequences, e.g., literal '\n' into an actual newline.
    if embedding_query_instruction:
        query_instruction = codecs.decode(embedding_query_instruction, "unicode_escape")
    else:
        query_instruction = None
    if ollama_base_url:
        embed_model = OllamaEmbedding(
            model_name=embedding_model,
            base_url=ollama_base_url,
            query_instruction=query_instruction,
        )
    else:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            query_instruction=query_instruction,
        )

    # Create the index object from the existing vector store.
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    return index


def build_filters(file_name: str) -> MetadataFilters:
    """
    Build a LlamaIndex MetadataFilters object to filter by filename.
    """
    return MetadataFilters(
        filters=[ExactMatchFilter(key="file_name", value=file_name)],
        condition=FilterCondition.AND,
    )


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
    Remove internal LlamaIndex node attributes.
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
        # Lock required to prevent race conditions in check-and-set operation
        # and to ensure citations are emitted in citation_id order.
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
            description="Path to a Milvus Lite database file or remote Milvus instance.",
        )
        milvus_collection_name: str = Field(
            default="llamacollection",
            description="Milvus collection containing both dense vectors and BM25 sparse vectors.",
        )
        embedding_model: str = Field(
            default="sentence-transformers/all-MiniLM-L6-v2",
            description="Model for query embeddings, which should match the model used to create the text embeddings.",
        )
        embedding_query_instruction: Optional[str] = Field(
            default=None,
            description="Instruction to prepend to query before embedding, e.g., 'Query:'. Escape sequences like \\n are interpreted.",
        )
        ollama_base_url: Optional[str] = Field(
            default=None,
            description="Base URL for Ollama API (recommended). When set, uses Ollama instead of downloading the embedding model from HuggingFace.",
        )

    def __init__(self):
        """
        Initialize the tool and its valves.
        Disables automatic citation handling to allow for custom citation events.
        """
        self.valves = self.Valves()
        self.citation = False
        self._index = None
        self._last_config = None
        self._lock = asyncio.Lock()

    async def retrieve_documents(
        self,
        query: str,
        top_k: int = 5,
        file_name: Optional[str] = None,
        __metadata__: Optional[Dict[str, Any]] = None,
        __event_emitter__: Optional[Callable[[Dict], Any]] = None,
    ) -> str:
        """
        Retrieve relevant documents from the Milvus vector store using hybrid search.

        :param query: The natural language search query.
        :param top_k: The number of top documents to retrieve.
        :param file_name: The filename to optionally filter results by.
        :param __metadata__: Injected by Open WebUI with information about the chat.
        :param __event_emitter__: Injected by Open WebUI to send events to the frontend.
        """

        async def emit_status(description: str, done: bool = False, hidden=False):
            """Helper function to emit status updates."""
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": description,
                            "done": done,
                            "hidden": hidden,
                        },
                    }
                )

        if top_k < 1 or top_k > 20:
            return "Error: top_k must be between 1 and 20."

        if file_name:
            parsed_filters = build_filters(file_name)
        else:
            parsed_filters = None

        filter_desc = f" in {file_name}" if file_name else ""
        await emit_status(f"Searching{filter_desc} for: {query}")

        try:
            # Cache and reuse the VectorStoreIndex object.
            # Lock required to prevent concurrent requests from closing/recreating
            # the index simultaneously, which could cause client errors.
            async with self._lock:
                current_config = self.valves.model_dump_json()
                if not self._index or self._last_config != current_config:
                    if self._index:
                        self._index.vector_store.client.close()
                    self._index = get_vector_index(**dict(self.valves))
                    self._last_config = current_config

            # Create a query engine with hybrid search mode and async execution.
            retriever = self._index.as_retriever(
                vector_store_query_mode="hybrid",
                similarity_top_k=top_k,
                filters=parsed_filters,
                use_async=True,
            )

            nodes = await retriever.aretrieve(query)

            if nodes:
                await emit_status("Search complete.", done=True, hidden=True)
            else:
                await emit_status("No documents found.", done=True)
                return "No relevant documents found for the query."

            if __metadata__:
                # Lock required to prevent concurrent requests from creating
                # separate CitationIndex instances, which would cause citation_id
                # collisions and loss of citation state across the conversation.
                if "document_search_citation_index" not in __metadata__:
                    async with self._lock:
                        if "document_search_citation_index" not in __metadata__:
                            __metadata__["document_search_citation_index"] = (
                                CitationIndex()
                            )
                citation_index = __metadata__["document_search_citation_index"]
            else:
                citation_index = CitationIndex()

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
            await emit_status(error_message, done=True, hidden=False)
            return error_message
