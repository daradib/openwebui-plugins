"""
title: Linkup Web Search
author: daradib
author_url: https://github.com/daradib/
git_url: https://github.com/daradib/openwebui-plugins.git
description: Search the web using the Linkup API. Provides real-time web search capabilities with citations.
requirements: linkup-sdk
version: 0.1.3
license: AGPL-3.0-or-later
"""

from datetime import date
import json
import re
from typing import Any, Callable, Dict, Optional

from linkup import LinkupClient
from pydantic import BaseModel, Field


CITATION_PATTERN = re.compile(r"\[\d+\]")


def clean(s: str) -> str:
    """Remove citations from string."""
    # Workaround for https://github.com/open-webui/open-webui/issues/17062
    return CITATION_PATTERN.sub("", s)


class Tools:
    class Valves(BaseModel):
        linkup_api_key: str = Field(
            default="", description="Get a Linkup API key at https://linkup.so/"
        )
        output_type: str = Field(
            default="searchResults",
            description="Choose between 'searchResults' for model grounding or 'sourcedAnswer' for a direct answer",
        )

    def __init__(self):
        """Initialize the Linkup Web Search Tool."""
        self.valves = self.Valves()
        # Disable automatic citations since we're handling them manually
        self.citation = False

    async def linkup_web_search(
        self,
        query: str,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        exclude_domains: Optional[list[str]] = None,
        include_domains: Optional[list[str]] = None,
        __event_emitter__: Optional[Callable[[Dict], Any]] = None,
    ) -> str:
        # The docstring below is based on the official MCP server schema
        # without configurable search depth (always standard) and additional
        # arguments (from_date, to_date, exclude_domains, include_domains):
        # https://github.com/LinkupPlatform/python-mcp-server/blob/main/src/mcp_search_linkup/server.py
        # https://github.com/LinkupPlatform/linkup-python-sdk/blob/main/src/linkup/client.py
        """
        Search the web in real time using Linkup. Use this tool whenever the user needs trusted facts, news, or source-backed information. Returns comprehensive content from the most relevant sources.

        :param query: Natural language search query. Full questions work best, e.g., 'How does the new EU AI Act affect startups?'
        :param from_date: Date (YYYY-MM-DD) from which search results should begin
        :param to_date: Date (YYYY-MM-DD) until which search results should end
        :param exclude_domains: List of domains to exclude from search results
        :param include_domains: List of domains to only return search results for
        """

        async def emit_status(description: str, done: bool = False):
            """Helper function to emit status updates."""
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": description,
                            "done": done,
                            "hidden": False,
                        },
                    }
                )

        # Validate API key
        if not self.valves.linkup_api_key:
            error_msg = "Linkup API key is not configured. Please set your API key in the tool settings."
            await emit_status(error_msg, done=True)
            return f"Error: {error_msg}"

        # Validate output type
        if self.valves.output_type not in {"sourcedAnswer", "searchResults"}:
            error_msg = "Output type must be set to 'searchResults' for model grounding or 'sourcedAnswer' for a direct answer."
            await emit_status(error_msg, done=True)
            return f"Error: {error_msg}"

        # Convert date to python type
        from_date_obj = date.fromisoformat(from_date) if from_date else None
        to_date_obj = date.fromisoformat(to_date) if to_date else None

        try:
            await emit_status(f"Searching the web for: {query}")

            # Initialize Linkup client
            client = LinkupClient(api_key=self.valves.linkup_api_key)

            # Perform search
            response = await client.async_search(
                query=query,
                depth="standard",
                output_type=self.valves.output_type,
                from_date=from_date_obj,
                to_date=to_date_obj,
                exclude_domains=exclude_domains,
                include_domains=include_domains,
            )

            await emit_status("Processing search results", done=False)

            # Handle different output types
            if self.valves.output_type == "sourcedAnswer":
                answer = getattr(response, "answer", "No answer provided.")
                sources = getattr(response, "sources", [])

                if __event_emitter__:
                    # Emit the main answer as a citation
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [answer],
                                "metadata": [{"source": "Linkup Web Search"}],
                                "source": {"name": "Linkup Sourced Answer"},
                            },
                        }
                    )

                    # Emit each source as a citation
                    for i, source in enumerate(sources, 1):
                        content = getattr(source, "snippet", "")
                        url = getattr(source, "url", "")
                        title = getattr(source, "name", f"Search Source {i}")

                        if content:
                            await __event_emitter__(
                                {
                                    "type": "citation",
                                    "data": {
                                        "document": [content],
                                        "metadata": [{"source": url}],
                                        "source": {"name": title, "url": url},
                                    },
                                }
                            )

                await emit_status("Search completed successfully", done=True)

                # Return the main answer and list of sources
                answer_with_sources = {
                    "answer": answer,
                    "sources": [source.dict(exclude={"snippet"}) for source in sources],
                }
                return clean(json.dumps(answer_with_sources, ensure_ascii=False))

            elif self.valves.output_type == "searchResults":
                results = getattr(response, "results", [])

                if not results:
                    await emit_status("No results found", done=True)
                    return "No search results found for the given query."

                if __event_emitter__:
                    # Emit each search result as a citation
                    for i, result in enumerate(results, 1):
                        content = getattr(result, "content", "")
                        url = getattr(result, "url", "")
                        title = getattr(result, "name", f"Search Result {i}")

                        if content:
                            await __event_emitter__(
                                {
                                    "type": "citation",
                                    "data": {
                                        "document": [content],
                                        "metadata": [{"source": url}],
                                        "source": {"name": title, "url": url},
                                    },
                                }
                            )

                await emit_status("Search completed successfully", done=True)

                return clean(str(response))

            else:
                raise NotImplementedError

        except Exception as e:
            error_msg = f"Error performing Linkup web search: {str(e)}"
            await emit_status(error_msg, done=True)
            return f"Error: {error_msg}"
