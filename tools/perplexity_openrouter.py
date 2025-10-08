"""
title: Perplexity Web Search (OpenRouter)
author: daradib
author_url: https://github.com/daradib
git_url: https://github.com/daradib/openwebui-plugins.git
description: Search the web using Perplexity AI via OpenRouter API.
version: 0.1.1
license: MIT
"""
# This is ported from the Perplexity Web Search Tool:
# https://openwebui.com/t/abhiactually/perplexity

import json
import re
from typing import Any, Callable, Optional

import aiohttp
from pydantic import BaseModel, Field

CITATION_PATTERN = re.compile(r"\[\d+\]")


def clean(s: str) -> str:
    """Remove citations from string."""
    # Workaround for https://github.com/open-webui/open-webui/issues/17062
    return CITATION_PATTERN.sub("", s)


class Tools:
    class Valves(BaseModel):
        openrouter_api_key: str = Field(
            default="", description="Required API key to access OpenRouter services"
        )
        openrouter_api_base_url: str = Field(
            default="https://openrouter.ai/api/v1",
            description="The base URL for OpenRouter API endpoints",
        )
        model: str = Field(
            default="perplexity/sonar-pro",
            description="Model to use for search",
        )

    def __init__(self) -> None:
        self.valves = self.Valves()

    async def perplexity_web_search(
        self, query: str, __event_emitter__: Optional[Callable[[dict], Any]] = None
    ) -> str:
        """
        Search the web using Perplexity AI

        :param query: The search query to look up
        """
        if not self.valves.openrouter_api_key:
            raise Exception("openrouter_api_key not provided in valves")

        # Status emitter helper
        async def emit_status(
            description: str, status: str = "in_progress", done: bool = False
        ) -> None:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": description,
                            "status": status,
                            "done": done,
                            "hidden": False,
                        },
                    }
                )

        # Initial status
        await emit_status(f"Asking Perplexity: {query}", "searching")

        headers = {
            "Authorization": f"Bearer {self.valves.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.valves.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful search assistant. Provide concise and accurate information.",
                },
                {"role": "user", "content": query},
            ],
            "stream": True,
        }

        try:
            await emit_status("Processing search results...", "processing")

            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.valves.openrouter_api_base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        raise Exception(
                            f"HTTP {response.status} {response.reason}: {error_text}"
                        )

                    content = ""
                    citations = []
                    buffer = b""
                    done = False

                    async for chunk in response.content.iter_chunked(1024):
                        if not chunk:
                            continue
                        buffer += chunk

                        while b"\n" in buffer:
                            line, buffer = buffer.split(b"\n", 1)
                            line = line.strip()
                            if not line:
                                continue

                            try:
                                decoded = line.decode("utf-8")
                            except UnicodeDecodeError:
                                continue

                            if decoded.startswith("data: "):
                                line_data = decoded[6:]
                                if line_data.strip() == "[DONE]":
                                    done = True
                                    break

                                try:
                                    chunk_json = json.loads(line_data)
                                except json.JSONDecodeError:
                                    continue

                                delta = chunk_json.get("choices", [{}])[0].get(
                                    "delta", {}
                                )

                                # Extract content
                                if "content" in delta and delta["content"]:
                                    content += delta["content"]

                                # Extract citations from annotations (if present)
                                for annotation in delta.get("annotations", []):
                                    if annotation.get("type") == "url_citation":
                                        citation = annotation.get("url_citation") or {}
                                        url = citation.get("url")
                                        if url and url not in citations:
                                            citations.append(url)

                        if done:
                            break

            # Emit Perplexity as primary source
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "citation",
                        "data": {
                            "document": [content],
                            "metadata": [{"source": "Perplexity AI Search"}],
                            "source": {"name": "Perplexity AI"},
                        },
                    }
                )

            # Emit each URL citation
            if citations and __event_emitter__:
                for url in citations:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": ["Perplexity AI Source"],
                                "metadata": [{"source": url}],
                                "source": {"name": url},
                            },
                        }
                    )

            # Complete status
            await emit_status(
                "Search completed successfully", status="complete", done=True
            )

            # Format response with all citations
            response_text = f"{content}\n\nSources:\n"
            response_text += "- Perplexity AI Search\n"
            for url in citations:
                response_text += f"- {url}\n"

            return clean(response_text)

        except Exception as e:
            error_msg = f"Error performing web search: {str(e)}"
            await emit_status(error_msg, status="error", done=True)
            return error_msg
