# openwebui-plugins

This repository contains custom tools for Open WebUI.

## Linkup Web Search

Linkup Web Search is a tool that provides real-time web search capabilities with citations.

The tool accepts `query` as a required argument. Optional additional arguments are `from_date`, `to_date`, `exclude_domains`, and `include_domains`.

### Setup

After importing the tool into Open WebUI (Workspace - Tools), click the valves settings icon and set the Linkup API key.

The tool can be configured to return search results for model grounding or a sourced answer for reduced model usage. When output type is set to "searchResults" (default), it returns the raw search results including content and emits a citation for each result. When output type is set to "sourcedAnswer", it returns an answer with a list of sources and emits a citation for each source. "searchResults" will generally provide more accurate model grounding, but use more model context.

## Perplexity Web Search (OpenRouter)

Perplexity Web Search (OpenRouter) is ported from the [Perplexity Web Search Tool](https://openwebui.com/t/abhiactually/perplexity) to use the OpenRouter API with a configurable model.

### Setup

After importing the tool into Open WebUI (Workspace - Tools), click the valves settings icon and set the OpenRouter API key.

The tool can be configured to use Perplexity Sonar Pro (default) or another model.
