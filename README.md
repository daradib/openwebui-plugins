# openwebui-plugins

This repository contains custom tools for Open WebUI.

## Linkup Web Search

Linkup Web Search is a tool that provides real-time web search capabilities with citations.

The tool accepts two arguments from the model: query and depth. The depth is "standard" for quick, general queries or "deep" for comprehensive analysis.

### Setup

After importing the tool into Open WebUI (Workspace - Tools), click the valves settings icon and set the Linkup API key.

The tool can be configured to return search results for model grounding or a sourced answer for reduced model usage. When output type is set to "searchResults" (default), it returns the raw search results including content and emits a citation for each result. When output type is set to "sourcedAnswer", it returns an answer with a list of sources and emits a citation for each source.
