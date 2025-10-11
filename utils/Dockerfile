ARG VARIANT=cpu

FROM python:3.12-slim-trixie AS base-cpu

RUN pip3 install --no-cache-dir uv
ENV UV_COMPILE_BYTECODE=1 UV_NO_BUILD=1 UV_NO_CACHE=1 UV_SYSTEM_PYTHON=1

RUN uv pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

RUN uv pip install \
    fastembed

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime AS base-cuda

RUN pip3 install --no-cache-dir uv
ENV UV_COMPILE_BYTECODE=1 UV_NO_BUILD=1 UV_NO_CACHE=1 UV_SYSTEM_PYTHON=1

RUN uv pip install \
    fastembed-gpu

FROM rocm/pytorch:rocm7.0.2_ubuntu24.04_py3.12_pytorch_release_2.8.0 AS base-rocm

RUN pip3 install --no-cache-dir uv
ENV UV_COMPILE_BYTECODE=1 UV_NO_BUILD=1 UV_NO_CACHE=1 UV_SYSTEM_PYTHON=1

RUN uv pip install \
    fastembed-gpu

FROM base-${VARIANT} AS final

RUN uv pip install \
    docx2txt \
    llama-index-embeddings-deepinfra \
    llama-index-embeddings-huggingface \
    llama-index-embeddings-ollama \
    llama-index-node-parser-docling \
    llama-index-readers-docling \
    llama-index-readers-file \
    llama-index-vector-stores-qdrant \
    PyMuPDF \
    pymupdf4llm \
    qdrant-client \
    tqdm

COPY build_document_store.py /app/

ENV PYTHONUNBUFFERED=1

WORKDIR /app

ENTRYPOINT ["python3", "/app/build_document_store.py"]

CMD ["--help"]
