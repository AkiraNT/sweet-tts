# Sử dụng base image CPU thay vì CUDA
FROM python:3.10-slim

ENV TZ=UTC DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Thiết lập Poetry
ENV POETRY_VERSION=1.8.3
ENV POETRY_CACHE_DIR=/tmp/poetry_cache
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VIRTUALENVS_IN_PROJECT=true
ENV POETRY_VIRTUALENVS_CREATE=true
ENV POETRY_REQUESTS_TIMEOUT=15

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc g++ libc-dev libffi-dev libgmp-dev libmpfr-dev libmpc-dev \
    ffmpeg \
    sox \
    git \
    && apt clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Cài đặt Poetry (cache layer riêng)
RUN pip install --no-cache-dir poetry==${POETRY_VERSION}

# Copy pyproject.toml và poetry.lock trước để tận dụng Docker cache
COPY pyproject.toml poetry.lock* /app/

# Cài đặt dependencies trước (layer này sẽ được cache nếu dependencies không thay đổi)
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi \
    && rm -rf /tmp/poetry_cache

# Cài đặt PyTorch CPU version cụ thể (override GPU version nếu có)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Copy source code (chỉ copy khi code thay đổi)
COPY ./viettts /app/viettts
COPY ./samples /app/samples
COPY ./web /app/web
COPY ./README.md /app/

# Cài đặt package ở development mode
RUN pip install -e . && pip cache purge

# Tạo directory cho pretrained models
RUN mkdir -p /app/pretrained-models

# Expose port
EXPOSE 8298

# Default command
CMD ["viettts", "server", "--host", "0.0.0.0", "--port", "8298"]