# ================================
# Stage 1: Build dependencies
# ================================
FROM python:3.10-slim as deps

WORKDIR /app

ENV TZ=UTC DEBIAN_FRONTEND=noninteractive
ENV POETRY_VERSION=1.8.3

# Cài đặt system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libc-dev libffi-dev libgmp-dev libmpfr-dev libmpc-dev \
    ffmpeg sox git \
    && rm -rf /var/lib/apt/lists/*

# Cài Poetry
RUN pip install --no-cache-dir poetry==$POETRY_VERSION

# Copy file dependency trước (để cache)
COPY pyproject.toml poetry.lock* ./

# Cài đặt dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi

# Cài đặt PyTorch CPU version
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# ================================
# Stage 2: Runtime image
# ================================
FROM python:3.10-slim

WORKDIR /app

# Copy toàn bộ Python + dependencies từ stage deps
COPY --from=deps /usr/local /usr/local

# Copy source code (chỉ thay đổi khi sửa code)
COPY ./viettts /app/viettts
COPY ./samples /app/samples
COPY ./web /app/web
COPY ./README.md /app/

# Tạo thư mục pretrained models
RUN mkdir -p /app/pretrained-models

EXPOSE 8298

CMD ["viettts", "server", "--host", "0.0.0.0", "--port", "8298"]
