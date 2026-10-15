# Проверка Scribe на Linux (процессор, без окна) - Scribe 2.4.
#   docker build -t scribe-linux -f tests/linux/Dockerfile .
#   docker run --rm -v "$PWD":/app -v <папка офлайн-пакета>:/models scribe-linux \
#       python3 tests/measure_platform.py --tag linux-docker
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip ffmpeg espeak-ng ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
COPY requirements-cli.txt /tmp/requirements-cli.txt
RUN pip install --no-cache-dir -r /tmp/requirements-cli.txt
WORKDIR /app
