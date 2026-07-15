FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY common ./common
COPY data ./data
COPY project_b_synth ./project_b_synth

ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["python", "-m", "project_b_synth.demo.app"]
