# pathology_inference/Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY . .
ENV MODEL_PATH=/models/qwen2.5-vl
ENV BASE_TILE_PATH=/data/tiles
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]