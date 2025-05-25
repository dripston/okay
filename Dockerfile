FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Cartopy, etc.
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY . .

RUN pip install --upgrade pip

# Split install: torch core first, then PyG add-ons
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
RUN pip install torch-geometric==2.6.1 torch_cluster==1.6.3 torch_scatter==2.1.2 torch_sparse==0.6.18 torch_spline_conv==1.2.2

RUN pip install -r requirements.txt

EXPOSE 10000

CMD ["gunicorn", "weather.app:app", "--bind", "0.0.0.0:10000"]
