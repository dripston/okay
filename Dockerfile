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

COPY requirements-torch.txt .
COPY requirements.txt .
COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements-torch.txt
RUN pip install -r requirements.txt

EXPOSE 10000

# If your Flask app is in weather/app.py and called "app"
CMD ["gunicorn", "weather.app:app", "--bind", "0.0.0.0:10000"]
