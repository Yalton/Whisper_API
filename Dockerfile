FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set a working directory
WORKDIR /app

# Both CTranslate2 (Whisper) and PyTorch (Qwen) use the CUDA runtime in this
# image. Qwen uses PyTorch SDPA by default, avoiding a compile-time dependency
# on FlashAttention while retaining 4090 acceleration.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy your requirements.txt file to the container
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code to the container
COPY . /app

# Command to run your application
CMD ["python3", "main.py"]
