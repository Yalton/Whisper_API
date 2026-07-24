# Use NVIDIA CUDA with cuDNN as the base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set a working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libcublas11 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Copy your requirements.txt file to the container
COPY requirements.txt /app/requirements.txt

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code to the container
COPY . /app

# Command to run your application
CMD ["python3", "main.py"]
