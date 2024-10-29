# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-distutils \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support first
RUN pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
COPY requirements.txt .
# Remove torch-related packages from requirements.txt if they exist
RUN grep -v "torch" requirements.txt > requirements_no_torch.txt || true
RUN pip install --no-cache-dir -r requirements_no_torch.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI server with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
