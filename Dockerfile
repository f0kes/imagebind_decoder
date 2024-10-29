# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-distutils \
    python3.9-dev \  
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN python --version

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA support first
RUN pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 torchaudio==2.0.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install torchdata==0.6.0 pytorch-lightning==2.3.0
# Install other dependencies
COPY requirements.txt .
# Remove torch-related packages from requirements.txt if they exist
RUN grep -v "torch" requirements.txt > requirements_no_torch.txt || true
RUN pip install --no-cache-dir -r requirements_no_torch.txt

RUN apt-get update && apt-get install -y git && apt-get clean
RUN pip install git+https://github.com/omegalabsinc/ImageBind@e2bfdec716b1c1d511d6bea806227b2b3dfcadee#egg=imagebind-0.0.1
RUN pip install "clip @ git+https://github.com/openai/CLIP.git"
RUN pip install regex==2017.4.5

# Copy application code
COPY . .


# Expose port
EXPOSE 8000

# Run FastAPI server with Uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
