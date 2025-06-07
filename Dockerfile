# Use official PyTorch image with CUDA 12.1 support
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Copy everything from the current directory into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


