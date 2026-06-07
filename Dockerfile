FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Install system dependencies (git, unzip)
RUN apt-get update && apt-get install -y git unzip && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Keep container running to allow docker exec for multiple phases
CMD ["tail", "-f", "/dev/null"]
