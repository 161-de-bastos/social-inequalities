# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.5.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install essential packages and Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    wget \
    nano \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/161-de-bastos/social-inequalities.git

# Make sure all dependancies are available
RUN pip install -r requirements.txt

# Create python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Create a non-root user
RUN useradd -m -s /bin/bash up \
    && chown -R up:up /workspace

USER up

# Set default command to bash
CMD ["/bin/bash"]