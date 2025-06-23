# Base image with CUDA 12.4.1 and Ubuntu 22.04
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install basic tools from bootstramp.sh
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    git \
    curl \
    gnupg \
    wget \
    unzip \
    htop \
    tmux \
    python3-pip \
    bash-completion && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Install oh-my-bash for a better shell experience
RUN git clone https://github.com/ohmybash/oh-my-bash.git /root/.oh-my-bash && \
    cp /root/.oh-my-bash/templates/bashrc.osh-template /root/.bashrc && \
    sed -i 's/^OSH_THEME=.*/OSH_THEME="font"/' /root/.bashrc

# Set bash as the default shell to automatically use oh-my-bash
SHELL ["/bin/bash", "--login", "-c"]

# Configure Git. These can be overridden at build time.
ARG GIT_USER_NAME="Yexi Jiang"
ARG GIT_USER_EMAIL="2237303+yxjiang@users.noreply.github.com"
RUN git config --global user.name "${GIT_USER_NAME}" && \
    git config --global user.email "${GIT_USER_EMAIL}" && \
    git config --global init.defaultBranch main && \
    git config --global core.editor vim && \
    git config --global color.ui auto

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set up the workspace directory
WORKDIR /workspace

# Copy requirements.txt and install Python packages.
# NOTE: The requested torch, torchao, torchtune versions are not available on PyPI.
# Using nightly PyTorch build for CUDA 12.4 and latest versions for other packages.
COPY requirements.txt .

# Install PyTorch nightly with CUDA 12.4 support.
# This is used because the requested PyTorch v2.6.0 is not yet released.
# The nightly build is the closest available option.
RUN pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Install other dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Copy the rest of the application code into the image
COPY . .

# Expose the default port for Ollama
EXPOSE 11434

# Start ollama in the background and open a bash session.
CMD ["/bin/bash", "-c", "ollama serve & /bin/bash"] 