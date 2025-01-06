# Use the official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s ~/.local/bin/poetry /usr/local/bin/poetry

# Set the working directory
WORKDIR /app

# Copy only the pyproject.toml and poetry.lock first (for caching dependencies)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# Copy the rest of the project
COPY . /app

# Default command to start the container
CMD ["poetry", "run", "python", "test_gpu.py"]
