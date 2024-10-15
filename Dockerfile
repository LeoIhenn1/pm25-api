# --- Base Stage ---
FROM python:3.10-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV POETRY_VERSION=1.5.1
ENV PYTHONPATH=/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install "poetry==$POETRY_VERSION"

# Copy pyproject.toml and poetry.lock
COPY pyproject.toml poetry.lock ./

# Configure Poetry
RUN poetry config virtualenvs.create false

# Install dependencies for both production and development
RUN poetry install --no-interaction --no-ansi

# Copy application and test code
COPY app ./app
COPY tests ./tests

# Copy the data directory
COPY data ./data

# --- Final Stage (Running the App) ---
FROM base AS final

# Expose the port
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Test Stage ---
FROM base AS test

# Run the tests
CMD ["pytest", "tests/"]
