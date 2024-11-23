# Build stage
FROM python:3.10-slim-buster AS builder

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Install build dependencies and create virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/venv/bin/pip install -e .

# Final stage
FROM python:3.10-slim-buster

# Create non-root user
RUN useradd -m -u 1000 appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Set environment variable for Python path
ENV PATH="/opt/venv/bin:$PATH"

# Switch to non-root user
USER appuser

# Command to run the application
CMD ["python", "app.py"]