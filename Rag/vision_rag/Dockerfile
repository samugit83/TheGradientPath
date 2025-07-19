FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including pgvector requirements
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    poppler-utils \
    postgresql-server-dev-16 \
    build-essential \
    git \
    && git clone --depth 1 https://github.com/pgvector/pgvector \
    && cd pgvector && make && make install \
    && cd .. && rm -rf pgvector \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for images and logs
RUN mkdir -p docs img logs

# Make scripts executable
RUN chmod +x init_db.py entrypoint.sh

# Expose port (if you plan to add a web interface later)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python init_db.py --check || exit 1

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]

# Default command
CMD ["python", "main.py"] 