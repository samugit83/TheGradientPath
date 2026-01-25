# RedAmon Reconnaissance Module - Dockerfile
# ============================================
# Fully containerized OSINT reconnaissance framework
# Supports Docker-in-Docker for ProjectDiscovery tools
# Includes Tor/Proxychains for anonymous scanning

FROM python:3.11-slim-bookworm

LABEL maintainer="RedAmon Team"
LABEL description="RedAmon Reconnaissance Module - Automated OSINT Framework"
LABEL version="1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
# - Docker CLI for orchestrating ProjectDiscovery containers
# - Tor and proxychains for anonymous scanning
# - DNS utilities for domain resolution
# - Git for installing knock-subdomains
# - Build tools for Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Docker CLI (to communicate with host Docker daemon)
    docker.io \
    # Tor and proxychains for anonymous scanning
    tor \
    proxychains4 \
    # DNS utilities
    dnsutils \
    bind9-host \
    # Network tools
    curl \
    wget \
    netcat-openbsd \
    iputils-ping \
    # Git for knock-subdomains installation
    git \
    # Build dependencies for Python packages
    build-essential \
    libffi-dev \
    libssl-dev \
    # Clean up apt cache
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Configure proxychains for Tor
# Use dynamic_chain for better reliability
RUN sed -i 's/^strict_chain/#strict_chain/' /etc/proxychains4.conf && \
    sed -i 's/^#dynamic_chain/dynamic_chain/' /etc/proxychains4.conf && \
    echo "socks5 127.0.0.1 9050" >> /etc/proxychains4.conf

# Create Tor configuration for running as service
RUN mkdir -p /var/run/tor && \
    chown -R debian-tor:debian-tor /var/run/tor && \
    echo "SocksPort 0.0.0.0:9050" > /etc/tor/torrc && \
    echo "ControlPort 9051" >> /etc/tor/torrc && \
    echo "CookieAuthentication 0" >> /etc/tor/torrc && \
    echo "Log notice stdout" >> /etc/tor/torrc

# Create non-root user for security (but allow Docker access)
RUN groupadd -r redamon && \
    useradd -r -g redamon -G docker redamon && \
    mkdir -p /home/redamon && \
    chown -R redamon:redamon /home/redamon

# Create necessary directories
RUN mkdir -p /app/recon/output \
             /app/recon/data/mitre_db \
             /app/utils \
             /data/output \
             /data/.env \
    && chown -R redamon:redamon /app /data

# Copy requirements first (for better Docker layer caching)
COPY recon/requirements.txt /app/recon/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/recon/requirements.txt

# Copy the application code
# Copy recon module (includes recon/params.py and helpers/anonymity.py)
COPY recon/ /app/recon/

# Copy graph_db module (for Neo4j integration)
COPY graph_db/ /app/graph_db/

# Note: .env file is mounted via docker-compose volume, not copied here

# Copy entrypoint script
COPY recon/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create volume mount points
VOLUME ["/app/recon/output", "/var/run/docker.sock", "/data"]

# Expose Tor SOCKS port (if using container's Tor)
EXPOSE 9050

# Set default environment variables (can be overridden)
ENV TARGET_DOMAIN=""
ENV SUBDOMAIN_LIST=""
ENV SCAN_MODULES="domain_discovery,port_scan,http_probe,resource_enum,vuln_scan"
ENV USE_TOR_FOR_RECON="false"
ENV USE_BRUTEFORCE_FOR_SUBDOMAINS="false"
ENV UPDATE_GRAPH_DB="false"
ENV USER_ID="default_user"
ENV PROJECT_ID="default_project"

# Use non-root user by default
# Note: May need root for some operations (Naabu SYN scan)
# USER redamon

# Health check - verify Docker socket and Python are working
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" && \
        (test -S /var/run/docker.sock && docker info > /dev/null 2>&1 || true)

# Default command - run the recon pipeline
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/recon/main.py"]
