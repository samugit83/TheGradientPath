#!/bin/bash

# ValidateService Hook - AWS CodeDeploy
# Test the deployed ML Attack Predictor service

set -e

# Configuration
SERVICE_NAME="ml-attack-predictor"
SERVICE_URL="http://localhost:8080"
MAX_RETRIES=30
RETRY_INTERVAL=2

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ValidateService] $1"
}

error() {
    echo "[ERROR] [ValidateService] $1" >&2
}

warning() {
    echo "[WARNING] [ValidateService] $1"
}

info() {
    echo "[INFO] [ValidateService] $1"
}

# Function to check if port is listening
check_port() {
    local port=$1
    ss -tlnp | grep ":${port} " >/dev/null 2>&1
}

# Function to show port status
show_port_status() {
    local port=$1
    info "Checking port $port status:"
    ss -tlnp | grep ":${port}" || echo "Port $port not found in ss output"
}

log "Starting ValidateService phase..."

# Check if service is running
log "Checking service status..."
if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    error "Service $SERVICE_NAME is not running"
    sudo systemctl status "$SERVICE_NAME" --no-pager || true
    exit 1
fi

log "Service is running"

# Test service connectivity with retries
log "Testing service connectivity..."
retry_count=0

while [ $retry_count -lt $MAX_RETRIES ]; do
    log "Attempt $((retry_count + 1))/$MAX_RETRIES - Testing health endpoint..."
    
    if curl -f -s "$SERVICE_URL/health" > /dev/null 2>&1; then
        log "Health endpoint responding"
        break
    else
        retry_count=$((retry_count + 1))
        if [ $retry_count -lt $MAX_RETRIES ]; then
            log "Health endpoint not ready, waiting ${RETRY_INTERVAL}s..."
            sleep $RETRY_INTERVAL
        else
            error "Health endpoint failed after $MAX_RETRIES attempts"
            
            # Show service logs for debugging
            error "Recent service logs:"
            sudo journalctl -u "$SERVICE_NAME" -n 20 --no-pager || true
            
            # Check if port is listening
            show_port_status 8080
            
            exit 1
        fi
    fi
done

# Test health endpoint response
log "Testing health endpoint response..."
health_response=$(curl -s "$SERVICE_URL/health" 2>/dev/null || echo "")
if [ -n "$health_response" ]; then
    log "Health endpoint response: $health_response"
else
    warning "Empty health response"
fi

# Test if the service returns expected content
log "Testing service endpoints..."

# Test root endpoint
if curl -f -s "$SERVICE_URL/" > /dev/null 2>&1; then
    log "Root endpoint accessible"
else
    warning "Root endpoint not accessible"
fi

# Verify service is listening on correct port
log "Verifying service is listening on port 8080..."
if check_port 8080; then
    log "Service listening on port 8080"
else
    error "Service not listening on port 8080"
    show_port_status 8080
    exit 1
fi

# Final service status check
log "Final service status check..."
service_status=$(systemctl is-active "$SERVICE_NAME" 2>/dev/null || echo "inactive")
if [ "$service_status" = "active" ]; then
    log "Service is active and healthy"
else
    error "Service is not active: $service_status"
    exit 1
fi

# Display deployment summary
log "Service validation completed successfully!"
info "Service Information:"
info "  - Service Status: Active"
info "  - Health Check: $SERVICE_URL/health"
info "  - Service URL: $SERVICE_URL"

# Show recent logs
info "Recent service logs (last 5 lines):"
sudo journalctl -u "$SERVICE_NAME" -n 5 --no-pager || true

log "ValidateService phase completed successfully"
