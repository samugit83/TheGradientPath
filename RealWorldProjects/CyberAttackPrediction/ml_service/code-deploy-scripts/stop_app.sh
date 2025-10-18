#!/bin/bash

# ApplicationStop Hook - AWS CodeDeploy
# Stop the ML Attack Predictor service

set -e

# Configuration
SERVICE_NAME="ml-attack-predictor"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [ApplicationStop]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR] [ApplicationStop]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] [ApplicationStop]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO] [ApplicationStop]${NC} $1"
}

log "Starting ApplicationStop phase..."

# Check if service exists
if systemctl list-unit-files | grep -q "^${SERVICE_NAME}.service"; then
    log "Found ${SERVICE_NAME} service"
    
    # Check if service is running
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        log "Service is running, stopping it..."
        sudo systemctl stop "$SERVICE_NAME"
        
        # Wait for service to stop completely
        timeout=30
        while systemctl is-active --quiet "$SERVICE_NAME" && [ $timeout -gt 0 ]; do
            log "Waiting for service to stop... ($timeout seconds remaining)"
            sleep 2
            ((timeout-=2))
        done
        
        if systemctl is-active --quiet "$SERVICE_NAME"; then
            warning "Service did not stop gracefully, forcing stop..."
            sudo systemctl kill "$SERVICE_NAME" || true
        else
            log "Service stopped successfully"
        fi
    else
        log "Service is not running"
    fi
    
    # Disable service to prevent auto-start during deployment
    if systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
        log "Disabling service..."
        sudo systemctl disable "$SERVICE_NAME" || warning "Could not disable service"
    fi
else
    log "No existing ${SERVICE_NAME} service found"
fi

# Kill any remaining processes on port 8080
log "Checking for processes on port 8080..."
if netstat -tlnp 2>/dev/null | grep -q ":8080 "; then
    warning "Found processes on port 8080, attempting to terminate..."
    sudo fuser -k 8080/tcp 2>/dev/null || true
    sleep 2
fi

log "ApplicationStop phase completed successfully ✅"
