#!/bin/bash

# BeforeInstall Hook - AWS CodeDeploy
# Backup existing project and prepare for new installation

set -e

# Configuration
PROJECT_NAME="ml-attack-predictor"
REMOTE_PATH="/opt/$PROJECT_NAME"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [BeforeInstall]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR] [BeforeInstall]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] [BeforeInstall]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO] [BeforeInstall]${NC} $1"
}

log "Starting BeforeInstall phase..."

# Stop existing service if running
if systemctl is-active --quiet ml-attack-predictor 2>/dev/null; then
    log "Stopping existing ml-attack-predictor service..."
    sudo systemctl stop ml-attack-predictor || true
fi

# Backup existing project if it exists
if [ -d "$REMOTE_PATH" ]; then
    log "Found existing project. Creating backup..."
    BACKUP_PATH="/opt/ml-attack-predictor.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Remove old backups (keep only last 3)
    sudo find /opt -name "ml-attack-predictor.backup.*" -type d -mtime +7 -exec rm -rf {} \; 2>/dev/null || true
    
    # Create backup
    sudo cp -r "$REMOTE_PATH" "$BACKUP_PATH"
    log "Existing project backed up to $BACKUP_PATH"
    
    # Remove existing project
    sudo rm -rf "$REMOTE_PATH"
    log "Existing project removed"
else
    log "No existing project found"
fi

# Create destination directory with correct permissions
log "Creating destination directory..."
sudo mkdir -p "$REMOTE_PATH"
sudo chown ubuntu:ubuntu "$REMOTE_PATH"

log "BeforeInstall phase completed successfully ✅"
