#!/bin/bash

# Install Hook - AWS CodeDeploy
# This phase is handled automatically by CodeDeploy
# Files are copied from source to destination as defined in appspec.yml

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [Install]${NC} $1"
}

log "Install phase - CodeDeploy is handling file copy automatically"
log "Files are being copied to /opt/ml-attack-predictor as specified in appspec.yml"
log "Install phase completed ✅"
