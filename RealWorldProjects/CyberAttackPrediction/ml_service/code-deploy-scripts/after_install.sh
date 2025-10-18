#!/bin/bash

# AfterInstall Hook - AWS CodeDeploy
# Setup server environment, Python dependencies, and system configuration

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
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [AfterInstall]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR] [AfterInstall]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] [AfterInstall]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO] [AfterInstall]${NC} $1"
}

log "Starting AfterInstall phase..."

# Change to project directory
cd "$REMOTE_PATH"

# Update system packages
log "Updating system packages..."
sudo apt update -y

# Install required system packages
log "Installing Python and development tools..."
sudo apt install -y python3 python3-pip python3-venv python3-dev build-essential libssl-dev libffi-dev curl

# Verify Python installation
log "Verifying Python installation..."
python3 --version
pip3 --version

# Setup Python virtual environment
log "Creating Python virtual environment..."
if [ -d "venv" ]; then
    rm -rf venv
fi
python3 -m venv venv

log "Activating virtual environment and installing dependencies..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU version for cost optimization)
log "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies
log "Installing Flask and Gunicorn..."
pip install flask gunicorn

log "Installing ML and data processing libraries..."
pip install river pandas numpy scikit-learn

log "Installing utility libraries..."
pip install dill pyyaml requests

# Install from requirements.txt if available
if [ -f "requirements.txt" ]; then
    log "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    warning "No requirements.txt found, skipping..."
fi

# Create necessary directories
log "Creating necessary directories..."
mkdir -p logs
mkdir -p models

# Create __init__.py files for Python modules
if [ -d "modules" ]; then
    touch modules/__init__.py
fi

# Test imports to ensure everything is working
log "Testing critical imports..."
python -c "
try:
    import flask
    import torch
    import pandas
    import numpy
    print('✅ All critical imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Configure firewall
log "Configuring firewall..."
sudo ufw allow 8080/tcp || warning "UFW not available or already configured"

# Set proper permissions
log "Setting file permissions..."
sudo chown -R ubuntu:ubuntu "$REMOTE_PATH"
find "$REMOTE_PATH" -type f -name "*.py" -exec chmod 644 {} \;
find "$REMOTE_PATH" -type f -name "*.sh" -exec chmod 755 {} \;

log "AfterInstall phase completed successfully ✅"
