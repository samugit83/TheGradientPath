#!/bin/bash

# ApplicationStart Hook - AWS CodeDeploy
# Setup and start the ML application

set -e

# Configuration
PROJECT_NAME="ml-attack-predictor"
REMOTE_PATH="/opt/$PROJECT_NAME"
SERVICE_NAME="ml-attack-predictor"
EC2_USER="ubuntu"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] [ApplicationStart]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR] [ApplicationStart]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING] [ApplicationStart]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO] [ApplicationStart]${NC} $1"
}

log "Starting ApplicationStart phase..."

# Change to project directory
cd "$REMOTE_PATH"

# Create systemd service file
log "Creating systemd service file..."
sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << EOF
[Unit]
Description=ML Attack Predictor API
After=network.target

[Service]
Type=simple
User=$EC2_USER
Group=$EC2_USER
WorkingDirectory=$REMOTE_PATH
Environment=PATH=$REMOTE_PATH/venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=$REMOTE_PATH
ExecStart=$REMOTE_PATH/venv/bin/gunicorn --bind 0.0.0.0:8080 --workers 4 --timeout 120 --access-logfile $REMOTE_PATH/logs/access.log --error-logfile $REMOTE_PATH/logs/error.log ml_ec2_service:app
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd daemon
log "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Enable service to start on boot
log "Enabling service to start on boot..."
sudo systemctl enable "$SERVICE_NAME"

# Start the service
log "Starting ML Attack Predictor service..."
sudo systemctl start "$SERVICE_NAME"

# Wait a moment for service to initialize
sleep 5

# Check service status
if systemctl is-active --quiet "$SERVICE_NAME"; then
    log "Service started successfully ✅"
    
    # Show service status
    info "Service status:"
    sudo systemctl status "$SERVICE_NAME" --no-pager -l || true
    
    # Show recent logs
    info "Recent service logs:"
    sudo journalctl -u "$SERVICE_NAME" -n 10 --no-pager || true
else
    error "Failed to start service"
    
    # Show error logs for debugging
    error "Service logs:"
    sudo journalctl -u "$SERVICE_NAME" -n 20 --no-pager || true
    
    exit 1
fi

log "ApplicationStart phase completed successfully ✅"
