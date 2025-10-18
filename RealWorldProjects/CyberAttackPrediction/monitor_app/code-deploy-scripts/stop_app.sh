#!/bin/bash

# Stop Application Script
# This script stops the running application services

echo "Starting stop_app.sh script..."

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_message "Stopping application services..."

# Stop PM2 processes using the config file
if [ -f "/home/ubuntu/DevergoLabs/pm2-config.json" ]; then
    log_message "Stopping PM2 processes from config..."
    cd /home/ubuntu/DevergoLabs
    
    # Stop processes defined in pm2-config.json
    pm2 stop pm2-config.json || log_message "Failed to stop PM2 processes from config"
    
    # Also stop all PM2 processes as fallback
    pm2 stop all || log_message "No PM2 processes to stop"
    
    log_message "PM2 processes stopped"
else
    log_message "PM2 config file not found, stopping all PM2 processes..."
    pm2 stop all || log_message "No PM2 processes to stop"
fi

log_message "Stopping nginx..."

# Stop nginx
if systemctl is-active --quiet nginx; then
    sudo systemctl stop nginx
    log_message "Nginx stopped successfully"
else
    log_message "Nginx is not running"
fi

# Stop Network Monitor Agent
log_message "Stopping Network Monitor Agent..."
if systemctl is-active --quiet network-monitor; then
    sudo systemctl stop network-monitor
    log_message "Network Monitor Agent stopped successfully"
else
    log_message "Network Monitor Agent is not running"
fi

# Wait a moment for graceful shutdown
sleep 2

log_message "stop_app.sh script completed successfully"
