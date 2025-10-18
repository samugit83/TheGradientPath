#!/bin/bash

# Before Install Script
# This script runs before the application installation
# Focuses on cleanup and preparation only

echo "Starting before_install.sh script..."

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_message "Performing pre-installation cleanup..."

# Verify CodeDeploy agent is running (should be installed via Launch Template user data)
log_message "Checking CodeDeploy agent status..."

# First check if the agent service exists and is installed
if ! systemctl list-unit-files | grep -q codedeploy-agent; then
    log_message "✗ CodeDeploy agent service not found - agent may not be installed"
    log_message "Attempting to install CodeDeploy agent..."
    
    # Try to install the agent if it's missing
    cd /home/ubuntu
    wget https://aws-codedeploy-eu-south-1.s3.eu-south-1.amazonaws.com/latest/install
    chmod +x ./install
    sudo ./install auto
    sudo systemctl start codedeploy-agent
    sudo systemctl enable codedeploy-agent
fi

# Check if agent is active
if systemctl is-active --quiet codedeploy-agent; then
    log_message "✓ CodeDeploy agent is running"
    
    # Additional connectivity test
    log_message "Testing CodeDeploy agent connectivity..."
    if sudo service codedeploy-agent status | grep -q "running"; then
        log_message "✓ CodeDeploy agent service is healthy"
    else
        log_message "⚠️ CodeDeploy agent service may have issues"
        sudo service codedeploy-agent status
    fi
else
    log_message "⚠️ CodeDeploy agent is not running - attempting to start"
    
    # Try to start the agent
    sudo systemctl start codedeploy-agent
    sleep 5
    
    if systemctl is-active --quiet codedeploy-agent; then
        log_message "✓ CodeDeploy agent started successfully"
    else
        log_message "✗ Failed to start CodeDeploy agent"
        log_message "Agent logs:"
        sudo tail -n 20 /var/log/aws/codedeploy-agent/codedeploy-agent.log || log_message "No agent logs found"
        log_message "System logs:"
        sudo journalctl -u codedeploy-agent --no-pager -l | tail -20
        
        # This is a warning, not a failure - let deployment continue
        log_message "⚠️ CodeDeploy agent issues detected but continuing with deployment"
    fi
fi

# Show agent configuration for debugging
log_message "CodeDeploy agent configuration:"
if [ -f "/etc/codedeploy-agent/conf/codedeployagent.yml" ]; then
    cat /etc/codedeploy-agent/conf/codedeployagent.yml
else
    log_message "No CodeDeploy agent configuration file found"
fi

log_message "Stopping existing PM2 processes..."

# Check if PM2 is installed and running
if command -v pm2 &> /dev/null; then
    # Stop all PM2 processes
    pm2 stop all || log_message "No PM2 processes to stop"
    
    # Delete all PM2 processes
    pm2 delete all || log_message "No PM2 processes to delete"
    
    log_message "PM2 processes stopped and deleted"
else
    log_message "PM2 is not installed, skipping PM2 cleanup"
fi

log_message "Stopping nginx service..."

# Stop nginx service
if systemctl is-active --quiet nginx; then
    sudo systemctl stop nginx
    log_message "Nginx stopped successfully"
else
    log_message "Nginx is not running or not installed"
fi

# Stop network monitor service if it's running
log_message "Stopping Network Monitor Agent if running..."
if systemctl is-active --quiet network-monitor; then
    sudo systemctl stop network-monitor
    log_message "Network Monitor Agent stopped successfully"
else
    log_message "Network Monitor Agent is not running or not installed"
fi

# Clean up old deployment directory if it exists
if [ -d "/home/ubuntu/DevergoLabs" ]; then
    log_message "Removing old deployment directory..."
    sudo rm -rf /home/ubuntu/DevergoLabs || log_message "Failed to remove old directory, continuing..."
fi

log_message "before_install.sh script completed successfully"
