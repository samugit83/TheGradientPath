#!/bin/bash

# After Install Script
# This script runs after the application installation is complete
# Performs final setup and validation

echo "Starting after_install.sh script..."

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Change to application directory
cd /home/ubuntu/DevergoLabs

log_message "Performing post-installation tasks..."

# Set final permissions
log_message "Setting final file permissions..."
sudo chown -R ubuntu:ubuntu /home/ubuntu/DevergoLabs
sudo chmod +x /home/ubuntu/DevergoLabs/code-deploy-scripts/*.sh

# Create necessary directories
log_message "Creating required directories..."
mkdir -p /home/ubuntu/DevergoLabs/logs
mkdir -p /home/ubuntu/DevergoLabs/tmp

# Set environment variables if .env file exists
if [ -f "/home/ubuntu/DevergoLabs/.env.production" ]; then
    log_message "Production environment file found"
    cp /home/ubuntu/DevergoLabs/.env.production /home/ubuntu/DevergoLabs/.env
elif [ -f "/home/ubuntu/DevergoLabs/.env.example" ]; then
    log_message "Copying example environment file"
    cp /home/ubuntu/DevergoLabs/.env.example /home/ubuntu/DevergoLabs/.env
    log_message "Please update .env file with production values"
fi

# Clean up npm cache and reinstall dependencies with correct Node.js version
log_message "Cleaning npm cache and reinstalling dependencies..."
npm cache clean --force

# Remove existing node_modules to ensure clean install with correct Node.js version
if [ -d "node_modules" ]; then
    log_message "Removing existing node_modules directory..."
    rm -rf node_modules
fi

# Check if package-lock.json exists (required for npm ci)
if [ ! -f "package-lock.json" ]; then
    log_message "⚠️ package-lock.json not found, using npm install instead of npm ci"
    
    # Use npm install for production dependencies
    log_message "Installing production dependencies with Node.js $(node --version)..."
    npm install --production
    
    if [ $? -eq 0 ]; then
        log_message "✓ Dependencies installed successfully with Node.js $(node --version)"
    else
        log_message "✗ Failed to install dependencies with npm install"
        log_message "NPM error details:"
        npm install --production 2>&1 | tail -20
        exit 1
    fi
else
    # Use npm ci for production dependencies (faster and more reliable)
    log_message "Installing production dependencies with Node.js $(node --version) using npm ci..."
    npm ci --production
    
    if [ $? -eq 0 ]; then
        log_message "✓ Dependencies installed successfully with Node.js $(node --version)"
    else
        log_message "✗ Failed to install dependencies with npm ci"
        log_message "NPM error details:"
        npm ci --production 2>&1 | tail -20
        log_message "Trying fallback with npm install..."
        
        # Fallback to npm install
        npm install --production
        if [ $? -eq 0 ]; then
            log_message "✓ Dependencies installed successfully with npm install fallback"
        else
            log_message "✗ Both npm ci and npm install failed"
            exit 1
        fi
    fi
fi

# Verify deployment files
log_message "Verifying deployment files..."

required_files=(
    "package.json"
    "pm2-config.json"
    ".next"
    "node_modules"
)

for file in "${required_files[@]}"; do
    if [ -e "/home/ubuntu/DevergoLabs/$file" ]; then
        log_message "✓ $file found"
    else
        log_message "✗ $file missing - deployment may have issues"
    fi
done

# Log deployment information
log_message "Deployment completed at $(date)"
log_message "Application directory: /home/ubuntu/DevergoLabs"
log_message "Node.js version: $(node --version)"
log_message "NPM version: $(npm --version)"

# Create deployment marker file
echo "Deployment completed at $(date)" > /home/ubuntu/DevergoLabs/.deployment-info
echo "Node.js version: $(node --version)" >> /home/ubuntu/DevergoLabs/.deployment-info
echo "NPM version: $(npm --version)" >> /home/ubuntu/DevergoLabs/.deployment-info

log_message "Setting up Network Monitor Agent..."

# Set up network monitoring agent
if [ -d "/home/ubuntu/DevergoLabs/network_agent" ]; then
    log_message "Setting up Network Monitor Agent Python environment..."
    
    # Install Python 3.11 if not already installed (required for CICFlowMeter 0.2.0)
    if ! command -v python3.11 &> /dev/null; then
        log_message "Installing Python 3.11 for CICFlowMeter compatibility..."
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt-get update
        sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
        
        if command -v python3.11 &> /dev/null; then
            log_message "✓ Python 3.11 $(python3.11 --version) installed successfully"
        else
            log_message "✗ Failed to install Python 3.11"
            exit 1
        fi
    else
        log_message "✓ Python 3.11 $(python3.11 --version) is already installed"
    fi
    
    # Create network monitor directory and virtual environment
    sudo mkdir -p /opt/network_monitor
    cd /opt/network_monitor
    
    # Create virtual environment with Python 3.11 (required for CICFlowMeter 0.2.0)
    log_message "Creating Python 3.11 virtual environment for CICFlowMeter..."
    sudo python3.11 -m venv venv
    
    # Change ownership to ubuntu user
    sudo chown -R ubuntu:ubuntu /opt/network_monitor
    
    # Activate virtual environment and install dependencies
    source venv/bin/activate
    
    # Verify we're using Python 3.11 in the virtual environment
    VENV_PYTHON_VERSION=$(python --version)
    log_message "Virtual environment Python version: $VENV_PYTHON_VERSION"
    
    pip install --upgrade pip
    
    # Copy network monitor files
    cp /home/ubuntu/DevergoLabs/network_agent/network_monitor_agent.py /opt/network_monitor/
    cp /home/ubuntu/DevergoLabs/network_agent/requirements.txt /opt/network_monitor/
    
    # Install Python dependencies (including CICFlowMeter 0.2.0)
    log_message "Installing network monitor Python dependencies with Python 3.11..."
    pip install -r requirements.txt
    
    if [ $? -eq 0 ]; then
        log_message "✓ CICFlowMeter and dependencies installed successfully with Python 3.11"
    else
        log_message "✗ Failed to install CICFlowMeter dependencies"
        log_message "Pip error details:"
        pip install -r requirements.txt 2>&1 | tail -20
        exit 1
    fi
    
    # Make network monitor executable
    chmod +x /opt/network_monitor/network_monitor_agent.py
    
    # Create log directories and files
    sudo mkdir -p /var/log/network_monitor
    sudo chown ubuntu:ubuntu /var/log/network_monitor
    
    # Create CICFlowMeter-specific log files
    sudo touch /var/log/network_monitor_cicflowmeter.log
    sudo touch /var/log/network_metrics_cicflowmeter.csv
    sudo touch /var/log/attacks_cicflowmeter.json
    sudo chown ubuntu:ubuntu /var/log/network_monitor_cicflowmeter.log
    sudo chown ubuntu:ubuntu /var/log/network_metrics_cicflowmeter.csv
    sudo chown ubuntu:ubuntu /var/log/attacks_cicflowmeter.json
    
    # Create legacy log files for compatibility
    touch /var/log/network_monitor.log
    touch /var/log/attacks.json
    touch /var/log/network_metrics.csv
    
    # Create systemd service for network monitor
    log_message "Creating systemd service for network monitor..."
    sudo tee /etc/systemd/system/network-monitor.service > /dev/null <<EOF
[Unit]
Description=Network Monitor Agent (CICFlowMeter)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/network_monitor
Environment=PATH=/opt/network_monitor/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EnvironmentFile=-/opt/network_monitor/.env
ExecStart=/opt/network_monitor/venv/bin/python network_monitor_agent.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/network_monitor/service.log
StandardError=append:/var/log/network_monitor/error.log

[Install]
WantedBy=multi-user.target
EOF
    
    # Create environment configuration file for network monitor
    log_message "Creating network monitor configuration file..."
    sudo tee /opt/network_monitor/.env > /dev/null <<EOF
# Network Monitor Agent Configuration (CICFlowMeter Version)
# Uncomment and modify these values as needed

# NETWORK_INTERFACE=ens5
# ML_ENDPOINT=http://15.160.68.117:8080/predict
# FLOW_TIMEOUT=60
# CAPTURE_WINDOW=30
# MAX_PACKETS_PER_WINDOW=5000
EOF
    
    # Set ownership of config file
    sudo chown ubuntu:ubuntu /opt/network_monitor/.env
    
    # Enable the service (but don't start it yet - will start in start_app.sh)
    sudo systemctl daemon-reload
    sudo systemctl enable network-monitor
    
    log_message "✓ Network Monitor Agent setup completed with Python 3.11 and CICFlowMeter 0.2.0"
    
    deactivate
else
    log_message "⚠️ Network agent directory not found, skipping network monitor setup"
fi

log_message "after_install.sh script completed successfully"