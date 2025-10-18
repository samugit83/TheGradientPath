#!/bin/bash

# Install Script
# This script runs during the installation phase
# Handles all system installations and environment setup

echo "Starting install.sh script..."

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Change to application directory
cd /home/ubuntu/DevergoLabs

log_message "Installing required system dependencies..."

# Update package list
log_message "Updating package list..."
sudo apt-get update

# Install essential packages
log_message "Installing essential packages..."
sudo apt-get install -y curl jq

log_message "Installing Python and network monitoring dependencies..."

# Install Python 3.8+ and development packages needed for network monitoring
if ! command -v python3 &> /dev/null; then
    log_message "Python3 not found, installing Python3..."
    sudo apt-get install -y python3 python3-pip python3-venv python3-dev
else
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    log_message "Python3 $PYTHON_VERSION is already installed"
    
    # Ensure pip and venv are installed
    sudo apt-get install -y python3-pip python3-venv python3-dev
fi

# Install system dependencies for packet capture
log_message "Installing packet capture dependencies..."
sudo apt-get install -y libpcap-dev build-essential tcpdump

log_message "Installing Node.js if not present..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    log_message "Node.js not found, installing Node.js 22..."
    
    # Install Node.js 22 using NodeSource repository (compatible with Next.js 14.x)
    curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    # Verify installation
    if command -v node &> /dev/null; then
        log_message "Node.js $(node --version) installed successfully"
        log_message "NPM $(npm --version) installed successfully"
    else
        log_message "Failed to install Node.js"
        exit 1
    fi
else
    NODE_VERSION=$(node --version)
    log_message "Node.js $NODE_VERSION is already installed"
    
    # Check if we need to upgrade/downgrade to Node.js 22
    NODE_MAJOR_VERSION=$(node --version | cut -d'.' -f1 | sed 's/v//')
    if [ "$NODE_MAJOR_VERSION" != "22" ]; then
        log_message "Current Node.js version ($NODE_VERSION) is not compatible with Next.js 14.x"
        log_message "Installing Node.js 22 for compatibility..."
        
        # Install Node.js 22 using NodeSource repository
        curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
        sudo apt-get install -y nodejs
        
        # Verify installation
        if command -v node &> /dev/null; then
            log_message "Node.js $(node --version) installed successfully"
            log_message "NPM $(npm --version) installed successfully"
        else
            log_message "Failed to install Node.js 22"
            exit 1
        fi
    fi
fi

log_message "Installing nginx if not present..."

# Install nginx if not present
if ! command -v nginx &> /dev/null; then
    log_message "Nginx not found, installing nginx..."
    sudo apt-get install -y nginx
    log_message "Nginx installed successfully"
else
    log_message "Nginx is already installed"
fi

# Enable nginx to start automatically on boot
log_message "Enabling nginx to start on boot..."
sudo systemctl enable nginx
if systemctl is-enabled --quiet nginx; then
    log_message "✓ Nginx enabled for automatic startup on boot"
else
    log_message "✗ Failed to enable nginx for automatic startup"
    exit 1
fi

log_message "Configuring nginx for DevergoLabs..."

# Create nginx configuration for DevergoLabs
sudo tee /etc/nginx/conf.d/devergolabs.conf > /dev/null <<EOF
server {
    listen 80 default_server;
    server_name www.devergolabs.com;  
    large_client_header_buffers 16 64k;
    client_max_body_size 20M;

    location / {
        proxy_pass http://localhost:3000;                                             
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;

        proxy_buffers 32 128k;
        proxy_buffer_size 128k;
    }
}
EOF

log_message "Nginx configuration for DevergoLabs created successfully"

# Remove default nginx site to avoid conflicts
log_message "Removing default nginx sites and configurations..."

# Remove all default sites that could conflict
sudo rm -f /etc/nginx/sites-enabled/default
sudo rm -f /etc/nginx/sites-available/default
sudo rm -f /etc/nginx/conf.d/default.conf

# Also remove any other potential default configurations
if [ -f "/etc/nginx/nginx.conf" ]; then
    # Backup original nginx.conf
    sudo cp /etc/nginx/nginx.conf /etc/nginx/nginx.conf.backup
    
    # Remove any default server blocks from main nginx.conf
    sudo sed -i '/server {/,/}/d' /etc/nginx/nginx.conf
fi

log_message "All default nginx configurations removed"

# Verify no conflicting server blocks exist
log_message "Checking for conflicting nginx configurations..."
if sudo nginx -T 2>/dev/null | grep -q "listen.*80.*default_server" | grep -v "devergolabs.conf"; then
    log_message "⚠️ Found conflicting default_server configurations"
    sudo nginx -T 2>/dev/null | grep -A5 -B5 "listen.*80.*default_server"
fi

# Test nginx configuration
log_message "Testing nginx configuration..."
if sudo nginx -t; then
    log_message "✓ Nginx configuration is valid"
    
    # Double-check that our devergolabs.conf is the only default_server
    log_message "Verifying devergolabs.conf is the only default_server..."
    DEFAULT_SERVER_COUNT=$(sudo nginx -T 2>/dev/null | grep -c "listen.*80.*default_server" || echo "0")
    
    if [ "$DEFAULT_SERVER_COUNT" -eq 1 ]; then
        log_message "✓ Only one default_server configuration found (devergolabs.conf)"
    elif [ "$DEFAULT_SERVER_COUNT" -eq 0 ]; then
        log_message "⚠️ No default_server found - this might cause issues"
    else
        log_message "✗ Multiple default_server configurations found ($DEFAULT_SERVER_COUNT)"
        log_message "This will cause nginx conflicts. Listing all default_server blocks:"
        sudo nginx -T 2>/dev/null | grep -A10 -B2 "listen.*80.*default_server"
        exit 1
    fi
else
    log_message "✗ Nginx configuration has errors"
    exit 1
fi

log_message "Installing PM2 process manager..."

# Ensure PM2 is installed globally
if ! command -v pm2 &> /dev/null; then
    log_message "Installing PM2 globally..."
    sudo npm install -g pm2
    log_message "PM2 installed successfully"
else
    log_message "PM2 is already installed"
fi

log_message "Setting proper file permissions..."

# Set ownership to ubuntu user
sudo chown -R ubuntu:ubuntu /home/ubuntu/DevergoLabs

# Set proper permissions
sudo chmod -R 755 /home/ubuntu/DevergoLabs

# Create logs directory if it doesn't exist
mkdir -p /home/ubuntu/DevergoLabs/logs

log_message "Verifying installations..."

# Verify all required tools are available
if command -v node &> /dev/null && command -v npm &> /dev/null && command -v nginx &> /dev/null && command -v pm2 &> /dev/null && command -v python3 &> /dev/null; then
    log_message "✓ All required tools installed successfully:"
    log_message "  - Node.js: $(node --version)"
    log_message "  - NPM: $(npm --version)"
    log_message "  - Nginx: $(nginx -v 2>&1)"
    log_message "  - PM2: $(pm2 --version)"
    log_message "  - Python3: $(python3 --version)"
else
    log_message "✗ Some required tools are missing"
    exit 1
fi

log_message "install.sh script completed successfully"


