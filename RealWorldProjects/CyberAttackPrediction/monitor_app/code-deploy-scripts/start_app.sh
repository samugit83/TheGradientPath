#!/bin/bash

# Start Application Script
# This script starts the application services using PM2 and nginx

echo "Starting start_app.sh script..."

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Change to application directory
cd /home/ubuntu/DevergoLabs

log_message "Starting application services..."

# Install and configure pm2-logrotate for daily log rotation
log_message "Setting up log rotation..."
if ! pm2 list | grep -q "pm2-logrotate"; then
    log_message "Installing pm2-logrotate module..."
    pm2 install pm2-logrotate
    
    # Configure log rotation settings
    pm2 set pm2-logrotate:max_size 100M          # Rotate when file reaches 100MB
    pm2 set pm2-logrotate:retain 20              # Keep 20 files (20 days)
    pm2 set pm2-logrotate:compress true          # Compress old logs with gzip
    pm2 set pm2-logrotate:dateFormat YYYY-MM-DD # Date format for rotated files
    pm2 set pm2-logrotate:rotateModule true      # Also rotate PM2 module logs
    pm2 set pm2-logrotate:workerInterval 30      # Check every 30 seconds
    pm2 set pm2-logrotate:rotateInterval '0 0 * * *'  # Rotate daily at midnight
    
    log_message "pm2-logrotate configured successfully"
else
    log_message "pm2-logrotate already installed"
fi

# Start PM2 processes from config file
if [ -f "/home/ubuntu/DevergoLabs/pm2-config.json" ]; then
    log_message "Starting PM2 processes from config file..."
    
    # Start processes defined in pm2-config.json
    pm2 start pm2-config.json
    
    if [ $? -eq 0 ]; then
        log_message "PM2 processes started successfully"
        
        # Save PM2 process list for auto-restart on reboot
        log_message "Saving PM2 process list for auto-restart..."
        pm2 save
        
        # Configure PM2 startup script for automatic restart on boot
        log_message "Configuring PM2 startup script..."
        
        # Generate startup script (this creates the systemd service)
        STARTUP_SCRIPT=$(pm2 startup systemd -u ubuntu --hp /home/ubuntu | grep 'sudo' | head -1)
        
        if [ -n "$STARTUP_SCRIPT" ]; then
            log_message "Executing PM2 startup configuration..."
            eval "$STARTUP_SCRIPT" || log_message "PM2 startup script execution completed (may already be configured)"
            
            # Verify PM2 startup is configured
            if systemctl is-enabled --quiet pm2-ubuntu; then
                log_message "✓ PM2 startup service enabled successfully"
            else
                log_message "⚠️ PM2 startup service may not be properly enabled"
            fi
        else
            log_message "PM2 startup already configured or no configuration needed"
        fi
        
        # Double-check by saving again to ensure persistence
        pm2 save
        log_message "✓ PM2 configuration saved for automatic restart on reboot"
        
    else
        log_message "Failed to start PM2 processes from config file"
        exit 1
    fi
else
    log_message "PM2 config file not found at /home/ubuntu/DevergoLabs/pm2-config.json"
    exit 1
fi

log_message "Starting nginx..."

# Ensure nginx configuration is clean before starting
log_message "Verifying nginx configuration before starting..."
if ! sudo nginx -t; then
    log_message "✗ Nginx configuration test failed"
    sudo nginx -T
    exit 1
fi

# Start nginx service
if systemctl is-enabled --quiet nginx; then
    # Stop nginx first to ensure clean restart
    log_message "Stopping nginx for clean restart..."
    sudo systemctl stop nginx || log_message "Nginx wasn't running"
    
    # Wait a moment
    sleep 2
    
    # Start nginx with fresh configuration
    log_message "Starting nginx with DevergoLabs configuration..."
    sudo systemctl start nginx
    
    if systemctl is-active --quiet nginx; then
        log_message "Nginx started successfully with DevergoLabs configuration"
        
        # Verify the configuration is working
        log_message "Testing nginx is serving on port 80..."
        sleep 3  # Give nginx time to fully start
        
        if curl -f -s -o /dev/null http://localhost:80; then
            log_message "✓ Nginx is responding on port 80"
        else
            log_message "⚠️ Nginx is not responding on port 80"
            log_message "Nginx error logs:"
            sudo tail -n 20 /var/log/nginx/error.log || log_message "No nginx error log found"
        fi
    else
        log_message "Failed to start nginx"
        log_message "Nginx status:"
        sudo systemctl status nginx --no-pager
        log_message "Nginx error logs:"
        sudo tail -n 20 /var/log/nginx/error.log || log_message "No nginx error log found"
        exit 1
    fi
else
    log_message "Nginx is not enabled, enabling and starting..."
    sudo systemctl enable nginx
    sudo systemctl start nginx
    
    if systemctl is-active --quiet nginx; then
        log_message "Nginx enabled and started successfully with DevergoLabs configuration"
    else
        log_message "Failed to enable and start nginx"
        exit 1
    fi
fi

# Wait a moment for services to fully start
sleep 3

# Verify services are running
log_message "Verifying services status..."

# Check PM2 processes
pm2 status

# Check nginx status
sudo systemctl status nginx --no-pager --lines=0

# Verify nginx configuration is working
log_message "Testing nginx proxy configuration..."
if curl -f -s http://localhost > /dev/null; then
    log_message "✓ Nginx proxy is responding correctly"
else
    log_message "⚠️ Nginx proxy test failed - check if Next.js app is running on port 3000"
fi

# Start Network Monitor Agent after main services are running
log_message "Starting Network Monitor Agent..."
if systemctl is-enabled --quiet network-monitor; then
    sudo systemctl start network-monitor
    
    if systemctl is-active --quiet network-monitor; then
        log_message "✓ Network Monitor Agent started successfully"
    else
        log_message "⚠️ Failed to start Network Monitor Agent"
        log_message "Network monitor logs:"
        sudo journalctl -u network-monitor --no-pager -l | tail -10 || log_message "No network monitor logs found"
    fi
else
    log_message "⚠️ Network Monitor Agent service not enabled, skipping startup"
fi

# Final verification of auto-startup configuration
log_message "Verifying auto-startup configuration..."

if systemctl is-enabled --quiet nginx; then
    log_message "✓ Nginx is configured to start automatically on boot"
else
    log_message "✗ Nginx auto-startup is NOT configured"
fi

if systemctl is-enabled --quiet pm2-ubuntu 2>/dev/null; then
    log_message "✓ PM2 is configured to start automatically on boot"
else
    log_message "⚠️ PM2 auto-startup may not be configured (check manually with: systemctl status pm2-ubuntu)"
fi

log_message "🎉 All services are now configured for automatic restart on instance reboot!"
log_message "start_app.sh script completed successfully"
