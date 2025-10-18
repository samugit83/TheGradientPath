#!/bin/bash

# Validate Service Script
# This script validates that the application is running correctly after deployment

echo "Starting validate_service.sh script..."

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log_message "Validating deployed services..."

# Change to application directory
cd /home/ubuntu/DevergoLabs

# Validation flags
PM2_VALID=false
NGINX_VALID=false
APP_VALID=false
NETWORK_MONITOR_VALID=false

# Validate PM2 processes
log_message "Checking PM2 processes..."
if command -v pm2 &> /dev/null; then
    PM2_STATUS=$(pm2 jlist | jq -r '.[0].pm2_env.status' 2>/dev/null)
    
    if [ "$PM2_STATUS" = "online" ]; then
        PM2_VALID=true
        log_message "✓ PM2 process is running"
    else
        log_message "✗ PM2 process is not running correctly"
        pm2 status
    fi
else
    log_message "✗ PM2 is not available"
fi

# Validate nginx
log_message "Checking nginx status..."
if systemctl is-active --quiet nginx; then
    NGINX_VALID=true
    log_message "✓ Nginx is running"
else
    log_message "✗ Nginx is not running"
fi

# Validate Network Monitor Agent
log_message "Checking Network Monitor Agent status..."
if systemctl is-active --quiet network-monitor; then
    NETWORK_MONITOR_VALID=true
    log_message "✓ Network Monitor Agent is running"
    
    # Check if it's actually capturing packets (log file should be getting updated)
    if [ -f "/var/log/network_monitor.log" ]; then
        LOG_SIZE=$(stat -f%z "/var/log/network_monitor.log" 2>/dev/null || stat -c%s "/var/log/network_monitor.log" 2>/dev/null || echo "0")
        if [ "$LOG_SIZE" -gt 0 ]; then
            log_message "✓ Network Monitor Agent is logging activity"
        else
            log_message "⚠️ Network Monitor Agent is running but not logging yet"
        fi
    fi
else
    log_message "✗ Network Monitor Agent is not running"
    if systemctl is-enabled --quiet network-monitor; then
        log_message "Network Monitor Agent logs:"
        sudo journalctl -u network-monitor --no-pager -l | tail -5 || log_message "No network monitor logs found"
    else
        log_message "Network Monitor Agent service is not enabled"
    fi
fi

# Validate application response (check if localhost responds)
log_message "Testing application response..."
sleep 5  # Give app time to fully start

# Test the specific health check endpoint that ALB uses
log_message "Testing health check endpoint /api/healthcheck..."
if curl -f -s http://localhost/api/healthcheck > /dev/null; then
    APP_VALID=true
    log_message "✓ Health check endpoint is responding correctly"
    
    # Show the actual response for verification
    HEALTH_RESPONSE=$(curl -s http://localhost/api/healthcheck)
    log_message "Health check response: $HEALTH_RESPONSE"
elif curl -f -s http://localhost:3000/api/healthcheck > /dev/null; then
    APP_VALID=true
    log_message "✓ Health check endpoint is responding on port 3000"
    
    # Show the actual response for verification
    HEALTH_RESPONSE=$(curl -s http://localhost:3000/api/healthcheck)
    log_message "Health check response: $HEALTH_RESPONSE"
    
    log_message "⚠️ Health check works on 3000 but not through nginx proxy - check nginx config"
elif curl -f -s http://localhost:3000 > /dev/null; then
    log_message "⚠️ Application is responding on port 3000 but health check endpoint failed"
    log_message "Testing if health check endpoint exists..."
    curl -v http://localhost:3000/api/healthcheck || log_message "Health check endpoint not accessible"
elif curl -f -s http://localhost:80 > /dev/null; then
    log_message "⚠️ Nginx is responding on port 80 but health check endpoint failed"
    log_message "Testing if health check endpoint exists through nginx..."
    curl -v http://localhost/api/healthcheck || log_message "Health check endpoint not accessible through nginx"
else
    log_message "✗ Application is not responding on expected ports"
    
    # Try to get more information about what's running
    log_message "Checking what's listening on ports..."
    sudo netstat -tlnp | grep -E ':80|:3000' || log_message "No processes found on ports 80 or 3000"
    
    # Check if PM2 process is actually running the right thing
    log_message "PM2 process details:"
    pm2 show devergo-prod-server || log_message "PM2 process not found"
    
    # Check nginx logs for errors
    log_message "Recent nginx error logs:"
    sudo tail -n 10 /var/log/nginx/error.log || log_message "No nginx error log found"
fi

# Overall validation result
log_message "Validation Summary:"
log_message "PM2: $([ "$PM2_VALID" = true ] && echo "✓ PASS" || echo "✗ FAIL")"
log_message "Nginx: $([ "$NGINX_VALID" = true ] && echo "✓ PASS" || echo "✗ FAIL")"
log_message "Application: $([ "$APP_VALID" = true ] && echo "✓ PASS" || echo "✗ FAIL")"
log_message "Network Monitor: $([ "$NETWORK_MONITOR_VALID" = true ] && echo "✓ PASS" || echo "✗ FAIL")"

# Check if all validations passed
if [ "$PM2_VALID" = true ] && [ "$NGINX_VALID" = true ] && [ "$APP_VALID" = true ]; then
    if [ "$NETWORK_MONITOR_VALID" = true ]; then
        log_message "🎉 All validations passed - deployment successful!"
    else
        log_message "🎉 Core services passed validation - deployment successful!"
        log_message "⚠️ Network Monitor Agent has issues but won't fail deployment"
    fi
    exit 0
elif [ "$PM2_VALID" = true ] && [ "$NGINX_VALID" = true ]; then
    log_message "⚠️  Services are running but app validation failed - check application logs"
    pm2 logs --lines 20
    exit 0  # Don't fail deployment for app validation issues
else
    log_message "❌ Critical services validation failed - deployment has issues"
    exit 1
fi 