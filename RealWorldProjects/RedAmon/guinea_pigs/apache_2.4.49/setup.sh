#!/bin/bash
# Install Docker and run the vulnerable Apache container
set -e

echo "=== Installing Docker ==="

# Detect OS and install Docker
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
fi

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose
elif [ "$OS" = "amzn" ] || [ "$OS" = "fedora" ] || [ "$OS" = "rhel" ]; then
    sudo dnf install -y docker
    sudo curl -sL "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

sudo systemctl start docker
sudo systemctl enable docker

echo "=== Cleaning up Docker space ==="
cd "$(dirname "$0")"
sudo docker-compose down --volumes --remove-orphans 2>/dev/null || true
sudo docker system prune -a -f --volumes

echo "=== Building and starting container ==="
sudo docker-compose up -d --build

echo ""
echo "=== DONE ==="
echo "Server: http://$(curl -s ifconfig.me 2>/dev/null || echo '<IP>'):8080"
echo "Health: http://<IP>:8080/health"
